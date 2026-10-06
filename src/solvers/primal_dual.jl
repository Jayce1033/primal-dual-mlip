"""Parameters for primal-dual fitting with a bounded force multiplier.

The primal and dual optimizers are independent Adam rules. `tol` and `patience`
apply to the gradient stopping condition. Additional stopping criteria monitor
force loss, stagnation, and multiplier stability.
"""
struct ConstrainedPDSolver

    primal_opt_rule::Optimisers.AbstractRule

    dual_opt_rule::Optimisers.AbstractRule

    max_iter::Int
    tol::Float64
    verbose::Bool

    epsilon_F::Float64
    lambda_F_0::Float64
    lambda_F_max::Float64
    dual_update_freq::Int

    patience::Int

    initial_learning_rate::Float64
    lr_decay_step::Int
    lr_decay_factor::Float64
end

function ConstrainedPDSolver(;
    primal_opt_rule = Optimisers.Adam(1e-3),
    dual_opt_rule = Optimisers.Adam(1e-3),
    max_iter = 5000,
    tol = 1e-6,
    verbose = true,
    epsilon_F = 0.02^2,
    lambda_F_0 = 0.0,
    lambda_F_max = 1e4,
    dual_update_freq = 50,
    patience = 50,
    initial_learning_rate = 1e-3,
    lr_decay_step = 0,
    lr_decay_factor = 0.5
)
    max_iter > 0 || throw(ArgumentError("max_iter must be positive"))
    dual_update_freq > 0 || throw(ArgumentError("dual_update_freq must be positive"))
    patience > 0 || throw(ArgumentError("patience must be positive"))
    lr_decay_step >= 0 || throw(ArgumentError("lr_decay_step must be nonnegative"))
    all(v -> isfinite(v) && v >= 0, (tol, epsilon_F, lambda_F_0)) ||
        throw(ArgumentError("Tolerance, epsilon_F and initial lambda must be finite and nonnegative"))
    isfinite(lambda_F_max) && lambda_F_max > 0 ||
        throw(ArgumentError("lambda_F_max must be finite and positive"))
    lambda_F_0 <= lambda_F_max ||
        throw(ArgumentError("Initial multiplier must not exceed lambda_F_max"))
    all(v -> isfinite(v) && v > 0, (initial_learning_rate, lr_decay_factor)) ||
        throw(ArgumentError("Learning rate and decay factor must be finite and positive"))
    return ConstrainedPDSolver(
        primal_opt_rule, dual_opt_rule, max_iter, tol, verbose,
        epsilon_F, lambda_F_0, lambda_F_max,
        dual_update_freq,
        patience,
        initial_learning_rate,
        lr_decay_step,
        lr_decay_factor
    )
end

function parallel_copy_cols!(dest::AbstractMatrix, src::AbstractMatrix, inds::AbstractVector)

    Threads.@threads for i in 1:length(inds)
        col_idx = inds[i]

        @inbounds @views copyto!(dest[:, i], src[:, col_idx])
    end
end

function parallel_copy_cols!(dest::AbstractVector, src::AbstractVector, inds::AbstractVector)
    Threads.@threads for i in 1:length(inds)
        idx = inds[i]

        @inbounds dest[i] = src[idx]
    end
end

"""Run the PD solver with features × observations matrices.

`batch_size_E` counts energy observations (one per labeled configuration),
whereas `batch_size_F` counts scalar force components. They are sampled
independently; `nothing` uses the complete corresponding block. The first
batch follows input order; permutations are drawn only when a cursor exhausts
its block, and incomplete tails are skipped.

Each primal update is followed by a full-training force evaluation whenever
a dual update or test evaluation is due. Dual updates use that new residual
and project the multiplier onto [0, lambda_F_max]. The force-target stopping
condition is checked only at dual updates after `warmup_steps`. No PD L2 term
is added; the gradient and stagnation stopping criteria are independent.

History `iteration` counts completed primal updates. Batch losses and
`grad_norm` use the coefficients at `train_iteration`; `lambda_F_before` is
the multiplier used by that gradient, and `lambda_F` is the updated value.
Test losses use post-update coefficients at `test_iteration` and are carried
forward between evaluations. `full_train_L_F` is the post-update full force
loss when `full_train_evaluated` is true, and NaN otherwise. The final row
contains full train/test losses at the returned coefficients, with NaN for
the gradient and its multiplier. `full_train` describes the training-loss
rows. `elapsed_seconds` uses a monotonic clock; threshold crossings should
use only rows with `test_evaluated=true`.
"""
function solve_pd2_hpc_complete1(
    solver::ConstrainedPDSolver,

    A_train::AbstractMatrix{T}, y_train::AbstractVector{T}, train_info,
    A_test::AbstractMatrix{T}, y_test::AbstractVector{T}, test_info;
    theta_init = nothing,
    lambda_F_init = nothing,
    primal_state_init = nothing,
    dual_state_init = nothing,
    current_lr_init = nothing,
    batch_size_E::Union{Int, Nothing} = nothing,
    batch_size_F::Union{Int, Nothing} = nothing,
    eval_every::Int = 50,
    warmup_steps::Int = 2000
) where T <: AbstractFloat

    validate_observations(A_train, y_train, train_info; orientation=:columns)
    validate_observations(A_test, y_test, test_info; orientation=:columns)
    size(A_train, 1) == size(A_test, 1) || throw(DimensionMismatch("Training and test feature counts differ"))
    eval_every > 0 || throw(ArgumentError("eval_every must be positive"))
    warmup_steps >= 0 || throw(ArgumentError("warmup_steps must be nonnegative"))
    for (name, batch_size, count) in (("batch_size_E", batch_size_E, length(train_info.E_inds)),
                                      ("batch_size_F", batch_size_F, length(train_info.F_inds)))
        isnothing(batch_size) || 1 <= batch_size <= count ||
            throw(ArgumentError("$name must be nothing or in 1:$count"))
    end
    if theta_init !== nothing
        length(theta_init) == size(A_train, 1) || throw(DimensionMismatch("Initial coefficient count differs from features"))
        all(isfinite, theta_init) || throw(ArgumentError("Initial coefficients must be finite"))
    end
    if lambda_F_init !== nothing
        isfinite(lambda_F_init) && 0 <= lambda_F_init <= solver.lambda_F_max ||
            throw(ArgumentError("Initial multiplier must lie in [0, lambda_F_max]"))
    end
    if current_lr_init !== nothing
        isfinite(current_lr_init) && current_lr_init > 0 ||
            throw(ArgumentError("Initial learning rate must be finite and positive"))
    end
    started = time_ns()

    num_features, total_samples = size(A_train)

    A_E_trn = view(A_train, :, train_info.E_inds)
    y_E_trn = view(y_train, train_info.E_inds)
    w_E_trn = train_info.W_E
    N_E_trn = length(y_E_trn)

    A_F_trn = view(A_train, :, train_info.F_inds)
    y_F_trn = view(y_train, train_info.F_inds)
    N_F_trn = length(y_F_trn)

    A_E_tst = view(A_test, :, test_info.E_inds)
    y_E_tst = view(y_test, test_info.E_inds)
    w_E_tst = test_info.W_E
    A_F_tst = view(A_test, :, test_info.F_inds)
    y_F_tst = view(y_test, test_info.F_inds)

    theta = isnothing(theta_init) ? zeros(T, num_features) : copy(theta_init)
    lambda_arr = isnothing(lambda_F_init) ? T[solver.lambda_F_0] : T[lambda_F_init]
    current_lr = isnothing(current_lr_init) ? T(solver.initial_learning_rate) : T(current_lr_init)

    primal_state = isnothing(primal_state_init) ? Optimisers.setup(solver.primal_opt_rule, theta) : deepcopy(primal_state_init)
    dual_state = isnothing(dual_state_init) ? Optimisers.setup(solver.dual_opt_rule, lambda_arr) : deepcopy(dual_state_init)
    Optimisers.adjust!(primal_state, current_lr)

    bs_E = isnothing(batch_size_E) ? N_E_trn : batch_size_E
    bs_F = isnothing(batch_size_F) ? N_F_trn : batch_size_F

    buf_A_E = Matrix{T}(undef, num_features, bs_E)
    buf_y_E = Vector{T}(undef, bs_E)
    buf_w_E = Vector{T}(undef, bs_E)

    buf_A_F = Matrix{T}(undef, num_features, bs_F)
    buf_y_F = Vector{T}(undef, bs_F)

    pred_E_buf = Vector{T}(undef, bs_E)
    resid_E_buf = Vector{T}(undef, bs_E)

    pred_F_buf = Vector{T}(undef, bs_F)
    resid_F_buf = Vector{T}(undef, bs_F)

    grad_E_buf = Vector{T}(undef, num_features)
    grad_F_buf = Vector{T}(undef, num_features)
    grad_theta_buf = Vector{T}(undef, num_features)

    perm_E = collect(1:N_E_trn)
    cursor_E = 1
    perm_F = collect(1:N_F_trn)
    cursor_F = 1

    history = Dict{String,Any}(
        "train_L_E" => T[], "train_L_F" => T[],
        "test_L_E"  => T[], "test_L_F"  => T[],
        "lambda_F"  => T[], "grad_norm" => T[], "lr" => T[],
        "iteration" => Int[], "test_evaluated" => Bool[], "full_train" => Bool[],
        "train_iteration" => Int[], "test_iteration" => Int[],
        "lambda_F_before" => T[], "full_train_L_F" => T[],
        "full_train_evaluated" => Bool[],
        "elapsed_seconds" => Float64[]
    )

    sizehint!(history["train_L_F"], solver.max_iter)

    best_L_F_observed = T(Inf)
    stagnation_counter = 0
    smooth_L_F = T(0.0)
    alpha_smooth = T(0.01)

    best_test_LF_observed = T(Inf)
    test_stagnation_counter = 0
    test_patience_limit = 20

    min_lambda_for_stopping = T(0.2)
    patience_threshold = 2000
    improvement_tol = 1e-5
    lambda_safety_cap = T(1e5)
    patience_counter = 0

    current_L_E_tst = T(NaN)
    current_L_F_tst = T(NaN)
    test_iteration = 0

    current_L_F_trn_full = T(NaN)

    if solver.verbose
        @printf("%6s | %9s %9s | %9s %9s | %8s | %8s | %8s\n",
                "Iter", "Tr_LE", "Tr_LF", "Te_LE", "Te_LF", "Lam_F", "|Grad|", "LR")
    end

    iterations = 0
    stopping_reason = "max_iter"
    prior_gc_state = GC.enable(false)
    try
        for iter in 1:solver.max_iter
            iterations = iter

            if isnothing(batch_size_E)

                A_E_active = A_E_trn; y_E_active = y_E_trn; w_E_active = w_E_trn
                curr_N_E = N_E_trn
            else
                if cursor_E + bs_E - 1 > N_E_trn
                    Random.randperm!(perm_E)
                    cursor_E = 1
                end

                inds = view(perm_E, cursor_E:(cursor_E + bs_E - 1))
                cursor_E += bs_E

                parallel_copy_cols!(buf_A_E, A_E_trn, inds)
                parallel_copy_cols!(buf_y_E, y_E_trn, inds)
                parallel_copy_cols!(buf_w_E, w_E_trn, inds)
                A_E_active = buf_A_E; y_E_active = buf_y_E; w_E_active = buf_w_E
                curr_N_E = bs_E
            end

            if isnothing(batch_size_F)
                A_F_active = A_F_trn; y_F_active = y_F_trn
                curr_N_F = N_F_trn
            else
                if cursor_F + bs_F - 1 > N_F_trn
                    Random.randperm!(perm_F)
                    cursor_F = 1
                end
                inds = view(perm_F, cursor_F:(cursor_F + bs_F - 1))
                cursor_F += bs_F

                parallel_copy_cols!(buf_A_F, A_F_trn, inds)
                parallel_copy_cols!(buf_y_F, y_F_trn, inds)

                A_F_active = buf_A_F; y_F_active = buf_y_F
                curr_N_F = bs_F
            end

            if solver.lr_decay_step > 0 && iter % solver.lr_decay_step == 0
                current_lr *= T(solver.lr_decay_factor)
                Optimisers.adjust!(primal_state, current_lr)
            end

            mul!(pred_E_buf, transpose(A_E_active), theta)
            mul!(pred_F_buf, transpose(A_F_active), theta)

            @. resid_E_buf = (pred_E_buf - y_E_active) * w_E_active
            @. resid_F_buf = pred_F_buf - y_F_active

            L_E_batch = mean(abs2, resid_E_buf)
            L_F_batch = mean(abs2, resid_F_buf)

            is_batch_mode = !isnothing(batch_size_E) || !isnothing(batch_size_F)
            perform_eval = !is_batch_mode || (iter % eval_every == 0 || iter == 1)
            perform_dual_update = iter % solver.dual_update_freq == 0
            evaluate_full_force = perform_eval || perform_dual_update

            lambda_val = lambda_arr[1]

            @. resid_E_buf *= w_E_active

            mul!(grad_E_buf, A_E_active, resid_E_buf)
            rmul!(grad_E_buf, T(2.0 / curr_N_E))

            mul!(grad_F_buf, A_F_active, resid_F_buf)
            rmul!(grad_F_buf, T(2.0 / curr_N_F))

            @. grad_theta_buf = grad_E_buf + lambda_val * grad_F_buf

            grad_norm = norm(grad_theta_buf)

            primal_state, theta = Optimisers.update(primal_state, theta, grad_theta_buf)

            if evaluate_full_force
                pred_F_trn_full = transpose(A_F_trn) * theta
                current_L_F_trn_full = mean(abs2, pred_F_trn_full - y_F_trn)
            end

            if perform_eval

                pred_E_tst = transpose(A_E_tst) * theta
                pred_F_tst = transpose(A_F_tst) * theta

                current_L_E_tst = mean(abs2, (pred_E_tst - y_E_tst) .* w_E_tst)
                current_L_F_tst = mean(abs2, pred_F_tst - y_F_tst)
                test_iteration = iter

                is_warming_up = (iter < warmup_steps) || (lambda_arr[1] < min_lambda_for_stopping)
                if is_warming_up
                    best_test_LF_observed = current_L_F_tst
                    test_stagnation_counter = 0
                else
                    if current_L_F_tst < best_test_LF_observed * (1 - improvement_tol)
                        best_test_LF_observed = current_L_F_tst
                        test_stagnation_counter = 0
                    else
                        test_stagnation_counter += 1
                    end
                end
            end

            if perform_dual_update
                constraint_violation = current_L_F_trn_full - solver.epsilon_F
                grad_lambda = -1.0 * constraint_violation
                dual_state, lambda_arr = Optimisers.update(dual_state, lambda_arr, [grad_lambda])
                lambda_arr[1] = clamp(lambda_arr[1], T(0.0), T(solver.lambda_F_max))
            end

            push!(history["train_L_E"], L_E_batch)
            push!(history["train_L_F"], L_F_batch)
            push!(history["test_L_E"],  current_L_E_tst)
            push!(history["test_L_F"],  current_L_F_tst)
            push!(history["lambda_F"],  lambda_arr[1])
            push!(history["grad_norm"], grad_norm)
            push!(history["lr"],        current_lr)
            push!(history["iteration"], iter)
            push!(history["test_evaluated"], perform_eval)
            push!(history["full_train"], !is_batch_mode)
            push!(history["train_iteration"], iter - 1)
            push!(history["test_iteration"], test_iteration)
            push!(history["lambda_F_before"], lambda_val)
            push!(history["full_train_L_F"], evaluate_full_force ? current_L_F_trn_full : T(NaN))
            push!(history["full_train_evaluated"], evaluate_full_force)
            push!(history["elapsed_seconds"], (time_ns() - started) / 1e9)

            if solver.verbose && (iter % 100 == 0 || iter == 1)
                @printf("%6d | %.3e %.3e | %.3e %.3e | %.3e | %.3e | %.1e\n",
                        iter, L_E_batch, L_F_batch, current_L_E_tst, current_L_F_tst, lambda_arr[1], grad_norm, current_lr)
            end

            if iter == 1
                smooth_L_F = L_F_batch
            else
                smooth_L_F = (1 - alpha_smooth) * smooth_L_F + alpha_smooth * L_F_batch
            end

            if smooth_L_F < best_L_F_observed * (1 - improvement_tol)
                best_L_F_observed = smooth_L_F
                stagnation_counter = 0
            else
                stagnation_counter += 1
            end

            if grad_norm < solver.tol
                patience_counter += 1
            else
                patience_counter = 0
            end

            if iter % 1000 == 0
                GC.enable(true)
                GC.gc()
                GC.enable(false)
            end

            is_target_reached = perform_dual_update && iter > warmup_steps &&
                                current_L_F_trn_full <= solver.epsilon_F
            is_stagnated = (stagnation_counter >= patience_threshold) && (iter > 50000)
            is_exploded = lambda_val > lambda_safety_cap
            is_grad_converged = (patience_counter >= solver.patience) && (iter > 1000)
            is_test_stagnated = (test_stagnation_counter >= test_patience_limit) && (iter > 100000)

            if is_target_reached
                stopping_reason = "target_reached"
                solver.verbose && @info "Converged: Target epsilon reached."
                break
            elseif is_test_stagnated
                stopping_reason = "test_stagnation"
                solver.verbose && @warn "Early Stopping: Test set Error stagnated."
                break
            elseif is_stagnated
                stopping_reason = "train_stagnation"
                solver.verbose && @warn "Early Stopping: Train Loss stagnated."
                break
            elseif is_exploded
                stopping_reason = "lambda_safety_stop"
                solver.verbose && @warn "Early Stopping: Lambda exploded."
                break
            elseif is_grad_converged
                stopping_reason = "gradient_converged"
                solver.verbose && println("Converged: Gradient norm small.")
                break
            end
        end
    finally
        GC.enable(prior_gc_state)
    end

    lambda_val = lambda_arr[1]

    pred_E_trn_full = transpose(A_E_trn) * theta
    pred_F_trn_full = transpose(A_F_trn) * theta
    L_E_trn_final = mean(abs2, (pred_E_trn_full - y_E_trn) .* w_E_trn)
    L_F_trn_final = mean(abs2, pred_F_trn_full - y_F_trn)

    pred_E_tst = transpose(A_E_tst) * theta
    pred_F_tst = transpose(A_F_tst) * theta
    L_E_tst_final = mean(abs2, (pred_E_tst - y_E_tst) .* w_E_tst)
    L_F_tst_final = mean(abs2, pred_F_tst - y_F_tst)

    push!(history["train_L_E"], L_E_trn_final)
    push!(history["train_L_F"], L_F_trn_final)
    push!(history["test_L_E"],  L_E_tst_final)
    push!(history["test_L_F"],  L_F_tst_final)
    push!(history["lambda_F"],  lambda_val)
    push!(history["grad_norm"], NaN)
    push!(history["lr"], current_lr)
    push!(history["iteration"], iterations)
    push!(history["test_evaluated"], true)
    push!(history["full_train"], true)
    push!(history["train_iteration"], iterations)
    push!(history["test_iteration"], iterations)
    push!(history["lambda_F_before"], NaN)
    push!(history["full_train_L_F"], L_F_trn_final)
    push!(history["full_train_evaluated"], true)
    push!(history["elapsed_seconds"], (time_ns() - started) / 1e9)

    return Dict(
        "C" => theta,
        "lambda_F" => lambda_val,
        "history" => history,
        "primal_state" => primal_state,
        "dual_state" => dual_state,
        "current_lr" => current_lr,
        "iterations" => iterations,
        "stopping_reason" => stopping_reason,
        "elapsed_seconds" => (time_ns() - started) / 1e9
    )
end
