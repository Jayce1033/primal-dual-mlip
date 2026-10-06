function _solve_weighted_lbfgs(theta_init::AbstractVector, A_train, y_train, train_info,
                                A_test, y_test, test_info;
                                a, b, max_iter::Int=1000, g_tol::Real=1e-4,
                                f_reltol::Real=1e-6, l2_reg::Real=1e-8,
                                verbose::Bool=true, eval_every::Int=10,
                                print_every::Int=50)
    _validate_objective_weights(a, b, l2_reg)
    max_iter > 0 || throw(ArgumentError("max_iter must be positive"))
    eval_every > 0 || throw(ArgumentError("eval_every must be positive"))
    print_every > 0 || throw(ArgumentError("print_every must be positive"))
    all(v -> isfinite(v) && v >= 0, (g_tol, f_reltol)) ||
        throw(ArgumentError("Optimizer tolerances must be finite and nonnegative"))
    length(theta_init) == size(A_train, 2) == size(A_test, 2) ||
        throw(DimensionMismatch("Training, test and coefficient feature counts differ"))
    all(isfinite, theta_init) || throw(ArgumentError("Initial coefficients must be finite"))

    training = _objective_blocks(A_train, y_train, train_info)
    testing = _objective_blocks(A_test, y_test, test_info)
    history_test_LE = Float64[]
    history_test_LF = Float64[]
    history_train_loss = Float64[]
    history_iters = Int[]

    function fg!(F, G, theta)
        value = _weighted_objective_gradient!(G, theta, training; a, b, gamma=l2_reg)
        return F === nothing ? nothing : value
    end

    function record!(iter, theta)
        test_losses = _component_losses(theta, testing)
        train_loss = _weighted_objective_gradient!(nothing, theta, training; a, b, gamma=l2_reg)
        # The final Optim minimizer can differ from the last trace state, so
        # replace the last entry when its iteration number already exists.
        if !isempty(history_iters) && history_iters[end] == iter
            history_test_LE[end] = test_losses.L_E
            history_test_LF[end] = test_losses.L_F
            history_train_loss[end] = train_loss
        else
            push!(history_iters, iter)
            push!(history_test_LE, test_losses.L_E)
            push!(history_test_LF, test_losses.L_F)
            push!(history_train_loss, train_loss)
        end
        return nothing
    end

    function trace_callback(trace_or_state)
        state = trace_or_state isa AbstractVector ? trace_or_state[end] : trace_or_state
        iter = state.iteration
        if iter == 0 || iter % eval_every == 0
            record!(iter, state.metadata["x"])
        end
        if verbose && (iter == 0 || iter % print_every == 0)
            @printf("Iter %6d | Training objective: %.6e | Gradient norm: %.3e\n",
                    iter, state.value, state.g_norm)
        end
        return false
    end

    # Keep only scalar histories: storing an extended trace would retain a
    # complete coefficient vector at every iteration of long research runs.
    options = Optim.Options(g_abstol=g_tol, f_reltol=f_reltol, iterations=max_iter,
                            store_trace=false, extended_trace=true,
                            show_trace=false, show_every=1, callback=trace_callback)
    method = Optim.LBFGS(linesearch=Optim.LineSearches.HagerZhang())
    if verbose
        @info "Starting L-BFGS" energy_weight=a force_weight=b l2_reg=l2_reg max_iter=max_iter
    end
    started = time()
    record!(0, theta_init)
    result = Optim.optimize(Optim.only_fg!(fg!), copy(theta_init), method, options)
    theta = copy(Optim.minimizer(result))
    record!(Optim.iterations(result), theta)
    elapsed = time() - started
    if verbose
        @info "L-BFGS finished" converged=Optim.converged(result) iterations=Optim.iterations(result)
    end

    return Dict{String,Any}(
        "theta" => theta,
        "optim_result" => result,
        "history_test_L_E" => history_test_LE,
        "history_test_L_F" => history_test_LF,
        "history_train_loss" => history_train_loss,
        "history_iters" => history_iters,
        "energy_weight" => a,
        "force_weight" => b,
        "l2_reg" => l2_reg,
        "iterations" => Optim.iterations(result),
        "elapsed_seconds" => elapsed,
        "converged" => Optim.converged(result),
    )
end

"""Refine coefficients with fixed `lambda_fixed` using `L_E + lambda*L_F + gamma*R`.

Matrices use observations × features. Supply the multiplier returned by PD.
"""
function solve_only_lbfgs(theta_init::AbstractVector, lambda_fixed::Real,
                          A_train, y_train, train_info, A_test, y_test, test_info;
                          f_reltol::Real=0.0, kwargs...)
    isfinite(lambda_fixed) && lambda_fixed >= 0 ||
        throw(ArgumentError("lambda_fixed must be finite and nonnegative"))
    result = _solve_weighted_lbfgs(theta_init, A_train, y_train, train_info,
                                  A_test, y_test, test_info; a=1.0, b=lambda_fixed, f_reltol, kwargs...)
    result["lambda"] = lambda_fixed
    return result
end

"""Fit static weighting `alpha*L_E + (1-alpha)*L_F + gamma*R` with L-BFGS.

Matrices use observations × features. Each alpha should start from the same
independently supplied initialization, rather than the preceding fit.
"""
function solve_static_lbfgs(theta_init::AbstractVector, alpha::Real,
                            A_train, y_train, train_info, A_test, y_test, test_info; kwargs...)
    isfinite(alpha) && 0 <= alpha <= 1 || throw(ArgumentError("alpha must lie in [0, 1]"))
    result = _solve_weighted_lbfgs(theta_init, A_train, y_train, train_info,
                                  A_test, y_test, test_info; a=alpha, b=1-alpha, kwargs...)
    result["alpha"] = alpha
    return result
end
