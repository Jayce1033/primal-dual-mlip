"""Validate observation indices and weights for an assembled ACE design matrix.

`orientation=:rows` means observations × features; `:columns` means features ×
observations. Energy weights are `1 / sqrt(N_atoms)`; force indices refer to
individual Cartesian components, not atoms or configurations.
"""
function validate_observations(A::AbstractMatrix, y::AbstractVector, info;
                               orientation::Symbol=:rows)
    orientation in (:rows, :columns) || throw(ArgumentError("Unknown matrix orientation: $orientation"))
    observation_axis = orientation === :rows ? 1 : 2
    feature_axis = orientation === :rows ? 2 : 1
    size(A, observation_axis) == length(y) || throw(DimensionMismatch("Observation axis and target length differ"))
    size(A, feature_axis) > 0 || throw(ArgumentError("The design matrix has no features"))
    isempty(y) && throw(ArgumentError("The dataset has no observations"))
    for field in (:E_inds, :F_inds, :W_E)
        hasproperty(info, field) || throw(ArgumentError("Observation metadata is missing $field"))
    end
    for (name, indices) in (("energy", info.E_inds), ("force", info.F_inds))
        isempty(indices) && throw(ArgumentError("The dataset has no $name observations"))
        all(i -> i isa Integer && 1 <= i <= length(y), indices) ||
            throw(ArgumentError("Invalid $name observation indices"))
        length(unique(indices)) == length(indices) ||
            throw(ArgumentError("Duplicate $name observation indices"))
    end
    isempty(intersect(info.E_inds, info.F_inds)) ||
        throw(ArgumentError("Energy and force observation indices overlap"))
    length(info.W_E) == length(info.E_inds) ||
        throw(DimensionMismatch("One energy weight is required per energy observation"))
    all(w -> isfinite(w) && w > 0, info.W_E) ||
        throw(ArgumentError("Energy weights must be finite and positive"))
    all(isfinite, y) || throw(ArgumentError("Targets contain non-finite values"))
    all(isfinite, A) || throw(ArgumentError("The design matrix contains non-finite values"))
    return nothing
end

function _objective_blocks(A, y, info)
    validate_observations(A, y, info)
    return (A_E=view(A, info.E_inds, :), y_E=view(y, info.E_inds),
            W_E=info.W_E, A_F=view(A, info.F_inds, :), y_F=view(y, info.F_inds))
end

function _component_losses(theta, blocks)
    residual_E = blocks.A_E * theta - blocks.y_E
    residual_F = blocks.A_F * theta - blocks.y_F
    return (L_E=mean(abs2, residual_E .* blocks.W_E),
            L_F=mean(abs2, residual_F))
end

"""Return `(L_E, L_F)` for an observations × features matrix.

`L_E` averages squared total-energy residuals divided by atom count. It is not
the squared energy-per-atom RMSE. `L_F` averages squared force-component errors.
"""
function component_losses(theta::AbstractVector, A::AbstractMatrix, y::AbstractVector, info)
    length(theta) == size(A, 2) || throw(DimensionMismatch("Coefficient and feature counts differ"))
    return _component_losses(theta, _objective_blocks(A, y, info))
end

function _validate_objective_weights(a, b, gamma)
    all(v -> isfinite(v) && v >= 0, (a, b, gamma)) ||
        throw(ArgumentError("Objective weights and regularization must be finite and nonnegative"))
    a + b > 0 || throw(ArgumentError("At least one data-loss weight must be positive"))
    return nothing
end

function _weighted_objective_gradient!(G, theta, blocks; a, b, gamma)
    residual_E = blocks.A_E * theta - blocks.y_E
    residual_F = blocks.A_F * theta - blocks.y_F
    loss_E = mean(abs2, residual_E .* blocks.W_E)
    loss_F = mean(abs2, residual_F)
    if G !== nothing
        grad_E = transpose(blocks.A_E) * (residual_E .* (blocks.W_E .^ 2))
        grad_F = transpose(blocks.A_F) * residual_F
        energy_scale = 2 * a / length(blocks.y_E)
        force_scale = 2 * b / length(blocks.y_F)
        @. G = energy_scale * grad_E + force_scale * grad_F + 2 * gamma * theta
    end
    return a * loss_E + b * loss_F + gamma * sum(abs2, theta)
end

"""Evaluate `a*L_E + b*L_F + gamma*sum(abs2, theta)` and write its gradient.

Pass `G=nothing` for loss only. The matrix must be observations × features.
Fixed-multiplier refinement uses `(a,b)=(1,lambda)`; static weighting uses
`(alpha,1-alpha)`. These objectives are not interchangeable at fixed `gamma`.
"""
function weighted_objective_gradient!(G, theta::AbstractVector, A::AbstractMatrix,
                                      y::AbstractVector, info; a=1.0, b=1.0, gamma=1e-8)
    _validate_objective_weights(a, b, gamma)
    length(theta) == size(A, 2) || throw(DimensionMismatch("Coefficient and feature counts differ"))
    G === nothing || length(G) == length(theta) || throw(DimensionMismatch("Gradient and coefficient lengths differ"))
    return _weighted_objective_gradient!(G, theta, _objective_blocks(A, y, info); a, b, gamma)
end
