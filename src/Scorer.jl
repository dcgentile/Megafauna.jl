"""
Process summary:
1: import data
2: import change points
3: import cluster labels
4: divide data according to change points and label points in each segments with cluster labels
5: construct counts matrix
6: fixed point method to find transition matrix
7: compute singular values
8: VAMP2 score on singular values
"""

using LinearAlgebra

function construct_counts_matrix(labels, lag_time)
    unique_labels = unique(labels)
    n = length(unique_labels)
    state_to_index = Dict(l => i for (i, l) in enumerate(unique_labels))
    C = zeros(Int, n, n)

    for i in 1:(length(labels) - lag_time)
        current_state = labels[i]
        next_state = labels[i + lag_time]
        current_index = state_to_index[current_state]
        next_index = state_to_index[next_state]

        C[current_index, next_index] += 1
    end

    return C
end


function fixed_point_iteration(C, tol=1e-5, max_iter=10000, verbose=false)
    # C is the matrix of counts (c_ij)
    n = size(C, 1)
    X = ones(n, n)
    X_new = copy(X)

    for iter in 1:max_iter
        c = sum(C, dims=2)
        x = sum(X, dims=2)

        for i in 1:n, j in 1:n
            if i != j
                X_new[i, j] = (C[i, j] + C[j, i]) / (c[i] / x[i] + c[j] / x[j])
            end
        end

        if norm(X_new - X) < tol
            if verbose println("Converged after $iter iterations.") end
            break
        end

        X .= X_new
    end

    if verbose println("Did not converge after $max_iter iterations.") end
    row_sums = sum(X, dims=2)
    normalized_matrix_X = X ./ row_sums
    return normalized_matrix_X
end

function get_singular_values(matrix)
    S = svd(matrix).S
    return S
end

function score_labeling(labels, lag_time)
    C = construct_counts_matrix(labels, lag_time)
    X = fixed_point_iteration(C)
    #score = sum(get_singular_values(X) .^2)
    score = norm(X)^2
    return score
end

export score_labeling
