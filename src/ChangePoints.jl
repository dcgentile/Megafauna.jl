using Distributed
using ProgressMeter
using OptimalTransport
using Distributions
using Distances
include("./DPC.jl")

function compute_metric_wasserstein_derivative(x, W, c)
    N = size(x,1)
    D = size(x,2)
    distances = zeros(N,D)
    for d in 1:D
        w = W[d]
        @showprogress @distributed for t in w+1:N-w
            slice_left = unique(x[t-w:t-1,:])
            slice_right = unique(x[t:t+w-1,:])
            len_left = length(slice_left)
            len_right = length(slice_right)
            p = fill(1/len_left,len_left)
            q = fill(1/len_right, len_right)
            μ = DiscreteNonParametric(slice_left, p)
            ν = DiscreteNonParametric(slice_right, q)
            distances[t, d] = ot_cost(c, μ, ν)^2
        end
    end
    return distances
end

function enumerate_change_points(distances, q)
    change_points = Vector{Int64}()
    D = size(distances, 2)
    for d in 1:D
        inflections = find_inflections(distances[:,d], q[d])
        append!(change_points, inflections)
    end
    return sort!(unique!(change_points))
end

function find_inflections(distances, q)
    T = size(distances,1)
    grad = lazy_gradient(distances)
    cutoff = quantile(distances,q)
    candidates = Vector{Int64}()
    for t in 1:T
        if distances[t] > cutoff
            append!(candidates, t)
        end
    end
    sort!(candidates)

    in_sequence = false
    seq_start = 0
    cps = [1, T]
    N = length(candidates)

    for n in 1:N-1
        nx = candidates[n+1]
        curr = candidates[n]
        if !in_sequence && nx - curr == 1
            in_sequence = true
            seq_start = curr
        end

        if in_sequence && nx - curr > 1
            in_sequence = false
            seq_end = curr
            sequence = grad[seq_start:seq_end]
            push!(cps, argmin(sequence) + seq_start)
            push!(cps, argmax(sequence) + seq_start)
        end
    end
    return sort!(cps)
end

function lazy_gradient(x)
    N = length(x)
    grad = zeros(N)
    for t in 2:N-1
        grad[t] = 0.5 * (x[t+1] - x[t - 1])
    end

    grad[1] = x[2] - x[1]
    grad[end] = x[end] - x[end-1]
    return grad
end


function compute_change_points(x, q, W, c=sqeuclidean)
    distances = compute_metric_wasserstein_derivative(x, W, c)
    cps = enumerate_change_points(distances, q)
    return cps
end

function label_series(x, cps, labels)
    T = length(cps)
    point_labels = zeros(size(x,1))
    for i in 1:T-1
        t0 = cps[i]
        t1 = cps[i + 1]
        label = labels[i]
        point_labels[t0:t1-1] .= label
    end
    return point_labels
end


export compute_change_points, label_series
