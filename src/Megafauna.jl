module Megafauna

using DelimitedFiles
using Distributions
using Distances

include("DPC.jl")
include("ChangePoints.jl")
include("SegmentDistances.jl")
include("Scorer.jl")
include("MakieViz.jl")


# W2 FUNCTIONALITY

function cluster_1d(X, Q, W, N, τ=100; cps=nothing)
    cps = isnothing(cps) ? compute_change_points(X, Q, W) : cps
    segment_distances = pairwise_segment_distances_1d(X, cps)
    labels = get_clusters(X, cps, segment_distances, N)
    pt_labels = label_series(X, cps, labels)
    filtered_cps, _ = filter_change_points_from_labels(X, cps, labels)
    score = score_labeling(pt_labels, τ)
    return (filtered_cps, pt_labels, score)
end

function periodic_cluster_1d(X, Q, W, N; cps=nothing)
    c(x,y) = peuclidean(x,y,1.0).^2
    cps = isnothing(cps) ? compute_change_points_periodic(X, Q, W) : cps
    segment_distances = pairwise_segment_distances_1d(X, cps, c)
    labels = get_clusters(X, cps, segment_distances, N)
    pt_labels = label_series(X, cps, labels)
    #filtered_cps, _ = filter_change_points_from_labels(X, cps, labels)
    score = score_labeling(pt_labels, 100)
    return (cps, pt_labels, score)
end

function componentwise_periodic_cluster(X, Q, W, N; cps=nothing)
    T, D = size(X)
    dummy_vec = [10^i for i in range(D-1, 0; step=-1)]'
    Y = zeros(T, D)
    if isnothing(cps)
        change_pts = []
        for d in 1:D
            changes, Y[:,d], _ = periodic_cluster_1d(X[:,d], Q[d], W[d], N[d])
            cps= vcat(cps, changes)
        end
    end
    sort!(change_pts)
    series_labels = [dummy_vec * Y[t,:] for t in 1:T]
    unique_labels = unique(series_labels)
    labels = [findfirst(idx -> idx == label, unique_labels) for label in series_labels]
    score = score_labeling(labels, 100)
    println(unique(change_pts))
    return (unique(change_pts), series_labels, score)
end

function componentwise_euclidean_cluster(X, Q, W, N; cps=nothing)
    T, D = size(X)
    dummy_vec = [10^i for i in range(D-1, 0; step=-1)]'
    Y = zeros(T, D)
    if isnothing(cps)
        change_pts = []
        for d in 1:D
            changes, Y[:,d], _ = cluster_1d(X[:,d], Q[d], W[d], N[d])
            cps = vcat(cps, changes)
        end
    end
    sort!(cps)
    series_labels = [dummy_vec * Y[t,:] for t in 1:T]
    unique_labels = unique(series_labels)
    labels = [findfirst(idx -> idx == label, unique_labels) for label in series_labels]
    score = score_labeling(labels, 100)
    return (unique(change_pts), series_labels, score)
end

# ENTROPIC FUNCTIONALITY

function entropic_cluster_1d(X, Q, W, N, ε, τ=100)
    cps = compute_change_points(X, Q, W)
    segment_distances = entropic_segment_distances(X, cps, ε)
    labels = get_clusters(X, cps, segment_distances, N)
    pt_labels = label_series(X, cps, labels)
    filtered_cps, _ = filter_change_points_from_labels(X, cps, labels)
    score = score_labeling(pt_labels, τ)
    return (filtered_cps, pt_labels, score)
end

# LINEAR ENTROPIC FUNCTIONALITY

function linear_entropic_cluster_QGKJL(X, Q, W, N, ε, τ=100)
    ρ = Uniform(-2,2)
    cps = compute_change_points(X, Q, W)
    segment_distances = linear_entropic_segment_distances_QGKJL(X, cps, ε, ρ)
    labels = get_clusters(X, cps, segment_distances, N)
    pt_labels = label_series(X, cps, labels)
    filtered_cps, _ = filter_change_points_from_labels(X, cps, labels)
    score = score_labeling(pt_labels, τ)
    return (filtered_cps, pt_labels, score)
end

function linear_entropic_cluster_mc_1d(X, Q, W, N, ε, τ=100)
    ρ = Uniform(-2,2)
    cps = compute_change_points(X, Q, W)
    segment_distances = linear_entropic_segment_distances_mc(X, cps, ε, ρ)
    labels = get_clusters(X, cps, segment_distances, N)
    pt_labels = label_series(X, cps, labels)
    filtered_cps, _ = filter_change_points_from_labels(X, cps, labels)
    score = score_labeling(pt_labels, τ)
    return (filtered_cps, pt_labels, score)
end

function linear_entropic_cluster_presampled_mc_1d(X, Q, W, N, ε, τ=100)
    m, M = minimum(X), maximum(X)
    ρ = Uniform(m, M)
    cps = compute_change_points(X, Q, W)
    segment_distances = linear_entropic_segment_distances_presampled_mc(X, cps, ε, ρ)
    labels = get_clusters(X, cps, segment_distances, N)
    pt_labels = label_series(X, cps, labels)
    filtered_cps, _ = filter_change_points_from_labels(X, cps, labels)
    score = score_labeling(pt_labels, τ)
    return (filtered_cps, pt_labels, score)
end

function linear_entropic_cluster_periodic_presampled_mc(X, Q, W, N, ε)
    ρ = fit(MvNormal, transpose(X))
    cps = compute_change_points_periodic(X, Q, W)
    segment_distances = linear_entropic_segment_distances_periodic_presampled(X, cps, ε, ρ)
    labels = get_clusters(X, cps, segment_distances, N)
    pt_labels = label_series(X, cps, labels)
    score = score_labeling(pt_labels, 100)
    return (cps, pt_labels, score)
end


function identify_change_points(X, Q, W, domain="periodic", filename="change_points.txt")
    if domain == "periodic"
	    cps = compute_change_points(X, Q, W)
        open(filename, "w") do io
            writedlm(io, cps)
        end
    elseif domain == "euclidean"
	    cps = compute_change_points_periodic(X, Q, W)
        open(filename, "w") do io
            writedlm(io, cps)
        end
    else
        return error("Specified domain is not valid, please use either periodic or euclidean")
    end
end


function CATBOSS(X, Q, W, N; domain="periodic", cps=nothing)
    if domain == "periodic"
        cps, labels, score = componentwise_periodic_cluster(X, Q, W, N, cps=cps)
    elseif domain == "euclidean"
        cps, labels, score = componentwise_euclidean_cluster(X, Q, W, N, cps=cps)
    else
        return error("Specified domain is not valid, please use either periodic or euclidean")
    end


	
end

export periodic_cluster_1d, componentwise_periodic_cluster, ramachandran, scatter_timeseries

end
