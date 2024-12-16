module Megafauna

using Distributions
using Distances

include("DPC.jl")
include("ChangePoints.jl")
include("SegmentDistances.jl")
include("Scorer.jl")
include("MakieViz.jl")

function cluster_1d(X, Q, W, N, title="plot")
    cps = compute_change_points(X, Q, W)
    segment_distances = pairwise_segment_distances_1d(X, cps)
    labels = get_clusters(X, cps, segment_distances, N)
    point_labels = label_series(X, cps, labels)
    filtered_cps, _ = filter_change_points_from_labels(X, cps, labels)
    plot_timeseries(X, filtered_cps, point_labels, title, "$(title).pdf")
end


function entropic_cluster_1d(X, Q, W, N, ε, title="plot")
    cps = compute_change_points(X, Q, W)
    segment_distances = entropic_segment_distances(X, cps, ε)
    labels = get_clusters(X, cps, segment_distances, N)
    pt_labels = label_series(X, cps, labels)
    filtered_cps, _ = filter_change_points_from_labels(X, cps, labels)
    score = score_labeling(labels, 100)
    println("Spectral norm of transition matrix: $(score)")
    fname = title * "-$(score).pdf"
    plot_timeseries(X, filtered_cps, pt_labels, title, fname)
end

function linear_entropic_cluster_QGKJL(X, Q, W, N, ε, title="plot")
    ρ = Uniform(-2,2)
    cps = compute_change_points(X, Q, W)
    println("Found $(size(cps,1)) change points")
    segment_distances = linear_entropic_segment_distances_QGKJL(X, cps, ε, ρ)
    labels = get_clusters(X, cps, segment_distances, N)
    pt_labels = label_series(X, cps, labels)
    filtered_cps, _ = filter_change_points_from_labels(X, cps, labels)
    score = score_labeling(labels, 100)
    println("Spectral norm of transition matrix: $(score)")
    fname = "$(score).pdf"
    plot_timeseries(X, filtered_cps, pt_labels, title, fname)
end

function linear_entropic_cluster_mc_1d(X, Q, W, N, ε, title="plot")
    ρ = Uniform(-2,2)
    cps = compute_change_points(X, Q, W)
    segment_distances = linear_entropic_segment_distances_mc(X, cps, ε, ρ)
    labels = get_clusters(X, cps, segment_distances, N)
    pt_labels = label_series(X, cps, labels)
    filtered_cps, _ = filter_change_points_from_labels(X, cps, labels)
    score = score_labeling(labels, 100)
    println("Spectral norm of transition matrix: $(score)")
    fname = title*"-$(score).pdf"
    plot_timeseries(X, filtered_cps, pt_labels, title, fname)
end

function linear_entropic_cluster_presampled_mc_1d(X, Q, W, N, ε, title="plot")
    ρ = Uniform(-2,2)
    cps = compute_change_points(X, Q, W)
    segment_distances = linear_entropic_segment_distances_presampled_mc(X, cps, ε, ρ)
    labels = get_clusters(X, cps, segment_distances, N)
    pt_labels = label_series(X, cps, labels)
    filtered_cps, _ = filter_change_points_from_labels(X, cps, labels)
    score = score_labeling(labels, 100)
    fname = title*"-$(score).pdf"
    println("Spectral norm of transition matrix: $(score)")
    plot_timeseries(X, filtered_cps, pt_labels, title, fname)
end

function periodic_cluster_1d(X, Q, W, N, title="plot")
    c(x,y) = peuclidean(x,y,1.0)
    cps = compute_change_points_periodic(X, Q, W)
    segment_distances = pairwise_segment_distances_1d(X, cps, c)
    println("getting clusters")
    labels = get_clusters(X, cps, segment_distances, N)
    println("finished getting clusters")
    println("creating labels")
    pt_labels = label_series(X, cps, labels)
    println("finished labeling")
    println("filtering labeling")
    filtered_cps, _ = filter_change_points_from_labels(X, cps, labels)
    println("finished filtering labeling")
    println("scoring")
    score = score_labeling(labels, 100)
    println("finished scoring")
    fname = title*"-$(score).png"
    #println("plotting")
    #scatter_timeseries(X, filtered_cps, pt_labels, title, fname)
    return pt_labels
end

function linear_entropic_cluster_periodic_presampled_mc(X, Q, W, N, ε, title="plot")
    #ρ = fit(MvNormal, transpose(X))
    mean = [0., 0]
    σ_1 = 0.075
    Σ_1 = [σ_1^2 0.0; 0.0 σ_1^2]
    ρ = MvNormal(mean, Σ_1)
    cps = compute_change_points_periodic(X, Q, W)
    segment_distances = linear_entropic_segment_distances_periodic_presampled(X, cps, ε, ρ)
    labels = get_clusters(X, cps, segment_distances, N)
    pt_labels = label_series(X, cps, labels)
    score = score_labeling(labels, 100)
    println("Spectral norm of transition matrix: $(score)")
    fname = "$(score)-" * title * ".png"
    plot_ramachandran(X, pt_labels, title, fname)
end

function illustrate_transport_maps(X, Q, W, ε)
    cps = compute_change_points_periodic(X, Q, W)
    ρ = fit(MvNormal, transpose(X))
    maps, μ = linear_entropic_maps_presampled(X, cps, ε, ρ)
    plot_maps(X, cps, maps, μ)
end

function componentwise_periodic_cluster(X, Q, W, N)
    T, D = size(X)
    dummy_vec = [10^i for i in range(D-1, 0; step=-1)]'
    Y = zeros(T, D)
    @showprogress for d in 1:D
        title = "$(Q[d])-$(W[d])-$N([d])"
        Y[:,d] = periodic_cluster_1d(X[:,d], Q[d], W[d], N[d], title)
    end
    series_labels = [dummy_vec * Y[t,:] for t in 1:T]
    unique_labels = unique(series_labels)
    labels = [findfirst(idx -> idx == label, unique_labels) for label in series_labels]
    score = score_labeling(labels, 100)
    println("Spectral norm of transition matrix: $(score)")
    fname = "$(score)-" *"$(Q)-$(W)-$(N)"  * ".png"
    plot_ramachandran(X, series_labels, "$(Q)-$(W)-$(N)", fname)
end

export periodic_cluster_1d
export componentwise_periodic_cluster
export cluster_1d, entropic_cluster_1d
export linear_entropic_cluster_mc_1d
export linear_entropic_cluster_presampled_mc_1d
export linear_entropic_cluster_periodic_presampled_mc
export illustrate_transport_maps

end
