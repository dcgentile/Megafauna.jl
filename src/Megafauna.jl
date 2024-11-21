module Megafauna

using Distributions
using Distances

include("DPC.jl")
include("ChangePoints.jl")
include("SegmentDistances.jl")
include("Scorer.jl")
include("MakieViz.jl")

function cluster(X, Q, W, N, title="plot")
    cps = compute_change_points(X, Q, W)
    segment_distances = pairwise_transport_1d(X, cps)
    labels = get_clusters(X, cps, segment_distances, N)
    point_labels = label_series(X, cps, labels)
    filtered_cps, filterd_labels = filter_change_points_from_labels(X, cps, labels)
    plot_timeseries(X, filtered_cps, point_labels, title, "$(title).pdf")
end

function periodic_cluster(X, Q, W, N, title="plot")
    c(x,y) = peuclidean(x,y,1.0)
    cps = compute_change_points_periodic(X, Q, W)
    segment_distances = pairwise_transport_1d(X, cps)
    labels = get_clusters(X, cps, segment_distances, N)
    point_labels = label_series(X, cps, labels)
    plot_sing_backbone(X, point_labels)
end

function entropic_cluster(X, Q, W, N, ε, title="plot")
    ρ = Uniform(-2,2)
    cps = compute_change_points_periodic(X, Q, W)
    segment_distances = linear_entropic_segment_distances_1d(X, cps, ε, ρ)
    labels = get_clusters(X, cps, segment_distances, N)
    pt_labels = label_series(X, cps, labels)
    filtered_cps, filterd_labels = filter_change_points_from_labels(X, cps, labels)
    score = score_labeling(labels, 100)
    println("Spectral norm of transition matrix: $(score)")
    fname = "$(score).pdf"
    plot_timeseries(X, filtered_cps, pt_labels, title, fname)
end

function entropic_cluster_presampled(X, Q, W, N, ε, title="plot")

    #ρ = fit(Normal, X)
    ρ = Uniform(-2,2)
    cps = compute_change_points_periodic(X, Q, W)
    segment_distances = linear_entropic_segment_distances_presampled(X, cps, ε, ρ)
    labels = get_clusters(X, cps, segment_distances, N)
    pt_labels = label_series(X, cps, labels)
    score = score_labeling(labels, 100)
    println("Spectral norm of transition matrix: $(score)")
    plot_timeseries(X, cps, pt_labels, title, "$(title).pdf")
end

function entropic_cluster_periodic_presampled(X, Q, W, N, ε, title="plot")

    ρ = fit(MvNormal, transpose(X))
    cps = compute_change_points_periodic(X, Q, W)
    segment_distances = linear_entropic_segment_distances_periodic_presampled(X, cps, ε, ρ)
    labels = get_clusters(X, cps, segment_distances, N)
    pt_labels = label_series(X, cps, labels)
    score = score_labeling(labels, 100)
    println("Spectral norm of transition matrix: $(score)")
    plot_ramachandran(X, labels, title, "~/Documents/$(title).pdf")
end

export cluster, entropic_cluster, periodic_cluster, entropic_cluster_periodic_presampled, entropic_cluster_presampled

end
