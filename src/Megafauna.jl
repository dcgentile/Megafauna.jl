module Megafauna

using Distributions

include("DPC.jl")
include("./ChangePoints.jl")
include("./SegmentDistances.jl")
include("./Visualizations.jl")
include("./Scorer.jl")

function cluster(X, Q, W, N, title="plot")
    cps = compute_change_points(X, Q, W)
    segment_distances = pairwise_transport_1d(X, cps)
    labels = get_clusters(X, cps, segment_distances, N)
    plot_timeseries(X, cps, labels, title, "~/Documents/$(title).png")
end

function periodic_cluster(X, Q, W, N, title="plot")
    println("Not implemented yet!")
    #cps = compute_change_points_periodic(X, Q, W)
    #segment_distances = pairwise_transport_1d(X, cps)
    #labels = get_clusters(X, cps, segment_distances, N)
    #plot_timeseries(X, cps, labels, title, "~/Documents/$(title).png")
end

@time function entropic_cluster(X, Q, W, N, ε, title="plot")
    ρ = Uniform(-2,2)
    cps = compute_change_points(X, Q, W)
    segment_distances = linear_entropic_segment_distances(X, cps, ε, ρ)
    labels = get_clusters(X, cps, segment_distances, N)
    pt_labels = label_series(X, cps, labels)
    score = score_labeling(labels, 100)
    println("Spectral norm of transition matrix: $(score)")
    plot_timeseries(X, cps, labels, title, "~/Documents/$(title).pdf")
end

export cluster, entropic_cluster

end
