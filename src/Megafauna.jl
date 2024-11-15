module Megafauna

include("./ChangePoints.jl")
include("./SegmentDistances.jl")
include("./Visualizations.jl")

function cluster(X, Q, W, N, title="plot")
    cps = compute_change_points(X, Q, W)
    segment_distances = pairwise_transport_1d(X, cps)
    labels = get_clusters(X, cps, segment_distances, N)
    labeled_series = label_series(X, cps, labels)
    plot_timeseries(X, cps, labels, title, "$(title).png")
end

export cluster

end
