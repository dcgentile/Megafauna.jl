using Plots
using DataFrames

function plot_ramachandran(x, cps, labels, title, fname, height=1000)
    n_clusters = length(unique(labels))
    p = plot(size=(1500, 1500), layout=(4,3))
    N = length(cps)
    colors = ["red", "blue", "green", "black", "purple", "orange", "magenta", "cyan", "yellow", "brown"]
    for i in 1:N-1
        t0 = cps[i]
        t1 = cps[i + 1]
        label = labels[i]
        scatter!(p,
                 x[t0:t1,1],
                 x[t0:t1,2],
                 color="blue",
                 markeralpha=0.1,
                 markersize=0.02,
                 subplot=label+1,
                 title="Cluster $(label)"
                 )
        scatter!(p,
                 x[t0:t1,1],
                 x[t0:t1,2],
                 color=colors[label + 1],
                 markeralpha=0.1,
                 markersize=0.02,
                 subplot=n_clusters+1,
                 title="All Data"
                 )

    end
    p[:plot_title] = title
    plot!(legend=false,)
    savefig(p, "$(fname)")
end


function plot_timeseries(x, cps, labels, title, fname, width=3000, height=500)
	p = plot()
    N = length(cps)
    colors = ["red", "blue", "green", "purple", "black", "magenta", "cyan", "yellow"]
    for i in 1:N-1
        t0 = cps[i]
        t1 = cps[i + 1]
        label = labels[i]
        plot!(p, t0:t1, x[t0:t1], size=(width, height), color=colors[label + 1])
        if i < N - 1 && label != labels[i + 1]
            vline!([t1], line=:dash, color=:red)
        end
    end
    plot!(legend=false, title=title)
    savefig(p, "$(fname)")
end

function plot_backbone_angle(df, title, fname="plot.pdf")
    # assumes the data is 1D (only one backbone angle) and normalized to lie in (-1,1)
    gr()
    p = plot()
    df[!, "x"] = [cos(π * t) for t in df[!,"position"]]
    df[!, "y"] = [sin(π * t) for t in df[!,"position"]]
    scatter!(p, df.x, df.y, color=ifelse(df.label. "blue", "red"), markersize=0.5)
    plot!(legend=false, title=title)
    savefig(p, "$fname")

end
