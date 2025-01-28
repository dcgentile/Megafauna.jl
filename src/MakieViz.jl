using Dates
using CairoMakie
using KernelDensity

function example()
	n = 1000;
    x = 1:n;
    y = sin.(collect(x) .* 0.05);
    z = vcat(1:n/2, n/2:-1:1) .- n/4

    colorrange = [-n/4, n/4];

    cmap = :balance

    fig = Figure()

    scatter(fig[1,1], x, y; color=z, colormap = cmap, markersize=10, strokewidth=0, colorrange=colorrange)
    Colorbar(fig[1,2], colorrange=colorrange, colormap=cmap, label="Colored Data")
    fig
    save("fig.pdf", fig)
end

function plot_sing_backbone(X, point_labels, cmap=:darktest)
    fig = Figure(size = (1000, 7000))
    x = [cos(π*t) for t in X]
    y = [sin(π*t) for t in X]
    scatter(fig[1,1], x, y; color=point_labels, colormap=Makie.Categorical(cmap), markersize=4, alpha=0.4)
    save("fig.pdf", fig)
end

function plot_timeseries(X, cps, point_labels, title, fname, cmap=:darktest)
    fig = Figure(size=(3000,500))
    ax = Axis(fig[1,1], title=title, xlabel="t", ylabel="x(t)")
    lines!(ax, 1:size(X,1), X[:,1], color=point_labels, colormap=Makie.Categorical(cmap))
    vlines!(cps, linestyle=:dash)
    save(fname, fig)
end

function scatter_timeseries(X, cps, point_labels, title, fname, cmap=:darktest)
    fig = Figure(size=(3000,500))
    ax = Axis(fig[1,1], title=title, xlabel="t", ylabel="x(t)")
    scatter!(ax, 1:size(X,1), X[:,1], color=point_labels, colormap=Makie.Categorical(cmap))
    vlines!(cps, linestyle=:dash)
    save(fname, fig)
end

function plot_changes(X, cps, title, fname, cmap=:darktest)
    fig = Figure(size=(3000,500))
    ax = Axis(fig[1,1], title=title, xlabel="t", ylabel="x(t)")
    scatter!(ax, 1:size(X,1), X[:,1])
    vlines!(cps, linestyle=:dash, color=:red)
    save(fname, fig)
end

function plot_maps(X, cps, maps, μ)
    N = size(cps,1)
    @showprogress for i in 1:N-1
        fig = Figure(size=(1500, 500))
        T = maps[i]
	    t0, t1 = cps[i], cps[i+1]
        Si = X[t0:t1,:]
        directions = reduce(vcat, [T(x) for x in eachrow(μ)]) .- μ
        ax1 = Axis(fig[1,1], title="Presampled Reference", xlabel="φ", ylabel="ψ", limits=(0,1,0,1))
        ax2 = Axis(fig[1,2], title="Segment $(i); Length of Segment: $(size(Si, 1))", xlabel="φ", ylabel="ψ", limits=(0,1,0,1))
        ax3 = Axis(fig[1,3], title="Illustrated Transport Map", xlabel="φ", ylabel="ψ", limits=(0,1,0,1))
        scatter!(ax1, μ; color=:red,  alpha=0.5)
        scatter!(ax2, Si; color=:blue, alpha=0.5)
        scatter!(ax3, Si; color=:blue, alpha=0.5)
        scatter!(ax3, μ; color=:red,  alpha=0.5)
        arrows!(ax3, μ[:,1], μ[:,2], directions[:,1], directions[:,2]; alpha=0.66)
        save("segment_$(i).png", fig)
    end
end

function plot_ramachandran(X, point_labels, title, fname, height=1000, density=true, cmap=:darktest)
    fig = Figure(size = (1000, 7000))
    labels = unique(point_labels)
    markersize=5
    alpha=0.95
    for (idx, label) in enumerate(labels)
        indices = findall(x->x==label, point_labels)
        cluster = X[indices,:]
        K = size(cluster,1)
        sample_size = size(cluster, 1)
        proportion = sample_size / size(X, 1)
        ax = Axis(fig[idx+1,1], xlabel="ψ", ylabel="φ", limits=(0,1,0,1), title="Cluster $(idx), Samples: $(sample_size), Proportion of Timeseries: $(proportion)")
        if density
            kernel = kde(cluster)
            colors = [pdf(kernel, cluster[k,1], cluster[k,2]) for k in 1:K]
            scatter!(ax, cluster; color=colors, colormap=cmap, markersize=markersize)
            current_figure()
        else
            scatter!(ax, cluster; color=point_labels, colormap=Makie.Categorical(cmap), markersize=markersize, alpha=alpha)
        end
    end
    ax = Axis(fig[1,1], title=title, xlabel="ψ", ylabel="φ", limits=(0,1,0,1))
    scatter!(ax, X; color=point_labels, colormap=Makie.Categorical(cmap), markersize=markersize, alpha=alpha)
    save(fname, fig)
end

function ramachandran(X, point_labels, title, fname; width=1000, density=true, cmap=:darktest, ext="png", step=5)
    #fig = Figure(size = (2*width, 4*width))
    labels = unique(point_labels)
    markersize=5
    alpha=0.95
    time = Dates.format(now(),"yy-mm-dd-HH-MM-SS")
    img_folder_path = "img-$(time)"
    mkdir(img_folder_path)
    for (idx, label) in enumerate(labels)
        fig = Figure(size=(width, width))
        println("plotting cluster $(idx)")
        f = fig[1,1]
        fcb = fig[1, 2]
        indices = findall(x->x==label, point_labels)
        cluster = X[indices,:]

        K = size(cluster,1)
        sample_size = size(cluster, 1)
        proportion = sample_size / size(X, 1)
        ax = Axis(f, xlabel="ψ", ylabel="φ", limits=(0,1,0,1), title="Cluster $(idx)\nSamples: $(sample_size)\nProportion of Timeseries: $(proportion)")
        if density
            kernel = kde(cluster)
            colors = [pdf(kernel, cluster[k,1], cluster[k,2]) for k in 1:K]
            Colorbar(fcb, limits = (minimum(colors), maximum(colors)), colormap = cmap ,flipaxis = false)
            scatter!(ax, cluster; color=colors, colormap=cmap, markersize=markersize)
            #if K > 10000
                #scatter!(ax, cluster[begin:5:end,:]; color=colors[begin:5:end], colormap=cmap, markersize=markersize)
            #else
                #scatter!(ax, cluster; color=colors, colormap=cmap, markersize=markersize)
            #end
        else
            scatter!(ax, cluster; color=point_labels, colormap=Makie.Categorical(cmap), markersize=markersize, alpha=alpha)
        end
        save("$(img_folder_path)/$(fname)-cluster-$(idx).$(ext)", fig)
    end
    fig = Figure(size=(width, width))
    ax = Axis(fig[1,1], title=title, xlabel="ψ", ylabel="φ", limits=(0,1,0,1))
    scatter!(ax, X[begin:end,:]; color=point_labels[begin:end], colormap=Makie.Categorical(cmap), markersize=markersize, alpha=alpha)
    save("$(img_folder_path)/$(fname)-ensemble.$(ext)", fig)


end

export plot_sing_backbone, plot_timeseries, plot_ramachandran

export ramachandran
