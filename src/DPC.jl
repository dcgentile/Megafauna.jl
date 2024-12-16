using Base.Threads
using Distributed
using SharedArrays
using Statistics
using Distances

function select_dc(dists, q=0.98)
    """
    Given a distance matrix, get the 2nd percentile of all distances

    Args:
        dists : square matrix of pairwise segment distances
    Returns:
        Float64 : distance cutoff
    """
    N = size(dists, 1)
    tt = reshape(dists, N * N)
    percent = 2.0
    position = Int(floor(N * (N - 1) * percent / 100))
    dc = sort(tt)[position + N]
    return dc
    #N = size(dists, 1)
    #arr = reshape(dists, N^2)
    #return quantile(arr, q)
end

function get_density(dists, dc; method::Union{Nothing, Symbol} = nothing)
    """
    Kernel density estimator

    Args:
        dists : square matrix of pairwise segment distances
        dc    : distance cutoff
    Returns:
        Array{Float64, 1} : array of relative densities of each trajectory segment
    """
    N = size(dists, 1)
    rho = zeros(Float64, N)
    for i in 1:N
        if method === nothing
            rho[i] = sum(dists[i, :] .< dc) - 1
        else
            rho[i] = sum(exp.(-((dists[i, :] ./ dc) .^ 2))) - 1
        end
    end
    return rho
end

function get_deltas(dists, rho)
    N = size(dists, 1)
    @assert length(rho) == N "Length of rho must match dimensions of dists"
    deltas = zeros(Float64, N)
    nn = zeros(Int, N)

    index_rho = sortperm(rho, rev=true)  # Indices sorted by descending rho

    # Loop through each point with density lower than the maximum
    for i in 2:N # Start from 2 since index 1 is the highest density
        index = index_rho[i]
        index_higher_rho = index_rho[1:i-1]  # Indices with higher density

        # Compute distances only for valid higher-density neighbors
        distances = dists[index, index_higher_rho]
        deltas[index] = minimum(distances)
        index_nn = argmin(distances)
        nn[index] = index_higher_rho[index_nn]
    end

    # Set delta for the highest density point
    deltas[index_rho[1]] = maximum(deltas)
    return deltas, nn
end


function find_centers_K(rho, deltas, K)
    """
    Get segments that represent cluster centers

    Args:
        rho    : array of relative densities of each trajectory segment
        deltas : array of distances to next closest segment of higher density
        K      : number of clusters to find
    Returns:
        Vector{Int} : an array of trajectory segments representing cluster centers
    """
    rho_delta = rho .* deltas
    centers = sortperm(rho_delta, rev=true)
    return centers[1:K]
end

function cluster_PD(rho, centers, nn)
    """
    Get cluster labels corresponding to each trajectory segment

    Args:
        rho     : array of relative densities of each trajectory segment
        centers : an array of trajectory segments representing cluster centers
        nn      : array of index of the segments corresponding to the segment of higher density
    Returns:
        Vector{Int} : array of cluster labels
    """
    K = length(centers)
    if K == 0
        println("Cannot find centers")
        return
    end
    N = length(rho)
    labs = fill(-1, N)  # Initialize labels array with -1
    for (i, center) in enumerate(centers)
        labs[center] = i - 1  # Julia uses 1-based indexing, so adjust accordingly
    end
    index_rho = sortperm(rho, rev=true)
    for index in index_rho
        if labs[index] == -1
            labs[index] = labs[nn[index]]
        end
    end
    return labs
end

function get_xx(dists, regions)
    N = size(dists, 1)
    new_dist_mtx = Vector{Vector{Float64}}()

    # Iterate over the upper triangular part of the matrix (i < j)
    for i in 1:N-1
        for j in i+1:N
            push!(new_dist_mtx, [i, j, dists[i, j], length(regions[i]), length(regions[j])])
        end
    end

    # Convert the list of rows to a matrix
    xx = hcat(new_dist_mtx...)'
    return xx
end

function updated_density_estimate(dists, regions, dc)
    xx = get_xx(dists, regions)
    #println("xx shape: $(size(xx))")

    N = size(dists, 1)  # Number of data points
    dist = zeros(Float64, N, N)
    density = zeros(Float64, N)

    # Fill in arrays based on values given in `xx`
    for i in 1:size(xx, 1)
        ii = Int(xx[i, 1])
        jj = Int(xx[i, 2])
        dist[ii, jj] = xx[i, 3]
        dist[jj, ii] = xx[i, 3]
        density[ii] = xx[i, 4]
        density[jj] = xx[i, 5]
    end

    # Initialize rho with a copy of density
    rho = copy(density)

    # Update rho values using the Gaussian kernel
    for i in 1:N-1
        for j in i+1:N
            gaussian_kernel = exp(-((dist[i, j] / dc) ^ 2))
            rho[i] += density[j] * gaussian_kernel
            rho[j] += density[i] * gaussian_kernel
        end
    end

    #println("rho size: ", size(rho))
    return rho
end

function split_data_by_change_points(data, cps)
    states = [data[cps[i]:cps[i+1]-1, :] for i in 1:length(cps)-1]
    return states
end

function get_clusters(data, cps, distances, n_clusters)
    states = split_data_by_change_points(data, cps)
    alternate_dc = select_dc(distances)
    println(alternate_dc)
    rho = updated_density_estimate(distances, states, alternate_dc)
    deltas, nearest_neighbor = get_deltas(distances,rho)
    centers=find_centers_K(rho, deltas, n_clusters)
    labels = cluster_PD(rho, centers, nearest_neighbor)
    return labels
end

export get_clusters
