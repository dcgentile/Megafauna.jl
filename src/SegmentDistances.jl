using Distributions
using ProgressMeter
using OptimalTransport
using Distances
using Distributed
using Integrals
using LinearAlgebra

#####################################################################################################
# Standard Wasserstein Segment Distances
#####################################################################################################

function pairwise_segment_distances_1d(x, cps, c=sqeuclidean)
    N = size(cps, 1) - 1
    A = zeros(N,N)
    prog = Progress((N*(N - 1)) ÷ 2)
    println("Number of segments = $(N)")
    println("Computing $((N * (N - 1)) ÷ 2) segment distances")
    @sync @distributed for i in 1:N
        t00, t01 = cps[i], cps[i+1]
        Si = unique(x[t00:t01,:])
        len_i = length(Si)
        p = fill(1/len_i, len_i)
        μ = DiscreteNonParametric(Si, p)
        for j in i+1:N
            t10, t11 = cps[j], cps[j+1]
            Sj = unique(x[t10:t11,:])
            len_j = length(Sj)
            q = fill(1/len_j, len_j)
            ν = DiscreteNonParametric(Sj, q)
            d = ot_cost(c, μ, ν)^2
            A[i,j] = d
            A[j,i] = d
            next!(prog)
        end
    end
    println("finished computing distances")
    return A
end

#####################################################################################################
# Entropic Wasserstein Segment Distances
#####################################################################################################

function entropic_segment_distances(x, cps, ε, c=SqEuclidean())
    N = size(cps, 1) - 1
    A = zeros(N,N)
    p = Progress(N*(N - 1) ÷ 2; dt=1.0)
    cost(X, Y) = pairwise(c, X, Y).^2
    @sync @distributed for i in 1:N
        t00, t01 = cps[i], cps[i+1]
        Si = x[t00:t01,:]
        μ = fill(1/(t01 - t00 + 1), t01 - t00 + 1)
        for j in i+1:N
            t10, t11 = cps[j], cps[j+1]
            Sj = x[t10:t11,:]
            ν = fill(1/(t11 - t10 + 1), t11 - t10 +1)
            C = cost(Si', Sj')
            d = sinkhorn2(μ, ν, C, ε)
            A[i,j] = d
            A[j,i] = d
            next!(p)
        end
    end
    return A
end

#####################################################################################################
# Linearized Entropic Wasserstein Segment Distances
#####################################################################################################
function gather_entropic_maps_mc(x, cps, ε, ρ::Distributions.Distribution, c)
    N = length(cps)
    D = size(x,2)
    entropic_maps = Array{Function}(undef, N-1)

    @sync @distributed for i in 1:N-1
        t0, t1 = cps[i], cps[i+1]
        n = t1 - t0 + 1
        spt_ν = x[t0:t1,:]
        spt_μ = rand(ρ, (n,D))
        C = c(spt_μ', spt_ν')
        μ = fill(1/n,n)
        T = entropic_transport_map(μ, μ, spt_ν, C, ε, SinkhornGibbs())
        entropic_maps[i] = T
    end

    return entropic_maps

end

function linear_entropic_transport_mc(Ti, Tj, μ)
    N = size(μ, 1)
    evaluations = zeros(N)
    @sync @distributed for i=1:N
	    evaluations[i] = (Ti([μ[i]]) - Tj([μ[i]]))^2
    end
    return (sum(evaluations) / N)^0.5
end

function linear_entropic_transport_QGKJL(Ti, Tj, dom=(-2,2))
    f(x, p) = (Ti([x]) - Tj([x]))^2
    i = solve(IntegralProblem(f, dom), QuadGKJL()).u
    return i^0.5
end

function pairwise_linear_entropic_dists_mc(entropic_maps, ρ::Distributions.Distribution, nsamples=1000)
    N = size(entropic_maps, 1)
    A = zeros(N,N)
    p = Progress((N*(N - 1)) ÷ 2)
    @sync @distributed for i in 1:N
        Ti = entropic_maps[i]
        for j in i:N
            μ = rand(ρ, nsamples)
            Tj = entropic_maps[j]
            d = linear_entropic_transport_mc(Ti, Tj, μ)
            A[i,j] = d
            A[j,i] = d
            next!(p)
        end
    end
    return A
end


function pairwise_linear_entropic_dists_QGKJL(entropic_maps)
    N = size(entropic_maps, 1)
    A = zeros(N,N)
    p = Progress((N*(N - 1)) ÷ 2)
    @sync @distributed for i in 1:N
        Ti = entropic_maps[i]
        for j in i:N
            Tj = entropic_maps[j]
            d = linear_entropic_transport_QGKJL(Ti, Tj)
            A[i,j] = d
            A[j,i] = d
            next!(p)
        end
    end
    return A
end

function linear_entropic_segment_distances_mc(X, cps, ε, ρ::Distributions.Distribution, N=1000)
    c(X, Y) = pairwise(SqEuclidean(), X, Y).^2
    entropic_maps = gather_entropic_maps_mc(X, cps, ε, ρ, c)
    return pairwise_linear_entropic_dists_mc(entropic_maps, ρ)
end

function linear_entropic_segment_distances_QGKJL(X, cps, ε, ρ::Distributions.Distribution)
    c(X,Y) = pairwise(SqEuclidean(), X,Y)
    entropic_maps =gather_entropic_maps_mc(X, cps, ε, ρ, c)
    return pairwise_linear_entropic_dists_QGKJL(entropic_maps)
end
#####################################################################################################
# Presampled Linearized Entropic Wasserstein Segment Distances
#####################################################################################################

function gather_entropic_maps_presampled_mc(X, cps, ε, spt_μ, c)
    N = length(cps)
    entropic_maps = Array{Function}(undef, N-1)
    μ = fill(1/size(spt_μ,1),size(spt_μ, 1))
    @sync @distributed for i in 1:N-1
        t0, t1 = cps[i], cps[i+1]
        n = t1 - t0 + 1
        spt_ν = X[t0:t1,:]
        C = c(spt_μ', spt_ν')
        ν = fill(1/n,n)
        T = entropic_transport_map(μ, ν, spt_ν, C, ε, SinkhornGibbs())
        entropic_maps[i] = T
    end
    return entropic_maps
end


function pairwise_linear_entropic_dists_presampled_mc(entropic_maps, μ)
    N = size(entropic_maps, 1)
    A = zeros(N,N)
    println("Computing segment distances, num_segments = $(N)")
    p = Progress((N*(N - 1)) ÷ 2)
    @sync @distributed for i in 1:N
        Ti = entropic_maps[i]
        for j in i:N
            Tj = entropic_maps[j]
            d = linear_entropic_transport_mc(Ti, Tj, μ)
            A[i,j] = d
            A[j,i] = d
            next!(p)
        end
    end
    return A
end

function linear_entropic_segment_distances_presampled_mc(X, cps, ε, ρ::Distributions.Distribution, N=1000)
    c(X, Y) = pairwise(SqEuclidean(), X, Y).^2
    spt_μ = rand(ρ, N)
    entropic_maps = gather_entropic_maps_presampled_mc(X, cps, ε, spt_μ, c)
    return pairwise_linear_entropic_dists_presampled_mc(entropic_maps, spt_μ')
end

function linear_entropic_maps_presampled(X, cps, ε, ρ::Distributions.Distribution, N=1000)
    c(X, Y) = pairwise(SqEuclidean(), X, Y).^2
    spt_μ = rand(ρ, N)' .% 1
    entropic_maps = gather_entropic_maps_presampled_mc(X, cps, ε, spt_μ, c)
    return (entropic_maps, spt_μ)
end

#####################################################################################################
# Presampled Linearized Entropic Wasserstein Segment Distances
#####################################################################################################
function linear_entropic_segment_distances_periodic_presampled(X, cps, ε, ρ::Distributions.Distribution, N=1000)
    c(X, Y) = pairwise(PeriodicEuclidean([1,1]), X, Y).^2
    spt_μ = rand(ρ, N)
    entropic_maps = gather_entropic_maps_presampled_mc(X, cps, ε, spt_μ', c)
    return fast_linear_entropic_segment_distances(entropic_maps, ρ)
    #return pairwise_linear_entropic_dists_presampled_mc(entropic_maps, spt_μ')
end


# Updated LEOT segment distance method by pre-sampling
function fast_linear_entropic_segment_distances(entropic_maps, ρ::Distributions.Distribution, num_samples=1000)
    # initialize distance matrix
    N = size(entropic_maps, 1)
    print(N)
    A = zeros(N,N)
    p = Progress((N*(N - 1)) ÷ 2)
    samples = rand(ρ, num_samples)

    T_eval = Vector{Vector{Matrix{Float64}}}(undef, N)
    #T_eval = zeros(N, 2, num_samples)

    @sync @distributed for i in 1:N
        Ti = entropic_maps[i]
        # Calculate the result for each sample and assign to T_eval[i]
        local_result = [Ti(samples[:, k]) for k in 1:num_samples]
        T_eval[i] = local_result
    end

    @sync @distributed  for i in 1:N-1
        for j in i+1:N
            diffs = T_eval[i] .- T_eval[j]
            norms = [(diffs[k] * diffs[k]')[1] for k in 1:num_samples]
            d = (sum(norms)/num_samples)^0.5
            A[i,j] = d
            A[j,i] = d
            next!(p)
        end
    end
    return A
end
