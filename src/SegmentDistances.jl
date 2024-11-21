using Distributions
using ProgressMeter
using OptimalTransport
using Distances
using Distributions
using Distributed
using Integrals

function gather_entropic_maps_presampled(X, cps, ε, spt_μ, c)
    entropic_maps = Function[]
    N = length(cps)
    μ = fill(1/size(spt_μ,1),size(spt_μ, 1))
    for i in 1:N-1
        t0, t1 = cps[i], cps[i+1]
        n = t1 - t0 + 1
        spt_ν = X[t0:t1,:]
        C = c(spt_μ', spt_ν')
        ν = fill(1/n,n)
        T = entropic_transport_map(μ, ν, spt_ν, C, ε, SinkhornGibbs())
        push!(entropic_maps, T)
    end
    return entropic_maps
end


function linear_entropic_transport_presampled(Ti, Tj, μ)
    N = size(μ, 1)
	f(x) = sqeuclidean(Ti(x), Tj(x))
    return (sum(mapslices(f, μ; dims=2)) / N)^0.5
end

function pairwise_linear_entropic_dists_presampled(entropic_maps, μ)
    N = size(entropic_maps, 1)
    A = zeros(N,N)
    p = Progress((N*(N - 1)) ÷ 2)
    for i in 1:N
        Ti = entropic_maps[i]
        for j in i:N
            Tj = entropic_maps[j]
            d = linear_entropic_transport_presampled(Ti, Tj, μ)
            A[i,j] = d
            A[j,i] = d
            next!(p)
        end
    end
    return A

end


function linear_entropic_segment_distances_periodic_presampled(X, cps, ε, ρ::Distributions.Distribution, N=1000)
    c(X, Y) = pairwise(PeriodicEuclidean([1,1]), X, Y).^2
    spt_μ = rand(ρ, N)
    entropic_maps = gather_entropic_maps_presampled(X, cps, ε, spt_μ', c)
    return pairwise_linear_entropic_dists_presampled(entropic_maps, spt_μ')
end

function linear_entropic_segment_distances_presampled(X, cps, ε, ρ::Distributions.Distribution, N=1000)
    c(X, Y) = pairwise(SqEuclidean(), X, Y).^2
    spt_μ = rand(ρ, N)
    println(size(spt_μ))
    entropic_maps = gather_entropic_maps_presampled(X, cps, ε, spt_μ, c)
    return pairwise_linear_entropic_dists_presampled(entropic_maps, spt_μ')
end

function gather_entropic_maps(x, cps, ε, ρ::Distributions.Distribution, c)
    entropic_maps = Vector{Function}()
    N = length(cps)
    D = size(x,2)

    for i in 1:N-1
        t0, t1 = cps[i], cps[i+1]
        n = t1 - t0 + 1
        spt_ν = x[t0:t1,:]
        spt_μ = rand(ρ, (n,D))
        C = c(spt_μ', spt_ν')
        μ = fill(1/n,n)
        T = entropic_transport_map(μ, μ, spt_ν, C, ε, SinkhornGibbs())
        push!(entropic_maps, T)
    end

    return entropic_maps

end

function linear_entropic_transport_QGKJL(Ti, Tj, dom=(-2,2))
    f(x, p) = (Ti([x]) - Tj([x]))^2
    i = solve(IntegralProblem(f, dom), QuadGKJL()).u
    return i^0.5
end

function linear_entropic_transport(Ti, Tj, domain)
    f(u, p) = sqeuclidean(Ti(u), Tj(u))
    prob = IntegralProblem(f, domain)
    sol = solve(prob, HCubatureJ())
    return sol.u
end


function pairwise_linear_entropic_dists_1d(entropic_maps)
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


function linear_entropic_segment_distances_1d(X, cps, ε, ρ::Distributions.Distribution)
    c(X,Y) = pairwise(SqEuclidean(), X,Y)
    entropic_maps = gather_entropic_maps(X, cps, ε, ρ, c)
    return pairwise_linear_entropic_dists_1d(entropic_maps)
end


function pairwise_entropic_segment_distances(x, cps, c, ε)
    N = size(cps, 1) - 1
    A = zeros(N,N)
    p = Progress(N*(N - 1) ÷ 2; dt=1.0)
    @sync @distributed for i in 1:N
        t00, t01 = cps[i], cps[i+1]
        Si = x[t00:t01,:]
        μ = fill(1/(t01 - t00 + 1), t01 - t00 + 1)
        for j in i+1:N
            t10, t11 = cps[j], cps[j+1]
            Sj = x[t10:t11,:]
            ν = fill(1/(t11 - t10 + 1), t11 - t10 +1)
            C = c(Si', Sj')
            d = sinkhorn2(μ, ν, C, ε)
            A[i,j] = d
            A[j,i] = d
            next!(p)
        end
    end
    return A
end

function pairwise_transport_1d(x, cps, c=sqeuclidean)
    N = size(cps, 1) - 1
    A = zeros(N,N)
    #p = Progress((N*(N - 1)) ÷ 2)
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
            #d = wasserstein(μ, ν; p=2)^2
            A[i,j] = d
            A[j,i] = d
    #        next!(p)
        end
    end
    return A

end

export linear_entropic_segment_distances
export linear_entropic_segment_distances_presampled
