using Distributions
using ProgressMeter
using OptimalTransport
using Distances
using Distributions
using Distributed
using Integrals

function get_segment_potentials(x, cps, ε, a, b, c)
    N = size(cps,1)
    D = size(x,2)
    potentials = Vector{Vector{Float64}}()
    @showprogress for i in 1:N-1
        t0, t1 = cps[i], cps[i+1]
        n = t1 - t0 + 1
        spt_μ = rand(Uniform(a,b), (n,D))
        spt_ν = x[t0:t1,:]
        μ = fill(1/n, n)
        C = c(spt_μ', spt_ν')
        f, g = sinkhorn_potentials(μ, μ, C, ε, SinkhornGibbs())
        push!(potentials, g)
    end
    return potentials
end

function build_entropic_map(ν, g, ε)
    N = size(ν, 1)
    function T(x)
        b = zeros(N)
        for i in 1:N
            b[i] = exp(1/ε * (g[i] - 0.5 * sqeuclidean(x,ν[i,:])))
        end
        return sum(b .* ν) / sum(b)
    end
    return T
end

function gather_entropic_maps(x, cps, potentials, ε)
    entropic_maps = Vector{Function}()
    N = length(cps)
    for i in 1:N-1
        t0, t1 = cps[i], cps[i+1]
        spt_ν = x[t0:t1,:]
        g = potentials[i]
        T = build_entropic_map(spt_ν, g, ε)
        push!(entropic_maps, T)
        end
    return entropic_maps
end

function linear_entropic_transport_QGKJL(Ti, Tj)
    f(x, p) = (Ti(x) - Tj(x))^2
    i = solve(IntegralProblem(f, (-2,2)), QuadGKJL()).u
    return i
end

function QGKJL_segment_distances(entropic_maps)
    N = length(entropic_maps)
    A = zeros(N,N)
    p = Progress(N*(N - 1) ÷ 2; dt=1.0)
    @sync @distributed for i in 1:N
        Ti = entropic_maps[i]
        for j in i+1:N
            Tj = entropic_maps[j]
            d = linear_entropic_transport_QGKJL(Ti, Tj)
            A[i,j] = d
            A[j,i] = d
            next!(p)
        end
    end
    return A
end

function linear_entropic_transport(Ti, Tj)
    f(u, p) = sqeuclidean(Ti(u), Tj(u))
    domain = ([0,0], [1,1])
    prob = IntegralProblem(f, domain)
    sol = solve(prob, VEGASMC())
    return sol.u
end

function linear_entropic_segment_distances(entropic_maps)
    N = size(entropic_maps, 1)
    A = zeros(N,N)
    p = Progress((N*(N - 1)) ÷ 2)
    @sync @distributed  for i in 1:N
        Ti = entropic_maps[i]
        for j in i:N
            Tj = entropic_maps[j]
            d = linear_entropic_transport(Ti, Tj)
            A[i,j] = d
            A[j,i] = d
            next!(p)
        end
    end
    return A
end


function entropic_segment_distances(x, cps, c, ε)
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

function pairwise_transport_1d(x, cps)
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
            d = wasserstein(μ, ν; p=2)^2
            A[i,j] = d
            A[j,i] = d
    #        next!(p)
        end
    end
    return A

end

export get_segment_potentials
export gather_entropic_maps
export linear_entropic_segment_distances
export pairwise_transport_1d
