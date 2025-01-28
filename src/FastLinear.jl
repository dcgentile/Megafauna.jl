module FastLinear

using Distributions: DiscreteNonParametricStats
using ProgressMeter
using OptimalTransport
using Distances
using Distributions
using Distributed
using Integrals
using Statistics

function wrap_to_torus(point)
    return mod1.(point, 1)
end

# Updates: Changed the sampling scheme from uniform over the torus to sampling from a gaussian overlaid on the torus
function get_segment_potentials(x, cps, ε, a, b, c)
    N = size(cps,1)
    D = size(x,2)
    potentials = Vector{Vector{Float64}}()
    @showprogress for i in 1:N-1
        t0, t1 = cps[i], cps[i+1]
        n = t1 - t0 + 1
        #Uniform sampling
        #spt_μ = rand(Uniform(a,b), (n,D))

        """
        #Centered at 'origin' of Ramachandran plot: Gaussian sampling
        mean = [0.5, 0.5]
        σ_1 = 0.05
        Σ_1 = [σ_1^2 0.0; 0.0 σ_1^2]
        gaussian = MvNormal(mean, Σ_1)
        spt_μ = rand(gaussian, n)
        spt_μ = transpose(spt_μ)
        """

        #Centered at 'center of mass' of data
        mean = [0.25, 0.76]
        σ_1 = 0.05
        Σ_1 = [σ_1^2 0.0; 0.0 σ_1^2]
        gaussian = MvNormal(mean, Σ_1)
        samples_COM = rand(gaussian, n)
        #samples_COM = transpose(spt_μ)
        
        wrapped_data_x_COM = wrap_to_torus.(eachrow(samples_COM[1,:]))
        wrapped_data_x_COM = reduce(vcat, wrapped_data_x_COM)
        
        wrapped_data_y_COM = wrap_to_torus.(eachrow(samples_COM[2,:]))
        wrapped_data_y_COM = reduce(vcat, wrapped_data_y_COM)
        
        wrapped_data_COM = hcat(wrapped_data_x_COM, wrapped_data_y_COM)

        spt_μ = wrapped_data_COM
        
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

################################################################################################

"""
# Monte carlo integration on the uniform distribution over 2D torus ... Takes around 35 minutes to compute distances for 200,000 pt segment
function linear_entropic_transport(Ti, Tj)
    f(u, p) = sqeuclidean(Ti(u), Tj(u))
    domain = ([0,0], [1,1])
    prob = IntegralProblem(f, domain)
    sol = solve(prob, HCubatureJL())
    return sol.u
end
"""

"""
Goal: monte carlo integration on a Gaussian overlaid over 2D torus 
 - V1: isotropic Gaussian centered at 0.5, 0.5 (done)
 - V2: isotropic Gaussian centered at the center of mass (barycenter of data points) (to do)
 - V3: non isotropic Gaussian (to do)
"""

"""
# Attempt 1: directly sample from a gaussian ... Takes around 90 minutes to compute distances for 200,000 pts (sampling 100 pts)
function linear_entropic_transport(Ti, Tj; num_samples=100, mean=[0.5, 0.5], sigma=0.1)
    f(u) = sqeuclidean(Ti(u), Tj(u))
    
    # Data centered at 'origin'
    #σ_1 = 0.05
    #Σ_1 = [σ_1^2 0.0; 0.0 σ_1^2]
    #gaussian = MvNormal(mean, Σ_1)
    #samples = rand(gaussian, num_samples)
    #samples = transpose(samples)

    # Data centered at center of mass
    mean = [0.25, 0.76]
    σ_1 = 0.05
    Σ_1 = [σ_1^2 0.0; 0.0 σ_1^2]
    gaussian = MvNormal(mean, Σ_1)
    samples_COM = rand(gaussian, num_samples)
    #samples_COM = transpose(spt_μ)
        
    wrapped_data_x_COM = wrap_to_torus.(eachrow(samples_COM[1,:]))
    wrapped_data_x_COM = reduce(vcat, wrapped_data_x_COM)
    wrapped_data_y_COM = wrap_to_torus.(eachrow(samples_COM[2,:]))
    wrapped_data_y_COM = reduce(vcat, wrapped_data_y_COM)
    wrapped_data_COM = hcat(wrapped_data_x_COM, wrapped_data_y_COM)

    samples = wrapped_data_COM

    tot = sum(f, eachrow(samples))

    #print(tot)
    integral_estimate = tot/num_samples

    return integral_estimate
end
"""

#to try: integrate discrete samples via integral package

"""
# Attempt 2: multiply f by Gaussian pdf and numerically integrate over entire torus ... (Much slower than the above...)
function linear_entropic_transport_V1(Ti, Tj)
    μ_1 = [0.5, 0.5]
    σ_1 = 0.05
    Σ_1 = [σ_1^2 0.0; 0.0 σ_1^2]
    dist = MvNormal(μ_1, Σ_1)

    # Evaluating the Gaussian pdf is incredibly slow...
    f(u, p) = pdf(dist, u) # * sqeuclidean(Ti(u), Tj(u))
    domain = ([0, 0], [1, 1])
    prob = IntegralProblem(f, domain)
    sol = solve(prob, VEGAS())
    return sol.u
end
"""

"""
# Attempt 3: use 'SampledIntegralProblem' in Integrals package... Takes around 75 minutes to compute distances for 200,000 pts (sampling 100 pts)
function linear_entropic_transport(Ti, Tj; num_samples=100, mean=[0.5, 0.5], sigma=0.1)
    f = u -> sqeuclidean(Ti(u), Tj(u))
    # Data centered at 'origin'
    σ_1 = 0.05
    Σ_1 = [σ_1^2 0.0; 0.0 σ_1^2]
    gaussian = MvNormal(mean, Σ_1)
    samples = rand(gaussian, num_samples)
    #samples = transpose(samples)

    y = [f(samples[:, i]) for i in 1:num_samples]

    x = 1:num_samples

    problem = SampledIntegralProblem(y, x)
    method = TrapezoidalRule()
    sol = solve(problem, method)
    return sol.u
end
"""

################################################################################################

"""
# Original LEOT segment distance method
function linear_entropic_segment_distances(entropic_maps)
    N = size(entropic_maps, 1)
    print(N)
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
"""

# Updated LEOT segment distance method by pre-sampling
function linear_entropic_segment_distances(entropic_maps)
    # initialize distance matrix
    N = size(entropic_maps, 1)
    print(N)
    A = zeros(N,N)
    p = Progress((N*(N - 1)) ÷ 2)

    # pre-sample data from Gaussian centered at 'origin'
    mean = [0.5, 0.5]
    σ_1 = 0.05
    Σ_1 = [σ_1^2 0.0; 0.0 σ_1^2]
    gaussian = MvNormal(mean, Σ_1)
    num_samples = 1000
    samples = rand(gaussian, num_samples)

    T_eval = Vector{Vector{Float64}}(undef, N)

    @sync @distributed for i in 1:N
        Ti = entropic_maps[i]
        # Calculate the result for each sample and assign to T_eval[i]
        local_result = [Ti(samples[:, k]) for k in 1:num_samples]
        T_eval[i] = local_result
    end
    
    @sync @distributed  for i in 1:N
        for j in i:N
            d = (sum(abs.(T_eval[i]-T_eval[j]).^2)/num_samples)^.5
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
    p = Progress(N*(N - 1) / 2; dt=1.0)
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

end
