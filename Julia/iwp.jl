using Base: Integer
using LinearAlgebra
using OrdinaryDiffEq
using ForwardDiff
using PyPlot
using ControlSystems
using Distributions
using GaussianProcesses
using BayesianOptimization

abstract type AbstractPolicy <: Function end
abstract type AbstractEnvironment end

struct LQRPolicy{T<:Real} <: AbstractPolicy
    A::Matrix{T}
    B::Vector{T}
    K::AbstractArray
    Q::Matrix{T}
    R::Matrix{T}
    max_ctrl::T
end
(u::LQRPolicy)(x) = clamp(-dot(u.K, x), -u.max_ctrl, u.max_ctrl)
(u::LQRPolicy)(p, x) = clamp(-dot(p, x), -u.max_ctrl, u.max_ctrl)

struct LinearPolicy{T<:Real} <: AbstractPolicy
    K::AbstractArray
    Q::Matrix{T}
    R::Matrix{T}
    max_ctrl::T
end
(u::LinearPolicy)(x) = clamp(-dot(u.K, x), -u.max_ctrl, u.max_ctrl)
(u::LinearPolicy)(p, x) = clamp(-dot(p, x), -u.max_ctrl, u.max_ctrl)

struct IWPParams{T}
    gravity::T
    mp::T
    mr::T
    m::T
    Jp::T
    Jr::T
    J::T
    lp::T
    lr::T
    l::T
    r::T
    max_ctrl::T
end

mutable struct IWPEnv{T<:Real} <: AbstractEnvironment
    params::IWPParams{T}
    policy::AbstractPolicy
    NX::Integer
    NU::Integer
    NT::Integer
    dt::T
    tspan::Tuple{T, T}
    trajectory::Array{T, 2}
    ctrl_input::Vector{T}
    loss::T
end
function Base.show(io::IO, params::IWPParams)
    for f in fieldnames(IWPParams)
        print(io, f)
        print(io, " => ")
        print(io, getfield(params, f))
        println()
    end
end

function IWPEnv(T::Type=Float32;
        gravity=T(9.81),
        mp=T(0.706),
        mr=T(0.223),
        m=T(mp+mr),
        Jp=T(0.00249),
        Jr=T(0.0014615),
        lr=T(0.18217),
        lp=T(lr/2),
        l=T(1/m*(mp*lp+mr*lr)),
        J=T(Jp + mp*lp*lp + mr*lr*lr),
        r=T(45/34*13/3),
        max_ctrl=T(5.0),
        # policy=LQRPolicy(
        #     rand(Float32, 4, 4), rand(Float32, 4), rand(Float32,4), 
        #     rand(Float32, 4, 4), Matrix{Float32}(undef, 1, 1), max_ctrl
        # ),
        policy=LinearPolicy(
                Vector{Float32}(undef, 4), Matrix{Float32}(undef, 4, 4), 
                Matrix{Float32}(undef, 1, 1), max_ctrl),
        dt=T(1/20),
        tspan=(T(0),T(10)),
    )

    params = IWPParams(
        T(gravity),
        T(mp),
        T(mr),
        T(m),
        T(Jp),
        T(Jr),
        T(J),
        T(lp),
        T(lr),
        T(l),
        T(r),
        T(max_ctrl)
    )

    u = IWPEnv(
        params,
        policy,
        4,
        1,
        length(range(tspan..., step=dt)),
        T(dt),
        T.(tspan),
        Array{T,2}(undef, (4,1)),
        Array{T,1}(undef, 1),
        T(0)
    )
    C = [1 0 0 0; 0 1 0 0];
    Q = C'*C
    R = Matrix{Float32}(undef, 1, 1)
    R[1,1] = 100.0
    # A, B, K, _ = lqr_gains(u; Q=Q, R=R[1])
    # u.policy.A[:] = A[:]
    # u.policy.B[:] = B[:]
    # u.policy.K[:] = K[:]
    u.policy.Q[:] = T.(Q)[:]
    u.policy.R[:] = T.(R)[:]
    u.policy.K[:] = [-3.0, -0.1, -0.5, -0.03]
    return u
end
function Base.show(io::IO, env::IWPEnv)
    print(io, "Reaction Wheel Pendulum Environment: ")
    print(io, "DataType=$(typeof(env).parameters[1]), ")
    print(io, "tspan=$(env.tspan), ")
    print(io, "dt=$(env.dt) ")
end

function _known_dynamics(env::IWPEnv, x, u)
    q1, q2, q1dot, q2dot = x
    J = env.params.J
    Jr = env.params.Jr
    m = env.params.m
    g = env.params.gravity
    l = env.params.l
    r = env.params.r
    return [
        x[3],
        x[4],
        1/(J*0.8) * (m*g*l*sin(q1) - r*u),  # Introduced a 20% error into J
        1/Jr * r * u
    ]
end

function _dynamics(env::IWPEnv, x, u)
    q1, q2, q1dot, q2dot = x
    J = env.params.J
    Jr = env.params.Jr
    m = env.params.m
    g = env.params.gravity
    l = env.params.l
    r = env.params.r
    return [
        x[3],
        x[4],
        1/J * (m*g*l*sin(q1) - r*u),
        1/Jr * r * u
    ]
end

function dynamics(env::IWPEnv, x, u)
    q1, q2, q1dot, q2dot = x
    J = env.params.J
    Jr = env.params.Jr
    m = env.params.m
    g = env.params.gravity
    l = env.params.l
    r = env.params.r
    return [
        x[3],
        x[4],
        1/J * (m*g*l*sin(q1) - r*u[1]),
        1/Jr * r * u[1]
    ]
end

function dynamics(env::IWPEnv, dx, x, u)
    q1, q2, q1dot, q2dot = x
    J = env.params.J
    Jr = env.params.Jr
    m = env.params.m
    g = env.params.gravity
    l = env.params.l
    r = env.params.r
    
    dx[1] = x[3]
    dx[2] = x[4]
    dx[3] = 1/J * (m*g*l*sin(q1) - r*u[1])
    dx[4] = 1/Jr * r * u[1]
    nothing
end


function lqr_gains(env::IWPEnv{T}; xbar=T[0,0,0,0], Q=I(4), R=1.0) where {T<:Real}
    A = ForwardDiff.jacobian(x->_known_dynamics(env, x, 0), xbar)
    B = ForwardDiff.derivative(u->_known_dynamics(env, xbar, u), 0)
    S = care(A,B,Q,R)
    K = R\B'*S
    A, B, K, S
end

function simulate(env::AbstractEnvironment, x0; tf=last(env.tspan), saveat=env.dt)
    OrdinaryDiffEq.solve(
        ODEProblem(
            ODEFunction((x,p,t) -> dynamics(env, x, env.policy(p,x))),
            x0,
            (first(env.tspan), tf),
            env.policy.K
        ),
        Tsit5(),
        # reltol=1e-8, abstol=1e-6,
        saveat=saveat
    )
end

function simulate(env::AbstractEnvironment, x0, k; tf=last(env.tspan), saveat=env.dt)
    OrdinaryDiffEq.solve(
        ODEProblem(
            ODEFunction((x,p,t) -> dynamics(env, x, env.policy(p,x))),
            x0,
            (first(env.tspan), tf),
            k
        ),
        Tsit5(),
        # reltol=1e-8, abstol=1e-6,
        saveat=saveat
    )
end

function simulate!(env::AbstractEnvironment, x0; tf=last(env.tspan), saveat=env.dt)
    env.trajectory = Array(simulate(env, x0; tf=tf, saveat=saveat))
    env.ctrl_input = mapslices(env.policy, env.trajectory, dims=1) |> vec
    env.loss = lqr_loss(env, env.trajectory)
end

function simulate!(env::AbstractEnvironment, x0, k; tf=last(env.tspan), saveat=env.dt)
    env.trajectory = Array(simulate(env, x0, k; tf=tf, saveat=saveat))
    env.ctrl_input = mapslices(env.policy, env.trajectory, dims=1) |> vec
    env.loss = lqr_loss(env, env.trajectory)
end

function lqr_loss(env::IWPEnv{T}, x) where {T<:Real}
    x = env.trajectory
    u = env.ctrl_input
    return sum(
        1/2*dot(x[:,i], env.policy.Q * x[:,i]) + 1/2*env.policy.R[1,1]*u[i]^2
        for i=1:size(x,2)
    )
end



## Bayesian Optimization to find acceptable linear policy gains from this point options

function find_gains_BO(env::AbstractEnvironment, reset::Bool=false; maxiters::Int=100)

    function f(env::AbstractEnvironment, k, x0=[1*π/180, 0.5, 0., 0])
        simulate!(env, x0, k)
        return env.loss
    end
    g(k) = f(env, k)

    # Choose as a model an elastic GP with input dimensions 2.
    # The GP is called elastic, because data can be appended efficiently.
    model = ElasticGPE(4,                            # 2 input dimensions
        mean = MeanConst(0.),         
        kernel = SEArd([0., 0., 0., 0.], 5.),
        logNoise = 0.,
        capacity = 3000)              # the initial capacity of the GP is 3000 samples.
    set_priors!(model.mean, [Normal(1, 2)])

    # Optimize the hyperparameters of the GP using maximum a posteriori (MAP) estimates every 50 steps
    modeloptimizer = MAPGPOptimizer(every = 50, noisebounds = [-4, 3],       # bounds of the logNoise
                kernbounds = [[-1, -1, -1, -1, 0], [4, 4, 4, 4, 10]],  # bounds of the 5 parameters GaussianProcesses.get_param_names(model.kernel)
                maxeval = 40)

    # modeloptimizer = NoModelOptimizer()

    opt = BOpt(g,
            model,
            UpperConfidenceBound(),                   # type of acquisition
            modeloptimizer,                        
            [-10., -10., -10., -10.], [0., 0., 0., 0.],                     # lowerbounds, upperbounds         
            # repetitions = 5,                          # evaluate the function for each input 5 times
            maxiterations = 200,                      # evaluate at 100 input positions
            sense = Min,                              # minimize the function
            acquisitionoptions = (method = :LD_LBFGS, # run optimization of acquisition function with NLopts :LD_LBFGS method
                        restarts = 5,       # run the NLopt method from 5 random initial conditions each time.
                        maxtime = 0.1,      # run the NLopt method for at most 0.1 second each time
                        maxeval = 1000),    # run the NLopt methods for at most 1000 iterations (for other options see https://github.com/JuliaOpt/NLopt.jl)
    verbosity = Progress)


    # x = Array{Array{Float64, 1}, 1}()
    # push!(x, K_estimate[:])
    # push!(x, K_true[:])
    # y = -f.(x)
    # append!(model, hcat(x...), y)

    opt = BOpt(g,
    model,
    #    UpperConfidenceBound(),
    ThompsonSamplingSimple(),
    # ExpectedImprovement(),
    modeloptimizer,                        
    [-10., -10., -10., -10.], [0., 0., 0., 0.], 
    maxiterations = maxiters,
    sense = Min,
    initializer_iterations = 0
    )

    
    result = boptimize!(opt)
    return result, opt
end