using Base: Integer
using LinearAlgebra
using OrdinaryDiffEq
using ForwardDiff
using PyPlot
using ControlSystems
using Distributions
# using GaussianProcesses
# using BayesianOptimization

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
(u::LQRPolicy)(p, x) = clamp(-dot(u.K, x), -u.max_ctrl, u.max_ctrl)

struct LinearPolicy{T<:Real} <: AbstractPolicy
    K::AbstractArray
    Q::Matrix{T}
    R::Matrix{T}
    max_ctrl::T
end
(u::LinearPolicy)(x) = clamp(-dot(u.K, x), -u.max_ctrl, u.max_ctrl)
(u::LinearPolicy)(p, x) = clamp(-dot(u.K, x), -u.max_ctrl, u.max_ctrl)

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
        dt=T(1/1000),
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
    A = ForwardDiff.jacobian(x->_dynamics(env, x, 0), xbar)
    B = ForwardDiff.derivative(u->_dynamics(env, xbar, u), 0)
    S = care(A,B,Q,R)
    K = R\B'*S
    A, B, K, S
end

function simulate(env::AbstractEnvironment, x0, tf=last(env.tspan); saveat=env.dt)
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

function simulate!(env::AbstractEnvironment, x0, tf=last(env.tspan); saveat=env.dt)
    env.trajectory = Array(simulate(env, x0, tf, saveat=saveat))
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
function f(k)
    
end