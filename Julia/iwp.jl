using Base: Integer
using LinearAlgebra
using OrdinaryDiffEq
using PyPlot
using ControlSystems
using Distributions
# using GaussianProcesses
# using BayesianOptimization

struct IWPParams{T}
    gravity::T
    mp::T
    mr::T
    m::T
    Jp::T
    Jr::T
    lp::T
    lr::T
    l::T
    max_ctrl::T
end

mutable struct IWPEnv{T<:Real} <: AbstractEnvironment
    params::IWPParams{T}
    policy::Function
    NX::Integer
    NU::Integer
    NT::Integer
    dt::T
    tspan::Tuple{T, T}
    trajectory::Array{T, 2}
    ctrl_input::Vector{T}
    loss::T
end
function Base.show(io::IO, params::AcrobotParams)
    for f in fieldnames(AcrobotParams)
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
        Jp=T(0.1),
        Jr=T(0.2),
        lr=T(0.18217),
        lp=T(lr/2),
        l=T(1/m*(mp*lp+mr*lr)),
        max_ctrl=T(1.0),
        policy=ConstantPolicy([T(0)]),
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
        T(lp),
        T(lr),
        T(l),
        T(max_ctrl)
    )

    IWPEnv(
        params,
        policy,
        4,
        1,
        length(range(tspan..., step=dt)),
        T(dt),
        T.(tspan)
        Array{T,2}(undef, (4,1)),
        Array{T,1}(undef, 1),
        T(0)
    )

end