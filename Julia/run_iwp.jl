using LinearAlgebra
using OrdinaryDiffEq
using ControlSystems
using PyPlot
using GaussianProcesses, BayesianOptimization, Distributions

const Kt = 1/(60.3e-3)
const r = 45/34*13/3        # Gear and belt ratio
const mp = 0.706
# const mr = 0.56424  # m_resin + 4*m_steel = 0.084 + 4 * 0.12946
const mr = 0.223
const m = mp + mr
const lr = 0.18217
const lp = lr/2
const l = (mp*lp + mr*lr)/m
const g = 9.81
const Ip_com = 0.00249
const I_epoxy = 1100 * 7.919e4 / 1e9 * (0.18/2)^2
const I_steel = 9.484e-4
const I_resin = 7.559e-4
const I_resin_new = 4.64e-4
# const I1 = Ip_com + mp*lp^2 + mr*lr^2
# const I2 = I_resin_new + 4*I_steel
const I2 = I_resin + I_epoxy
const I1_estimate = Ip_com + mp*lp^2 + mr*lr^2
const I1_true = I1_estimate*(1 + 0.22);
const τ_max = 0.2
const γ1 = 2. # 1.5*ceil(m*g*l)
const γ2 = 25. # 3*ceil(I1/I2) * γ1 / (γ1 - m*g*l)
const k = (1.0, 10.0)
I1 = I1_true

function compute_cost(sol::ODESolution, par::AbstractArray)
    C = [1 0 0 0; 0 1 0 0];
    Q = C' * C
    R = 100.0
    ctrl(u) = -dot(par,u)
    return sum( 1/2*dot(sol.u[i], Q*sol.u[i]) + 1/2*R*ctrl(sol.u[i])^2 for i=1:length(sol) )
end

function compute_lqr_gain(I1)
    A = [0 0 1 0; 0 0 0 1; m*g*l/I1 0 0 0; 0 0 0 0]
    B = r*[0; 0; -1/I1; 1/I2]
    C = [1 0 0 0; 0 1 0 0];
    sys = ss(A, B, C, 0)
    Q = C' * C
    # Q = 1.0*I
    R = 100.0*I
    L = lqr(sys, Q, R)
    return L
end

compute_lqr_estimate(u) = -(K_estimate * u)[1]
compute_lqr_true(u) = -(K_true * u)[1]

function compute_ida_pbc(u)
    q1, q2, q1dot, q2dot = u
    qbar = q2 + γ2*q1
    qbar_dot = q2dot + γ2*q1dot
    u_es = γ1*sin(q1) + k[1]*sin(qbar)
    u_di = k[2]*qbar_dot
    return 1/r*( u_es + u_di )
end

function compute_switching_control(u)
    q1, q2, q1dot, q2dot = u
    τ = compute_ida_pbc(u)
    if 1 - cos(q1) + 1 - cos(q2) + abs(q1) + abs(q2) < 0.1
        τ = compute_lqr_estimate(u)
        # τ = compute_lqr_true(u)
    end
    return τ
end

function iwp_eom!(du, u, p, t)
    # τ = clamp(compute_lqr_estimate(u), -τ_max, τ_max)
    # τ = clamp(compute_lqr_true(u), -τ_max, τ_max)
    # τ = clamp(compute_ida_pbc(u), -τ_max, τ_max)
    # τ = clamp(compute_switching_control(u), -τ_max, τ_max)

    τ = clamp(-dot(p,u), -τ_max, τ_max)

    du[1] = u[3]
    du[2] = u[4]
    du[3] = 1/I1 * (m*g*l*sin(u[1]) - r*τ)
    du[4] = 1/I2 * r*τ
end

function compute_control(sol::ODESolution)
    # return hcat( [clamp((-K*sol.u[i])[1], -τ_max, τ_max) for i = 1:length(sol)]... )'
    # return hcat( [clamp(compute_ida_pbc(sol.u[i]), -τ_max, τ_max) for i = 1:length(sol)]... )'
    # return hcat( [clamp(compute_lqr_estimate(sol.u[i]), -τ_max, τ_max) for i = 1:length(sol)]... )'
    return hcat( [clamp(compute_lqr_true(sol.u[i]), -τ_max, τ_max) for i = 1:length(sol)]... )'
end

K_estimate = compute_lqr_gain(I1_estimate)
K_true = compute_lqr_gain(I1_true)
u0 = [1*π/180, 0.5, 0, 0]
tspan = (0.0, 2.0)
p = K_true
prob = ODEProblem(iwp_eom!, u0, tspan, p)
sol = solve(prob, Tsit5(), saveat=0.01, dtmax=0.01)


function f(x)
    prob = ODEProblem(iwp_eom!, u0, tspan, x)
    sol = solve(prob, Tsit5(), saveat=0.01, dtmax=0.01)
    return compute_cost(sol, x)
end



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
                                kernbounds = [[-1, -1, -1, -1, 0], [4, 4, 4, 4, 10]],  # bounds of the 3 parameters GaussianProcesses.get_param_names(model.kernel)
                                maxeval = 40)

# modeloptimizer = NoModelOptimizer()

opt = BOpt(f,
            model,
            UpperConfidenceBound(),                   # type of acquisition
            modeloptimizer,                        
            [-10., -10., -10., -10.], [0., 0., 0., 0.],                     # lowerbounds, upperbounds         
            repetitions = 5,                          # evaluate the function for each input 5 times
            maxiterations = 1000,                      # evaluate at 100 input positions
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

opt = BOpt(f,
           model,
        #    UpperConfidenceBound(),
           ThompsonSamplingSimple(),
           modeloptimizer,                        
           [-10., -10., -10., -10.], [0., 0., 0., 0.], 
           maxiterations = 1000,
           sense = Min,
           initializer_iterations = 0
          )

                     
result = boptimize!(opt)

















################################################################################
## Plotting

# fig = figure(1)
# fig.clf()
# ax = fig.add_subplot(2,1,1)
# ax.plot(sol.t, getindex.(sol.u, 1), label=L"$q_1$")
# ax.plot(sol.t, getindex.(sol.u, 3), label=L"$\dot{q}_1$")
# ax.legend(fontsize=15)
# ax = fig.add_subplot(2,1,2)
# ax.plot(sol.t, getindex.(sol.u, 2), label=L"$q_2$")
# ax.plot(sol.t, getindex.(sol.u, 4), label=L"$\dot{q}_2$")
# ax.legend(fontsize=15)
# ax.set_xlabel("time", fontsize=15)

# fig = figure(2)
# fig.clf()
# ax = fig.add_subplot(1,1,1)
# ax.plot(sol.t, compute_control(sol), label=L"$\tau$")
# ax.set_xlabel("time", fontsize=15)
# ax.legend(fontsize=15)