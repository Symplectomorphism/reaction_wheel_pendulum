include("iwp.jl")

env = IWPEnv()
x0 = [1*π/180, 0.5, 0, 0]
simulate!(env, x0)

A, B, K, S = lqr_gains(env; Q=env.policy.Q, R=env.policy.R);


function test_bo_vs_lqr(;iters::Int=1000, reset::Bool=false)
    res, opt = find_gains_BO(env; maxiters=100)

    unc_lqr_loss = simulate!(env, x0, K)
    bo_loss = simulate!(env, x0, res.observed_optimizer)

    while bo_loss > unc_lqr_loss && opt.iterations.i < iters
        if reset
            res, opt = find_gains_BO(env; maxiters=100) 
        else 
            res = boptimize!(opt)
        end
        bo_loss = simulate!(env, x0, res.observed_optimizer)
    end
    return bo_loss, unc_lqr_loss, res
end

bo_loss, unc_lqr_loss, res = test_bo_vs_lqr(iters=1000, reset=true)
@assert bo_loss <= unc_lqr_loss