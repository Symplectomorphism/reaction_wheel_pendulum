include("iwp.jl")

env = IWPEnv()
x0 = [1*π/180, 0.5, 0, 0]
simulate!(env, x0)

A, B, K, S = lqr_gains(env; Q=env.policy.Q, R=env.policy.R);


function test_bo_vs_lqr()
    res, opt = find_gains_BO(env; maxiters=100)

    unc_lqr_loss = simulate!(env, x0, K)
    bo_loss = simulate!(env, x0, res.observed_optimizer)

    while bo_loss > unc_lqr_loss && opt.iterations.i < 2000
        res = boptimize!(opt)
        bo_loss = simulate!(env, x0, res.observed_optimizer)
    end
    return bo_loss, unc_lqr_loss
end

bo_loss, unc_lqr_loss = test_bo_vs_lqr()
@assert bo_loss <= unc_lqr_loss