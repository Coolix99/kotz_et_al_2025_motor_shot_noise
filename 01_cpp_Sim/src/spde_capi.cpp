#include "simulate_pde.hpp"
#include <vector>
#include <cstring>
#include <iostream>

extern "C" int spde_simulate(const spde::Params* p_in,
                             double* t_out,
                             double* gamma_out,
                             double* nplus_out,
                             double* nminus_out)
{
    if (!p_in || !t_out || !gamma_out || !nplus_out || !nminus_out) {
        std::cerr << "spde_simulate(): null pointer argument\n";
        return -1;
    }

    try {
        spde::Params p = *p_in;

        // Mirror main.cpp behaviour:
        // negative Nmotor selects deterministic limit, regardless of passed mode
        if (p.Nmotor < 0.0) {
            if (p.mode == spde::SimulationMode::ThreeD_Poisson)
                p.mode = spde::SimulationMode::ThreeD_Deterministic;
            else if (p.mode != spde::SimulationMode::ThreeD_Deterministic)
                p.mode = spde::SimulationMode::Deterministic;
        }

        spde::Results R;

        // Dispatch to the correct solver based on mode
        switch (p.mode) {
        case spde::SimulationMode::Deterministic:
            // Deterministic limit (Nmotor → ∞)
            {
                spde::ParamsDeterministic pd = {p.n, p.fstar, p.mu, p.eta, p.zeta, p.beta, p.t_sub, p.T, p.dt, p.ZETA_GAMMA_THRESHOLD, p.seed, p.mu_a};
                R = spde::simulate_episode_deterministic(pd);
            }
            break;

        case spde::SimulationMode::Poisson:
            // Default stochastic model (finite Nmotor, Poisson events)
            {
                spde::ParamsPoisson pp = {p.n, p.fstar, p.mu, p.eta, p.zeta, p.beta, p.t_sub, p.T, p.dt, p.ZETA_GAMMA_THRESHOLD, p.seed, p.Nmotor, p.LAMBDA_SMALL_THRESHOLD, p.mu_a};
                R = spde::simulate_episode(pp);
            }
            break;

        case spde::SimulationMode::WhiteNoise:
            // White-noise approximation
            {
                spde::ParamsWhiteNoise pw = {p.n, p.fstar, p.mu, p.eta, p.zeta, p.beta, p.t_sub, p.T, p.dt, p.ZETA_GAMMA_THRESHOLD, p.seed, p.Nmotor, p.mu_a};
                R = spde::simulate_episode_white_noise(pw);
            }
            break;

        case spde::SimulationMode::ConstWhiteNoise:
            // Constant white-noise, default BC
            {
                spde::ParamsConstWhiteNoise pc = {p.n, p.fstar, p.mu, p.eta, p.zeta, p.beta, p.t_sub, p.T, p.dt, p.ZETA_GAMMA_THRESHOLD, p.seed, p.Nmotor, p.mu_a};
                R = spde::simulate_episode_const_white_noise(pc);
            }
            break;

        case spde::SimulationMode::ConstWhiteNoiseOpen:
            // Constant white-noise, open BC
            {
                spde::ParamsConstWhiteNoiseOpen po = {p.n, p.fstar, p.mu, p.eta, p.zeta, p.beta, p.t_sub, p.T, p.dt, p.ZETA_GAMMA_THRESHOLD, p.seed, p.Nmotor, p.mu_a};
                R = spde::simulate_episode_const_white_noise_open(po);
            }
            break;

        case spde::SimulationMode::ConstWhiteNoisePeriodic:
            // Constant white-noise, periodic BC
            {
                spde::ParamsConstWhiteNoisePeriodic pp = {p.n, p.fstar, p.mu, p.eta, p.zeta, p.beta, p.t_sub, p.T, p.dt, p.ZETA_GAMMA_THRESHOLD, p.seed, p.Nmotor, p.mu_a};
                R = spde::simulate_episode_const_white_noise_periodic(pp);
            }
            break;

        case spde::SimulationMode::CM_PS:
            // CM+PS model
            {
                spde::ParamsCMPS pcmps = {p.n, p.fstar, p.mu, p.eta, p.zeta, p.beta, p.t_sub, p.T, p.dt, p.ZETA_GAMMA_THRESHOLD, p.seed, p.Nmotor, p.ps_rate, p.LAMBDA_SMALL_THRESHOLD, p.mu_a, 0.0};
                R = spde::simulate_episode_cm_ps(pcmps);
            }
            break;

        case spde::SimulationMode::ThreeD_Deterministic:
            // 3D deterministic
            {
                spde::Params3DDet p3d = {p.n, p.fstar, p.mu, p.eta, p.zeta, p.beta, p.t_sub, p.T, p.dt, p.ZETA_GAMMA_THRESHOLD, p.seed, p.d_tilde_1, p.d_tilde_2, p.d_tilde_3, p.d_tilde_4, p.mu_a};
                R = spde::simulate_episode_3d_deterministic(p3d);
            }
            break;

        case spde::SimulationMode::ThreeD_Poisson:
            // 3D Poisson
            {
                spde::Params3D p3 = {p.n, p.fstar, p.mu, p.eta, p.zeta, p.beta, p.t_sub, p.T, p.dt, p.ZETA_GAMMA_THRESHOLD, p.seed, p.Nmotor, p.d_tilde_1, p.d_tilde_2, p.d_tilde_3, p.d_tilde_4, p.LAMBDA_SMALL_THRESHOLD, p.mu_a};
                R = spde::simulate_episode_3d(p3);
            }
            break;

        default:
            std::cerr << "spde_simulate(): unknown SimulationMode value\n";
            return -1;
        }

        const std::size_t n_coarse = R.n_coarse;
        const std::size_t N        = R.n_nodes;

        // Caller must allocate:
        //   t_out      : n_coarse
        //   gamma_out  : n_coarse * N
        //   nplus_out  : n_coarse * N
        //   nminus_out : n_coarse * N
        std::memcpy(t_out,      R.t_coarse.data(),       n_coarse     * sizeof(double));
        std::memcpy(gamma_out,  R.gamma_mat.data(),      n_coarse * N * sizeof(double));
        std::memcpy(nplus_out,  R.nplus_mat.data(),      n_coarse * N * sizeof(double));
        std::memcpy(nminus_out, R.nminus_mat.data(),     n_coarse * N * sizeof(double));

        // You can still use the return value as "number of coarse time points"
        return static_cast<int>(n_coarse);
    }
    catch (const std::exception& e) {
        std::cerr << "⚠️ spde_simulate(): caught C++ exception: " << e.what() << std::endl;
        return -1;
    }
    catch (...) {
        std::cerr << "⚠️ spde_simulate(): caught unknown exception." << std::endl;
        return -1;
    }
}


// CM+PS entry point 
extern "C" int spde_simulate_cmps(const spde::ParamsCMPS* p_in,
                                  double* t_out,
                                  double* gamma_out,
                                  double* nplus_out,
                                  double* nminus_out)
{
    if (!p_in || !t_out || !gamma_out || !nplus_out || !nminus_out) {
        std::cerr << "spde_simulate_cmps(): null pointer argument\n";
        return -1;
    }

    try {
        spde::ParamsCMPS p = *p_in;

        auto to_base_params = [&](spde::SimulationMode mode) {
            spde::Params base{};
            base.n = p.n;
            base.fstar = p.fstar;
            base.mu = p.mu;
            base.eta = p.eta;
            base.zeta = p.zeta;
            base.beta = p.beta;
            base.t_sub = p.t_sub;
            base.T = p.T;
            base.dt = p.dt;
            base.ZETA_GAMMA_THRESHOLD = p.ZETA_GAMMA_THRESHOLD;
            base.seed = p.seed;
            base.Nmotor = p.Nmotor;
            base.LAMBDA_SMALL_THRESHOLD = p.LAMBDA_SMALL_THRESHOLD;
            base.ps_rate = p.ps_rate;
            base.d_tilde_1 = 0.0;
            base.d_tilde_2 = 0.0;
            base.d_tilde_3 = 0.0;
            base.d_tilde_4 = 0.0;
            base.mode = mode;
            base.mu_a = p.mu_a;
            return base;
        };

        // Deterministic override always wins
        if (p.Nmotor < 0.0) {
            std::cout << "spde_simulate_cmps(): Nmotor<0, falling back to Deterministic mode.\n";
            spde::Params base = to_base_params(spde::SimulationMode::Deterministic);
            return spde_simulate(&base, t_out, gamma_out, nplus_out, nminus_out);
        }

        // Negative ps_rate → Poisson fallback (same rule as main.cpp)
        if (p.ps_rate < 0.0) {
            std::cout << "spde_simulate_cmps(): negative ps_rate, falling back to Poisson mode.\n";
            spde::Params base = to_base_params(spde::SimulationMode::Poisson);
            return spde_simulate(&base, t_out, gamma_out, nplus_out, nminus_out);
        }

        // Proper CM+PS execution
        std::cout << "spde_simulate_cmps(): running CM+PS simulation (ps_rate="
                  << p.ps_rate << ")\n";
        spde::Results R = spde::simulate_episode_cm_ps(p);

        const std::size_t nt = R.n_coarse;
        const std::size_t ns = R.n_nodes;

        std::memcpy(t_out,      R.t_coarse.data(),   nt       * sizeof(double));
        std::memcpy(gamma_out,  R.gamma_mat.data(),  nt * ns  * sizeof(double));
        std::memcpy(nplus_out,  R.nplus_mat.data(),  nt * ns  * sizeof(double));
        std::memcpy(nminus_out, R.nminus_mat.data(), nt * ns  * sizeof(double));

        return static_cast<int>(nt);
    }
    catch (const std::exception& e) {
        std::cerr << "spde_simulate_cmps(): exception: " << e.what() << "\n";
        return -1;
    }
    catch (...) {
        std::cerr << "spde_simulate_cmps(): unknown exception\n";
        return -1;
    }
}
