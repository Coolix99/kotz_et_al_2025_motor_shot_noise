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

        // negative Nmotor selects deterministic limit, regardless of passed mode
        if (p.Nmotor < 0.0) {
            p.mode = spde::SimulationMode::Deterministic;
        }

        spde::Results R;

        // Dispatch to the correct solver based on mode
        switch (p.mode) {
        case spde::SimulationMode::Deterministic:
            // Deterministic limit (Nmotor → ∞)
            R = spde::simulate_episode_deterministic(p);
            break;

        case spde::SimulationMode::Poisson:
            // Default stochastic model (finite Nmotor, Poisson events)
            R = spde::simulate_episode(p);
            break;

        case spde::SimulationMode::WhiteNoise:
            // White-noise approximation
            R = spde::simulate_episode_white_noise(p);
            break;

        case spde::SimulationMode::ConstWhiteNoise:
            // Constant white-noise, default BC
            R = spde::simulate_episode_const_white_noise(p);
            break;

        case spde::SimulationMode::ConstWhiteNoiseOpen:
            // Constant white-noise, open BC
            R = spde::simulate_episode_const_white_noise_open(p);
            break;

        case spde::SimulationMode::ConstWhiteNoisePeriodic:
            // Constant white-noise, periodic BC
            R = spde::simulate_episode_const_white_noise_periodic(p);
            break;

        default:
            std::cerr << "spde_simulate(): unknown SimulationMode value\n";
            return -1;
        }

        const std::size_t n_coarse = R.n_coarse;
        const std::size_t N        = R.n_nodes;

        std::memcpy(t_out,      R.t_coarse.data(),       n_coarse     * sizeof(double));
        std::memcpy(gamma_out,  R.gamma_mat.data(),      n_coarse * N * sizeof(double));
        std::memcpy(nplus_out,  R.nplus_mat.data(),      n_coarse * N * sizeof(double));
        std::memcpy(nminus_out, R.nminus_mat.data(),     n_coarse * N * sizeof(double));

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
