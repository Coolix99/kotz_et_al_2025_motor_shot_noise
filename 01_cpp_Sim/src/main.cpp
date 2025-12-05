#include "simulate_pde.hpp"
#include <iostream>
#include <chrono>
#include <sstream>
#include <zlib.h>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <cstdint>

// -----------------------------------------------------------------------------
// Helper: write a vector to gz file
// -----------------------------------------------------------------------------
template<typename T>
void gzwrite_vector(gzFile f, const std::vector<T>& v) {
    std::uint64_t n = v.size();
    gzwrite(f, &n, sizeof(n));
    if (n > 0)
        gzwrite(f, v.data(), sizeof(T) * n);
}

// -----------------------------------------------------------------------------
// Save compressed results (binary gzip)
// -----------------------------------------------------------------------------
void save_compressed(const std::string& fname, const spde::Params& p, const spde::Results& R, double runtime_s) {
    gzFile f = gzopen(fname.c_str(), "wb9");
    if (!f) throw std::runtime_error("Failed to open gzip file: " + fname);

    // --- Magic header ---
    const char magic[8] = {'S','P','D','E','0','0','1','\0'};
    gzwrite(f, magic, sizeof(magic));

    // --- Parameters ---
    gzwrite(f, &p.seed, sizeof(p.seed));
    gzwrite(f, &p.Nmotor, sizeof(p.Nmotor));
    gzwrite(f, &p.mu_a, sizeof(p.mu_a));
    gzwrite(f, &p.n, sizeof(p.n));
    gzwrite(f, &p.fstar, sizeof(p.fstar));
    gzwrite(f, &p.mu, sizeof(p.mu));
    gzwrite(f, &p.eta, sizeof(p.eta));
    gzwrite(f, &p.zeta, sizeof(p.zeta));
    gzwrite(f, &p.beta, sizeof(p.beta));
    gzwrite(f, &p.t_sub, sizeof(p.t_sub));
    gzwrite(f, &p.T, sizeof(p.T));
    gzwrite(f, &p.dt, sizeof(p.dt));
    gzwrite(f, &p.ZETA_GAMMA_THRESHOLD, sizeof(p.ZETA_GAMMA_THRESHOLD));
    gzwrite(f, &p.LAMBDA_SMALL_THRESHOLD, sizeof(p.LAMBDA_SMALL_THRESHOLD));

    int mode_int = static_cast<int>(p.mode);
    gzwrite(f, &mode_int, sizeof(mode_int));


    // --- Results metadata ---
    gzwrite(f, &R.n_coarse, sizeof(R.n_coarse));
    gzwrite(f, &R.n_nodes,  sizeof(R.n_nodes));
    gzwrite(f, &R.count_large_zeta_gamma, sizeof(R.count_large_zeta_gamma));
    gzwrite(f, &R.count_regular_poisson,  sizeof(R.count_regular_poisson));

    // --- Runtime in seconds ---
    gzwrite(f, &runtime_s, sizeof(runtime_s));

    // --- Matrices ---
    gzwrite_vector(f, R.t_coarse);
    gzwrite_vector(f, R.gamma_mat);
    gzwrite_vector(f, R.nplus_mat);
    gzwrite_vector(f, R.nminus_mat);

    gzclose(f);
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------
int main(int argc, char** argv) {
    using namespace spde;

    if (argc < 5) {
        std::cerr << "Usage: " << argv[0]
                  << " <basename> <seed> <Nmotor> <mu_a> [--param=value ...]\n";
        return 1;
    }

    std::string basename = argv[1];
    Params p;
    p.seed   = std::stoull(argv[2]);
    p.Nmotor = std::stod(argv[3]);
    p.mu_a   = std::stod(argv[4]);

    // --- Default values ---
    p.n      = 100;
    p.fstar  = 2.0;
    p.mu     = 10.0;
    p.eta    = 0.096;
    p.zeta   = 0.96;
    p.beta   = 10.0;
    p.t_sub  = 20.0;
    p.T      = 100.0;
    p.dt     = 1e-4;
    p.mode   = spde::SimulationMode::Poisson;
    p.ZETA_GAMMA_THRESHOLD = 13.0;
    p.LAMBDA_SMALL_THRESHOLD = 1e-3;

    std::string outdir = "./";

    // --- Parse optional arguments ---
    for (int i = 5; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "--wn") {
            p.mode = spde::SimulationMode::WhiteNoise;
            continue;
        }
        if (arg == "--cwn") {
            p.mode = spde::SimulationMode::ConstWhiteNoise;
            continue;
        }
        if (arg == "--cwn_open") {
            p.mode = spde::SimulationMode::ConstWhiteNoiseOpen;
            continue;
        }
        if (arg == "--cwn_periodic") {
            p.mode = spde::SimulationMode::ConstWhiteNoisePeriodic;
            continue;
        }
        auto eq = arg.find('=');
        if (eq == std::string::npos) continue;

        std::string key = arg.substr(0, eq);
        std::string val_str = arg.substr(eq + 1);

        if (key == "--outdir") {
            outdir = val_str;  // string, no conversion
            continue;
        }

        // numeric parameters
        double val = 0.0;
        try {
            val = std::stod(val_str);
        } catch (...) {
            std::cerr << "Warning: invalid numeric value for " << key << " = " << val_str << "\n";
            continue;
        }

        if (key == "--n") p.n = (int)val;
        else if (key == "--fstar") p.fstar = val;
        else if (key == "--mu") p.mu = val;
        else if (key == "--eta") p.eta = val;
        else if (key == "--zeta") p.zeta = val;
        else if (key == "--beta") p.beta = val;
        else if (key == "--t_sub") p.t_sub = val;
        else if (key == "--T") p.T = val;
        else if (key == "--dt") p.dt = val;
        else if (key == "--ZETA_GAMMA_THRESHOLD") p.ZETA_GAMMA_THRESHOLD = val;
        else if (key == "--LAMBDA_SMALL_THRESHOLD") p.LAMBDA_SMALL_THRESHOLD = val;
    }


    std::ostringstream fname;
    fname << outdir
          << basename
          << "_seed_" << p.seed;

    bool deterministic_mode = (p.Nmotor < 0.0);
    if (deterministic_mode) {
        p.mode = spde::SimulationMode::Deterministic;
        fname << "_Nmotor_infty";
    } else {
        fname << "_Nmotor_" << p.Nmotor;
    }

    // ---- Mode tags ----
    switch (p.mode) {
    case spde::SimulationMode::WhiteNoise:
        fname << "_wn";
        break;
    case spde::SimulationMode::ConstWhiteNoise:
        fname << "_cwn";
        break;
    case spde::SimulationMode::ConstWhiteNoiseOpen:
        fname << "_cwn_open";
        break;
    case spde::SimulationMode::ConstWhiteNoisePeriodic:
        fname << "_cwn_periodic";
        break;
    default:
        break;
    }

    fname << "_mu_a_" << p.mu_a << ".gz";

    std::cout << "Running simulation: " << fname.str() << std::endl;

    // --- Run simulation ---
    auto start = std::chrono::high_resolution_clock::now();
    spde::Results R;

    switch (p.mode) {
    case SimulationMode::Deterministic:
        std::cout << "Mode: deterministic (Nmotor→∞)\n";
        R = simulate_episode_deterministic(p);
        break;

    case SimulationMode::Poisson:
        std::cout << "Mode: Poisson (default motor stochasticity)\n";
        R = simulate_episode(p);
        break;

    case SimulationMode::WhiteNoise:
        std::cout << "Mode: white-noise approximation (Nmotor = "
                  << p.Nmotor << ")\n";
        R = simulate_episode_white_noise(p);
        break;

    case SimulationMode::ConstWhiteNoise:
        std::cout << "Mode: constant white-noise (default BC)\n";
        R = simulate_episode_const_white_noise(p);
        break;

    case SimulationMode::ConstWhiteNoiseOpen:
        std::cout << "Mode: constant white-noise (open BC)\n";
        R = simulate_episode_const_white_noise_open(p);
        break;

    case SimulationMode::ConstWhiteNoisePeriodic:
        std::cout << "Mode: constant white-noise (periodic BC)\n";
        R = simulate_episode_const_white_noise_periodic(p);
        break;
    }

    auto end = std::chrono::high_resolution_clock::now();
    double seconds = std::chrono::duration<double>(end - start).count();

    std::cout << "Done in " << seconds << " s\n";
    std::cout << "  Saturations: " << R.count_large_zeta_gamma
              << "  Regular Poisson: " << R.count_regular_poisson << "\n";

    try {
        save_compressed(fname.str(), p, R, seconds);
        std::cout << "Saved compressed results to " << fname.str() << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Error saving results: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
