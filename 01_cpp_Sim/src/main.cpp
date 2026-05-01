#include "simulate_pde.hpp"
#include <iostream>
#include <chrono>
#include <sstream>
#include <zlib.h>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <cstdint>

// Helper: write a vector to gz file
template<typename T>
void gzwrite_vector(gzFile f, const std::vector<T>& v) {
    std::uint64_t n = v.size();
    gzwrite(f, &n, sizeof(n));
    if (n > 0)
        gzwrite(f, v.data(), sizeof(T) * n);
}

struct ParamKV {
    const char* name;
    double value;
};

static void gzwrite_param(gzFile f, const ParamKV& p) {
    char namebuf[32] = {0};
    std::strncpy(namebuf, p.name, sizeof(namebuf) - 1);
    gzwrite(f, namebuf, sizeof(namebuf));
    gzwrite(f, &p.value, sizeof(double));
}


// Save compressed results (binary gzip)
void save_compressed(const std::string& fname,
                     const spde::Params& p,
                     const spde::Results& R,
                     double runtime_s)
{
    gzFile f = gzopen(fname.c_str(), "wb9");
    if (!f) throw std::runtime_error("Failed to open gzip file: " + fname);

    // Magic + version
    const char magic[8] = {'S','P','D','E','0','0','2','\0'};
    gzwrite(f, magic, sizeof(magic));

    uint32_t header_version = 3;
    gzwrite(f, &header_version, sizeof(header_version));

    uint32_t model_id = static_cast<uint32_t>(p.mode);
    gzwrite(f, &model_id, sizeof(model_id));

    // Parameter list
    std::vector<ParamKV> params = {
        {"seed",   double(p.seed)},
        {"n",      double(p.n)},
        {"fstar",  p.fstar},
        {"mu",     p.mu},
        {"mu_a",   p.mu_a},
        {"eta",    p.eta},
        {"zeta",   p.zeta},
        {"beta",   p.beta},
        {"t_sub",  p.t_sub},
        {"T",      p.T},
        {"dt",     p.dt},
        {"ZETA_GAMMA_THRESHOLD",   p.ZETA_GAMMA_THRESHOLD},
    };

    // Add mode-specific parameters
    if (p.mode == spde::SimulationMode::Poisson || p.mode == spde::SimulationMode::WhiteNoise ||
        p.mode == spde::SimulationMode::ConstWhiteNoise || p.mode == spde::SimulationMode::ConstWhiteNoiseOpen ||
        p.mode == spde::SimulationMode::ConstWhiteNoisePeriodic || p.mode == spde::SimulationMode::CM_PS ||
        p.mode == spde::SimulationMode::ThreeD_Poisson) {
        params.push_back({"Nmotor", p.Nmotor});
    }

    if (p.mode == spde::SimulationMode::Poisson || p.mode == spde::SimulationMode::CM_PS ||
        p.mode == spde::SimulationMode::ThreeD_Poisson) {
        params.push_back({"LAMBDA_SMALL_THRESHOLD", p.LAMBDA_SMALL_THRESHOLD});
    }

    if (p.mode == spde::SimulationMode::CM_PS) {
        params.push_back({"ps_rate", p.ps_rate});
    }

    if (p.mode == spde::SimulationMode::ThreeD_Deterministic || p.mode == spde::SimulationMode::ThreeD_Poisson) {
        params.push_back({"d_tilde_1", p.d_tilde_1});
        params.push_back({"d_tilde_2", p.d_tilde_2});
        params.push_back({"d_tilde_3", p.d_tilde_3});
        params.push_back({"d_tilde_4", p.d_tilde_4});
    }

    uint32_t n_params = static_cast<uint32_t>(params.size());
    gzwrite(f, &n_params, sizeof(n_params));

    for (const auto& kv : params)
        gzwrite_param(f, kv);

    // Results metadata 
    gzwrite(f, &R.n_coarse, sizeof(R.n_coarse));
    gzwrite(f, &R.n_nodes,  sizeof(R.n_nodes));
    gzwrite(f, &R.count_large_zeta_gamma, sizeof(R.count_large_zeta_gamma));
    gzwrite(f, &R.count_regular_poisson,  sizeof(R.count_regular_poisson));
    gzwrite(f, &R.count_poisson_overflow, sizeof(R.count_poisson_overflow));
    gzwrite(f, &R.count_invalid_denom, sizeof(R.count_invalid_denom));
    gzwrite(f, &runtime_s, sizeof(runtime_s));

    // Data
    gzwrite_vector(f, R.t_coarse);
    gzwrite_vector(f, R.gamma_mat);
    gzwrite_vector(f, R.nplus_mat);
    gzwrite_vector(f, R.nminus_mat);
    // 3D model has 4 nplus and 4 nminus
    gzclose(f);
}

static spde::SimulationMode parse_mode(const std::string& s) {
    if (s == "Deterministic")          return spde::SimulationMode::Deterministic;
    if (s == "Poisson")                return spde::SimulationMode::Poisson;
    if (s == "WhiteNoise")             return spde::SimulationMode::WhiteNoise;
    if (s == "ConstWhiteNoise")        return spde::SimulationMode::ConstWhiteNoise;
    if (s == "ConstWhiteNoiseOpen")    return spde::SimulationMode::ConstWhiteNoiseOpen;
    if (s == "ConstWhiteNoisePeriodic")return spde::SimulationMode::ConstWhiteNoisePeriodic;
    if (s == "CM_PS")                  return spde::SimulationMode::CM_PS;
    if (s == "3D_deterministic") return spde::SimulationMode::ThreeD_Deterministic;
    if (s == "3D_poisson")       return spde::SimulationMode::ThreeD_Poisson;
    throw std::invalid_argument("Unknown mode: " + s);
}


// Main
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
    p.mode   = SimulationMode::Poisson;
    p.ZETA_GAMMA_THRESHOLD = 13.0;
    p.LAMBDA_SMALL_THRESHOLD = 1e-3;
    // 3D defaults
    p.d_tilde_1 = 0.220;
    p.d_tilde_2 = 0.337;
    p.d_tilde_3 = 0.296;
    p.d_tilde_4 = 0.117;

    // --- CM+PS default ---
    double ps_rate = 1.0;

    std::string outdir = "./results/";

    // --- Parse optional arguments ---
    for (int i = 5; i < argc; ++i) {
        std::string arg(argv[i]);

        if (arg == "--wn") {
            p.mode = SimulationMode::WhiteNoise;
            continue;
        }
        if (arg == "--cwn") {
            p.mode = SimulationMode::ConstWhiteNoise;
            continue;
        }
        if (arg == "--cwn_open") {
            p.mode = SimulationMode::ConstWhiteNoiseOpen;
            continue;
        }
        if (arg == "--cwn_periodic") {
            p.mode = SimulationMode::ConstWhiteNoisePeriodic;
            continue;
        }
        if (arg == "--cm_ps") {
            p.mode = SimulationMode::CM_PS;
            continue;
        }
        if (arg == "--3d_det") {
            p.mode = SimulationMode::ThreeD_Deterministic;
            continue;
        }
        if (arg == "--3d_poi") {
            p.mode = SimulationMode::ThreeD_Poisson;
            continue;
        }
        auto eq = arg.find('=');
        if (eq == std::string::npos) continue;

        std::string key = arg.substr(0, eq);
        std::string val_str = arg.substr(eq + 1);

        if (key == "--outdir") {
            outdir = val_str;
            continue;
        }

        double val = 0.0;
        try {
            val = std::stod(val_str);
        } catch (...) {
            std::cerr << "Warning: invalid numeric value for " << key << "\n";
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
        else if (key == "--ps_rate") ps_rate = val;
    }


    if (p.Nmotor < 0.0) {
        if (p.mode == SimulationMode::ThreeD_Poisson)
            p.mode = SimulationMode::ThreeD_Deterministic;
        else if (p.mode != SimulationMode::ThreeD_Deterministic)
            p.mode = SimulationMode::Deterministic;
    }

    bool deterministic_mode = (p.mode == SimulationMode::Deterministic || p.mode == SimulationMode::ThreeD_Deterministic);

    // Filename
    std::ostringstream fname;
    fname << outdir << basename << "_seed_" << p.seed;

    if (deterministic_mode)
        fname << "_Nmotor_infty";
    else
        fname << "_Nmotor_" << p.Nmotor;

    switch (p.mode) {
    case SimulationMode::WhiteNoise:              fname << "_wn"; break;
    case SimulationMode::ConstWhiteNoise:         fname << "_cwn"; break;
    case SimulationMode::ConstWhiteNoiseOpen:     fname << "_cwn_open"; break;
    case SimulationMode::ConstWhiteNoisePeriodic: fname << "_cwn_periodic"; break;
    case SimulationMode::CM_PS:
        fname << "_cmps_psrate_" << ps_rate;
        break;
    case SimulationMode::ThreeD_Deterministic:    fname << "_3d_det"; break;
    case SimulationMode::ThreeD_Poisson:          fname << "_3d_poi"; break;
    default:
        break;
    }

    fname << "_mu_a_" << p.mu_a << ".gz";

    std::cout << "Running simulation: " << fname.str() << std::endl;

    // -------------------------------------------------------------------------
    // Run simulation
    // -------------------------------------------------------------------------
    auto start = std::chrono::high_resolution_clock::now();
    Results R;
    ParamsCMPS pcmps;
    switch (p.mode) {
    case SimulationMode::Deterministic:
        std::cout << "Mode: deterministic (Nmotor→∞)\n";
        {
            ParamsDeterministic pd = {p.n, p.fstar, p.mu, p.eta, p.zeta, p.beta, p.t_sub, p.T, p.dt, p.ZETA_GAMMA_THRESHOLD, p.seed, p.mu_a};
            R = simulate_episode_deterministic(pd);
        }
        break;

    case SimulationMode::Poisson:
        std::cout << "Mode: Poisson\n";
        {
            ParamsPoisson pp = {p.n, p.fstar, p.mu, p.eta, p.zeta, p.beta, p.t_sub, p.T, p.dt, p.ZETA_GAMMA_THRESHOLD, p.seed, p.Nmotor, p.LAMBDA_SMALL_THRESHOLD, p.mu_a};
            R = simulate_episode(pp);
        }
        break;

    case SimulationMode::WhiteNoise:
        {
            ParamsWhiteNoise pw = {p.n, p.fstar, p.mu, p.eta, p.zeta, p.beta, p.t_sub, p.T, p.dt, p.ZETA_GAMMA_THRESHOLD, p.seed, p.Nmotor, p.mu_a};
            R = simulate_episode_white_noise(pw);
        }
        break;

    case SimulationMode::ConstWhiteNoise:
        {
            ParamsConstWhiteNoise pc = {p.n, p.fstar, p.mu, p.eta, p.zeta, p.beta, p.t_sub, p.T, p.dt, p.ZETA_GAMMA_THRESHOLD, p.seed, p.Nmotor, p.mu_a};
            R = simulate_episode_const_white_noise(pc);
        }
        break;

    case SimulationMode::ConstWhiteNoiseOpen:
        {
            ParamsConstWhiteNoiseOpen po = {p.n, p.fstar, p.mu, p.eta, p.zeta, p.beta, p.t_sub, p.T, p.dt, p.ZETA_GAMMA_THRESHOLD, p.seed, p.Nmotor, p.mu_a};
            R = simulate_episode_const_white_noise_open(po);
        }
        break;

    case SimulationMode::ConstWhiteNoisePeriodic:
        {
            ParamsConstWhiteNoisePeriodic pp = {p.n, p.fstar, p.mu, p.eta, p.zeta, p.beta, p.t_sub, p.T, p.dt, p.ZETA_GAMMA_THRESHOLD, p.seed, p.Nmotor, p.mu_a};
            R = simulate_episode_const_white_noise_periodic(pp);
        }
        break;

    case SimulationMode::CM_PS: {
        
        ParamsCMPS pcmps = {p.n, p.fstar, p.mu, p.eta, p.zeta, p.beta, p.t_sub, p.T, p.dt, p.ZETA_GAMMA_THRESHOLD, p.seed, p.Nmotor, ps_rate, p.LAMBDA_SMALL_THRESHOLD, p.mu_a, 0.0};

        std::cout << "Mode: CM+PS (ps_rate=" << ps_rate << ")\n";
        R = simulate_episode_cm_ps(pcmps);
        break;
    }
    case SimulationMode::ThreeD_Deterministic:
        std::cout << "Mode: 3D deterministic\n";
        {
            Params3DDet p3d = {p.n, p.fstar, p.mu, p.eta, p.zeta, p.beta, p.t_sub, p.T, p.dt, p.ZETA_GAMMA_THRESHOLD, p.seed, p.d_tilde_1, p.d_tilde_2, p.d_tilde_3, p.d_tilde_4, p.mu_a};
            R = simulate_episode_3d_deterministic(p3d);
        }
        break;

    case SimulationMode::ThreeD_Poisson:
        std::cout << "Mode: 3D Poisson\n";
        {
            Params3D p3 = {p.n, p.fstar, p.mu, p.eta, p.zeta, p.beta, p.t_sub, p.T, p.dt, p.ZETA_GAMMA_THRESHOLD, p.seed, p.Nmotor, p.d_tilde_1, p.d_tilde_2, p.d_tilde_3, p.d_tilde_4, p.LAMBDA_SMALL_THRESHOLD, p.mu_a};
            R = simulate_episode_3d(p3);
        }
        break;
    }

    auto end = std::chrono::high_resolution_clock::now();
    double seconds = std::chrono::duration<double>(end - start).count();

    std::cout << "Done in " << seconds << " s\n";
    std::cout << "  Saturations: " << R.count_large_zeta_gamma
              << "  Regular Poisson: " << R.count_regular_poisson
              << "  Poisson Overflow: " << R.count_poisson_overflow
              << "  Invalid Denom: " << R.count_invalid_denom << "\n";

    try {
        save_compressed(fname.str(), p, R, seconds);

        std::cout << "Saved compressed results to " << fname.str() << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Error saving results: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
