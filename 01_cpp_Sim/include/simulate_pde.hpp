#pragma once
#include <vector>
#include <string>
#include <cstddef>
#include <cstdint>

namespace spde {

enum class SimulationMode {
    Deterministic,
    Poisson,          
    WhiteNoise,       
    ConstWhiteNoise,         
    ConstWhiteNoiseOpen,      
    ConstWhiteNoisePeriodic,
    CM_PS,
    ThreeD_Deterministic,
    ThreeD_Poisson                
};


// Common parameters used by all modes
struct ParamsCommon {
    int    n;        // number of spatial intervals; array size is n+1
    double fstar;
    double mu;
    double eta;
    double zeta;
    double beta;
    double t_sub;    // coarse sampling per second (Hz)
    double T;        // total time
    double dt;       // fine time step
    double ZETA_GAMMA_THRESHOLD;
    unsigned long long seed;
};

// Deterministic mode (no Nmotor, no stochastic)
struct ParamsDeterministic {
    int n;
    double fstar, mu, eta, zeta, beta, t_sub, T, dt;
    double ZETA_GAMMA_THRESHOLD;
    unsigned long long seed;
    double mu_a;
};

// Poisson mode (finite Nmotor, stochastic binding)
struct ParamsPoisson {
    int n;
    double fstar, mu, eta, zeta, beta, t_sub, T, dt;
    double ZETA_GAMMA_THRESHOLD;
    unsigned long long seed;
    double Nmotor;
    double LAMBDA_SMALL_THRESHOLD;
    double mu_a;
};

// White noise approximation
struct ParamsWhiteNoise {
    int n;
    double fstar, mu, eta, zeta, beta, t_sub, T, dt;
    double ZETA_GAMMA_THRESHOLD;
    unsigned long long seed;
    double Nmotor;
    double mu_a;
};

// Constant white noise
struct ParamsConstWhiteNoise {
    int n;
    double fstar, mu, eta, zeta, beta, t_sub, T, dt;
    double ZETA_GAMMA_THRESHOLD;
    unsigned long long seed;
    double Nmotor;
    double mu_a;
};

// Constant white noise open
struct ParamsConstWhiteNoiseOpen {
    int n;
    double fstar, mu, eta, zeta, beta, t_sub, T, dt;
    double ZETA_GAMMA_THRESHOLD;
    unsigned long long seed;
    double Nmotor;
    double mu_a;
};

// Constant white noise periodic
struct ParamsConstWhiteNoisePeriodic {
    int n;
    double fstar, mu, eta, zeta, beta, t_sub, T, dt;
    double ZETA_GAMMA_THRESHOLD;
    unsigned long long seed;
    double Nmotor;
    double mu_a;
};

// CM+PS model
struct ParamsCMPS {
    int n;
    double fstar, mu, eta, zeta, beta, t_sub, T, dt;
    double ZETA_GAMMA_THRESHOLD;
    unsigned long long seed;
    double Nmotor;
    double ps_rate;
    double LAMBDA_SMALL_THRESHOLD;
    double mu_a;
    double rest_rate;
    SimulationMode mode;
};

// 3D deterministic
struct Params3DDet {
    int n;
    double fstar, mu, eta, zeta, beta, t_sub, T, dt;
    double ZETA_GAMMA_THRESHOLD;
    unsigned long long seed;
    double d_tilde_1;
    double d_tilde_2;
    double d_tilde_3;
    double d_tilde_4;
    double mu_a;
};

// 3D Poisson
struct Params3D {
    int n;
    double fstar, mu, eta, zeta, beta, t_sub, T, dt;
    double ZETA_GAMMA_THRESHOLD;
    unsigned long long seed;
    double Nmotor;
    double d_tilde_1;
    double d_tilde_2;
    double d_tilde_3;
    double d_tilde_4;
    double LAMBDA_SMALL_THRESHOLD;
    double mu_a;
};

// For backward compatibility and main.cpp
struct Params {
    int n;
    double fstar, mu, eta, zeta, beta, t_sub, T, dt;
    double ZETA_GAMMA_THRESHOLD;
    unsigned long long seed;
    double Nmotor;
    double LAMBDA_SMALL_THRESHOLD;
    double ps_rate;
    double d_tilde_1, d_tilde_2, d_tilde_3, d_tilde_4;
    SimulationMode mode;
    double mu_a;
};

struct Results {
    std::size_t n_coarse{};
    std::size_t n_nodes{};

    std::vector<double> t_coarse;
    std::vector<double> gamma_mat;
    std::vector<double> nplus_mat;
    std::vector<double> nminus_mat;

    // Diagnostics
    std::uint64_t count_large_zeta_gamma{};
    std::uint64_t count_regular_poisson{};
    std::uint64_t count_poisson_overflow{};
    std::uint64_t count_invalid_denom{};

    static inline std::size_t idx(std::size_t row, std::size_t col, std::size_t ncols) noexcept {
        return row * ncols + col;
    }
};


// Initialization
void initialize_fields(int n, double eta, double fstar, double zeta, double Nmotor,
                       std::vector<double>& s,
                       std::vector<double>& gamma0,
                       std::vector<double>& nplus0,
                       std::vector<double>& nminus0);

// Slightly randomized initialization for deterministic mode
void initialize_fields_deterministic(int n, double eta, double fstar, double zeta, unsigned long long seed,
                                     std::vector<double>& s,
                                     std::vector<double>& gamma0,
                                     std::vector<double>& nplus0,
                                     std::vector<double>& nminus0);


// Core solvers
Results simulate_episode(const ParamsPoisson& p);                // Stochastic (finite Nmotor)
Results simulate_episode_deterministic(const ParamsDeterministic& p);  // Deterministic limit (Nmotor→∞)
Results simulate_episode_white_noise(const ParamsWhiteNoise& p);    // wn approximation
Results simulate_episode_const_white_noise(const ParamsConstWhiteNoise& p);
Results simulate_episode_const_white_noise_open(const ParamsConstWhiteNoiseOpen& p);
Results simulate_episode_const_white_noise_periodic(const ParamsConstWhiteNoisePeriodic& p);
Results simulate_episode_cm_ps(const ParamsCMPS& p);  // Cass + Power Stroke model
Results simulate_episode_3d_deterministic(const Params3DDet& p);
Results simulate_episode_3d(const Params3D& p);


} // namespace spde
