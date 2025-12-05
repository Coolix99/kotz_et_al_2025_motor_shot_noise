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
    ConstWhiteNoisePeriodic  
};


// -----------------------------------------------------------------------------
// Parameter + result structs
// -----------------------------------------------------------------------------
struct Params {
    int    n;        // number of spatial intervals; array size is n+1
    double fstar;
    double mu;
    double eta;
    double zeta;
    double beta;
    double t_sub;    // coarse sampling 
    double T;        // total time
    double dt;       // fine time step
    double mu_a;
    double Nmotor;   // number of motors (negative = deterministic limit)
    SimulationMode mode;
    unsigned long long seed;
    double ZETA_GAMMA_THRESHOLD;
    double LAMBDA_SMALL_THRESHOLD;
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

    static inline std::size_t idx(std::size_t row, std::size_t col, std::size_t ncols) noexcept {
        return row * ncols + col;
    }
};

// -----------------------------------------------------------------------------
// Initialization
// -----------------------------------------------------------------------------
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

// -----------------------------------------------------------------------------
// Core solvers
// -----------------------------------------------------------------------------
Results simulate_episode(const Params& p);                // Stochastic (finite Nmotor)
Results simulate_episode_deterministic(const Params& p);  // Deterministic limit (Nmotor→∞)
Results simulate_episode_white_noise(const Params& p);    // wn approximation
Results simulate_episode_const_white_noise(const Params& p);
Results simulate_episode_const_white_noise_open(const Params& p);
Results simulate_episode_const_white_noise_periodic(const Params& p);

} // namespace spde
