#include "simulate_pde.hpp"

#include <cmath>
#include <algorithm>
#include <random>
#include <fstream>
#include <stdexcept>
#include <limits>
#include <iostream>
#include <cstdint>
#include <array> 

namespace spde {

static inline void compute_constant_white_noise_amplitudes(
        double eta, double fstar, double Nmotor,
        double &D_const){   
    const double exp_eq  = std::exp(fstar);
    const double b  = eta + (1.0 - eta) * exp_eq;
    const double n0 = eta / b;

    const double bind_plus_eq    = eta * (1.0 - n0);
    const double bind_minus_eq   = eta * (1.0 - n0);
    const double unbind_plus_eq  = (1 - eta) * n0 * exp_eq;
    const double unbind_minus_eq = (1 - eta) * n0 * exp_eq;

    D_const = (bind_plus_eq  + unbind_plus_eq)  / Nmotor;
}

static inline bool is_finite(double x) noexcept {
    return std::isfinite(x);
}

// Initialization
void initialize_fields(int n, double eta, double fstar, double /*zeta*/, double Nmotor,
                       std::vector<double>& s,
                       std::vector<double>& gamma0,
                       std::vector<double>& nplus0,
                       std::vector<double>& nminus0)
{
    const std::size_t N = static_cast<std::size_t>(n) + 1;
    s.resize(N);
    gamma0.resize(N);
    nplus0.resize(N);
    nminus0.resize(N);

    const double inv_n = 1.0 / static_cast<double>(n);
    for (int i = 0; i <= n; ++i)
        s[static_cast<std::size_t>(i)] = i * inv_n;

    for (std::size_t i = 0; i < N; ++i) {
        const double si = s[i] - 0.5;
        gamma0[i] = 0.0015 * std::exp(-(si * si) / (0.1 * 0.1));
    }

    const double b  = eta + (1.0 - eta) * std::exp(fstar);
    double n0 = eta / b;
    if (Nmotor > 0.0)
        n0 = std::round(n0 * Nmotor) / Nmotor;

    std::fill(nplus0.begin(),  nplus0.end(),  n0);
    std::fill(nminus0.begin(), nminus0.end(), n0);
}

void initialize_fields(int n, double eta, double fstar, double /*zeta*/, double Nmotor,
                       std::vector<double>& s,
                       std::vector<double>& gamma0,
                       std::vector<double>& nplus0_rest,
                       std::vector<double>& nminus0_rest,
                       std::vector<double>& nplus0_ps,
                       std::vector<double>& nminus0_ps)

{
    const std::size_t N = static_cast<std::size_t>(n) + 1;
    s.resize(N);
    gamma0.resize(N);
    nplus0_rest.resize(N);
    nminus0_rest.resize(N);
    nplus0_ps.resize(N);
    nminus0_ps.resize(N);

    const double inv_n = 1.0 / static_cast<double>(n);
    for (int i = 0; i <= n; ++i)
        s[static_cast<std::size_t>(i)] = i * inv_n;

    for (std::size_t i = 0; i < N; ++i) {
        const double si = s[i] - 0.5;
        gamma0[i] = 0.0015 * std::exp(-(si * si) / (0.1 * 0.1));
    }

    const double b  = eta + (1.0 - eta) * std::exp(fstar);
    double n0 = eta / b;
    if (Nmotor > 0.0)
        n0 = std::round(n0 * Nmotor) / Nmotor;

    std::fill(nplus0_rest.begin(),  nplus0_rest.end(),  n0);
    std::fill(nminus0_rest.begin(), nminus0_rest.end(), n0);
    std::fill(nplus0_ps.begin(),  nplus0_ps.end(),  n0);
    std::fill(nminus0_ps.begin(), nminus0_ps.end(), n0);
}


void initialize_fields_deterministic(
    int n, double eta, double fstar, double /*zeta*/, unsigned long long seed,
    std::vector<double>& s,
    std::vector<double>& gamma0,
    std::vector<double>& nplus0,
    std::vector<double>& nminus0)
{
    const std::size_t N = static_cast<std::size_t>(n) + 1;
    s.resize(N);
    gamma0.resize(N);
    nplus0.resize(N);
    nminus0.resize(N);

    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> jitter(-0.1, 0.1);  // small random offset

    const double inv_n = 1.0 / static_cast<double>(n);
    for (int i = 0; i <= n; ++i)
        s[static_cast<std::size_t>(i)] = i * inv_n;

    for (std::size_t i = 0; i < N; ++i) {
        // Add small random horizontal offset in Gaussian center
        const double shift = 0.5 + 0.1 * jitter(rng);
        const double si = s[i] - shift;
        gamma0[i] = 0.0015 * std::exp(-(si * si) / (0.1 * 0.1));
    }

    const double b  = eta + (1.0 - eta) * std::exp(fstar);
    const double n0 = eta / b;

    std::fill(nplus0.begin(),  nplus0.end(),  n0);
    std::fill(nminus0.begin(), nminus0.end(), n0);
}



// Simulation
Results simulate_episode(const ParamsPoisson& p)
{
    if (p.n < 1) throw std::invalid_argument("n must be >= 1");
    if (p.dt <= 0.0) throw std::invalid_argument("dt must be > 0");
    if (p.t_sub <= 0.0) throw std::invalid_argument("t_sub must be > 0");
    if (p.T < 0.0) throw std::invalid_argument("T must be >= 0");

    const int n = p.n;
    const std::size_t N = static_cast<std::size_t>(n) + 1;
    const double n2 = static_cast<double>(n) * static_cast<double>(n);

    // Precompute constants
    const double mu_a_zeta = p.mu_a * p.zeta;
    const double Nm_dt = p.Nmotor * p.dt;
    const double one_minus_eta = 1.0 - p.eta;

    // Initialize fields
    std::vector<double> s, gamma, nplus, nminus;
    initialize_fields(n, p.eta, p.fstar, p.zeta, p.Nmotor, s, gamma, nplus, nminus);

    // Prepare result container
    Results R;
    const auto llr = [](double x){ return static_cast<long long>(std::llround(x)); };
    R.n_coarse = static_cast<std::size_t>(llr(p.T * p.t_sub)) + 1;
    R.n_nodes  = N;
    R.t_coarse.resize(R.n_coarse);
    for (std::size_t i = 0; i < R.n_coarse; ++i)
        R.t_coarse[i] = (R.n_coarse == 1) ? 0.0 : (static_cast<double>(i) / (R.n_coarse - 1)) * p.T;

    R.gamma_mat.resize(R.n_coarse * N);
    R.nplus_mat.resize(R.n_coarse * N);
    R.nminus_mat.resize(R.n_coarse * N);

    // Buffers
    std::vector<double> gamma_t(N, 0.0);
    std::vector<double> lap(N, 0.0);

    // RNG
    std::mt19937_64 rng(p.seed);
    std::uniform_real_distribution<double> uni(0.0, 1.0);

    // Thresholds
    const double ZETA_GAMMA_THRESHOLD = p.ZETA_GAMMA_THRESHOLD;
    const double LAMBDA_SMALL_THRESHOLD = p.LAMBDA_SMALL_THRESHOLD;

    auto sample_poisson = [&](double lambda) -> std::uint64_t {
        if (lambda <= 0.0) return 0;
        if (lambda < LAMBDA_SMALL_THRESHOLD)
            return (uni(rng) < lambda) ? 1U : 0U;

        if (lambda > 1e12) {
            throw std::runtime_error("sample_poisson(): lambda exceeds 1e12 — possible numerical instability");
        }

        ++R.count_regular_poisson;
        std::poisson_distribution<std::uint64_t> dist(lambda);
        return dist(rng);
    };

    // Time setup
    const std::size_t nt = static_cast<std::size_t>(llr(p.T / p.dt)) + 1;
    const std::size_t t_sub_refine = std::max<std::size_t>(1, llr(1.0 / (p.dt * p.t_sub)));

    std::size_t coarse_idx = 0;

    // Main simulation loop
    for (std::size_t it = 0; it < nt; ++it) {

        // Laplacian (inner points)
        for (int i = 1; i < n; ++i)
            lap[i] = (gamma[i - 1] - 2.0 * gamma[i] + gamma[i + 1]) * n2;

        // gamma_t (inner + boundary)
        for (int i = 1; i < n; ++i) {
            const double denom = p.beta + mu_a_zeta * (nplus[i] + nminus[i]);
            if (denom < 1.0 || !std::isfinite(denom)) {
                std::cerr << "Small/invalid denom at i=" << i
                        << " denom=" << denom
                        << " beta=" << p.beta
                        << " mu_a_zeta=" << mu_a_zeta
                        << " n+ n-=" << nplus[i] << " " << nminus[i]
                        << " gamma=" << gamma[i] << "\n";
            }

            gamma_t[i] = (lap[i] - p.mu * gamma[i] + p.mu_a * (nminus[i] - nplus[i])) / denom;
        }
        gamma_t[0] = -gamma[0] / p.dt;
        {
            const std::size_t kn = static_cast<std::size_t>(n);
            const double denom_n = p.beta + mu_a_zeta * (nplus[kn] + nminus[kn]);
            if (denom_n < 1.0 || !std::isfinite(denom_n)) {
                std::cerr << "Invalid denom_n=" << denom_n << "\n";
            }

            const double lap_last = (2.0 * gamma[n - 1] - 2.0 * gamma[kn]) * n2;
            gamma_t[kn] = (lap_last - p.mu * gamma[kn] + p.mu_a * (nminus[kn] - nplus[kn])) / denom_n;
        }

        // Stochastic motor updates
        for (std::size_t i = 0; i < N; ++i) {
            gamma[i]  += gamma_t[i] * p.dt;
            
            const double zeta_gamma = p.zeta * gamma_t[i];
            const bool saturated = std::fabs(zeta_gamma) > ZETA_GAMMA_THRESHOLD;

            double exp_plus  = 1.0, exp_minus = 1.0;
            if (!saturated) {
                exp_plus  = std::exp(p.fstar * (1.0 + zeta_gamma));
                exp_minus = std::exp(p.fstar * (1.0 - zeta_gamma));
            } else {
                ++R.count_large_zeta_gamma;
            }

            double bind_plus   = p.eta * (1.0 - nplus[i])  * Nm_dt;
            double bind_minus  = p.eta * (1.0 - nminus[i]) * Nm_dt;
            double unbind_plus  = one_minus_eta * nplus[i]  * exp_plus  * Nm_dt;
            double unbind_minus = one_minus_eta * nminus[i] * exp_minus * Nm_dt;

            if (saturated) {
                // Early exit: only sample one side depending on zeta_gamma sign
                unbind_plus = unbind_minus = 0.0;

                if (zeta_gamma > 0.0) {
                    nplus[i] = 0.0;
                    bind_plus = 0.0;
                    const double dNm = static_cast<double>(sample_poisson(bind_minus));
                    nminus[i] = std::clamp(nminus[i] + dNm / p.Nmotor, 0.0, 1.0);
                } else {
                    nminus[i] = 0.0;
                    bind_minus = 0.0;
                    const double dNp = static_cast<double>(sample_poisson(bind_plus));
                    nplus[i] = std::clamp(nplus[i] + dNp / p.Nmotor, 0.0, 1.0);
                }
                continue;  // skip rest for saturated case
            }

            const double dNp = static_cast<double>(sample_poisson(bind_plus))
                            - static_cast<double>(sample_poisson(unbind_plus));
            const double dNm = static_cast<double>(sample_poisson(bind_minus))
                            - static_cast<double>(sample_poisson(unbind_minus));

            
            nplus[i]  = std::clamp(nplus[i] + dNp / p.Nmotor,  0.0, 1.0);
            nminus[i] = std::clamp(nminus[i] + dNm / p.Nmotor, 0.0, 1.0);
        }


        // Record coarse data
        if (it % t_sub_refine == 0 && coarse_idx < R.n_coarse) {
            const std::size_t base = coarse_idx * N;
            std::copy(gamma.begin(),  gamma.end(),  R.gamma_mat.begin()  + base);
            std::copy(nplus.begin(),  nplus.end(),  R.nplus_mat.begin()  + base);
            std::copy(nminus.begin(), nminus.end(), R.nminus_mat.begin() + base);
            ++coarse_idx;
        }
    }

    return R;
}

Results simulate_episode_deterministic(const ParamsDeterministic& p)
{
    if (p.n < 1) throw std::invalid_argument("n must be >= 1");
    if (p.dt <= 0.0) throw std::invalid_argument("dt must be > 0");
    if (p.t_sub <= 0.0) throw std::invalid_argument("t_sub must be > 0");
    if (p.T < 0.0) throw std::invalid_argument("T must be >= 0");

    const int n = p.n;
    const std::size_t N = static_cast<std::size_t>(n) + 1;
    const double n2 = static_cast<double>(n) * static_cast<double>(n);

    // Precompute constants
    const double mu_a_zeta = p.mu_a * p.zeta;
    const double one_minus_eta = 1.0 - p.eta;

    // Initialize fields with small random perturbation
    std::vector<double> s, gamma, nplus, nminus;
    initialize_fields_deterministic(n, p.eta, p.fstar, p.zeta, p.seed, s, gamma, nplus, nminus);

    // Prepare result container
    Results R;
    const auto llr = [](double x){ return static_cast<long long>(std::llround(x)); };
    R.n_coarse = static_cast<std::size_t>(llr(p.T * p.t_sub)) + 1;
    R.n_nodes  = N;
    R.t_coarse.resize(R.n_coarse);
    for (std::size_t i = 0; i < R.n_coarse; ++i)
        R.t_coarse[i] = (R.n_coarse == 1) ? 0.0 : (static_cast<double>(i) / (R.n_coarse - 1)) * p.T;

    R.gamma_mat.resize(R.n_coarse * N);
    R.nplus_mat.resize(R.n_coarse * N);
    R.nminus_mat.resize(R.n_coarse * N);

    // Buffers
    std::vector<double> gamma_t(N, 0.0);
    std::vector<double> lap(N, 0.0);

    const double ZETA_GAMMA_THRESHOLD = p.ZETA_GAMMA_THRESHOLD;

    // Time setup
    const std::size_t nt = static_cast<std::size_t>(llr(p.T / p.dt)) + 1;
    const std::size_t t_sub_refine = std::max<std::size_t>(1, llr(1.0 / (p.dt * p.t_sub)));

    std::size_t coarse_idx = 0;

    // Deterministic loop (no stochastic binding)
    for (std::size_t it = 0; it < nt; ++it) {

        // Laplacian
        for (int i = 1; i < n; ++i)
            lap[i] = (gamma[i - 1] - 2.0 * gamma[i] + gamma[i + 1]) * n2;

        // gamma_t (inner + boundary)
        for (int i = 1; i < n; ++i) {
            const double denom = p.beta + mu_a_zeta * (nplus[i] + nminus[i]);
            gamma_t[i] = (lap[i] - p.mu * gamma[i] + p.mu_a * (nminus[i] - nplus[i])) / denom;
        }
        gamma_t[0] = -gamma[0] / p.dt;
        {
            const std::size_t kn = static_cast<std::size_t>(n);
            const double denom_n = p.beta + mu_a_zeta * (nplus[kn] + nminus[kn]);
            const double lap_last = (2.0 * gamma[n - 1] - 2.0 * gamma[kn]) * n2;
            gamma_t[kn] = (lap_last - p.mu * gamma[kn] + p.mu_a * (nminus[kn] - nplus[kn])) / denom_n;
        }

        // Deterministic update (mean-field ODEs)
        for (std::size_t i = 0; i < N; ++i) {
            const double zeta_gamma = p.zeta * gamma_t[i];
            const bool saturated = std::fabs(zeta_gamma) > ZETA_GAMMA_THRESHOLD;

            double exp_plus  = 1.0, exp_minus = 1.0;
            if (!saturated) {
                exp_plus  = std::exp(p.fstar * (1.0 + zeta_gamma));
                exp_minus = std::exp(p.fstar * (1.0 - zeta_gamma));
            } else {
                ++R.count_large_zeta_gamma;
            }

            const double bind_plus   = p.eta * (1.0 - nplus[i]);
            const double bind_minus  = p.eta * (1.0 - nminus[i]);
            const double unbind_plus  = one_minus_eta * nplus[i]  * exp_plus;
            const double unbind_minus = one_minus_eta * nminus[i] * exp_minus;

            // ODE evolution (no stochastic rounding)
            const double dNp = bind_plus - unbind_plus;
            const double dNm = bind_minus - unbind_minus;

            gamma[i]  += gamma_t[i] * p.dt;
            nplus[i]  = std::clamp(nplus[i] + p.dt * dNp,  0.0, 1.0);
            nminus[i] = std::clamp(nminus[i] + p.dt * dNm, 0.0, 1.0);
        }

        // Record coarse data
        if (it % t_sub_refine == 0 && coarse_idx < R.n_coarse) {
            const std::size_t base = coarse_idx * N;
            std::copy(gamma.begin(),  gamma.end(),  R.gamma_mat.begin()  + base);
            std::copy(nplus.begin(),  nplus.end(),  R.nplus_mat.begin()  + base);
            std::copy(nminus.begin(), nminus.end(), R.nminus_mat.begin() + base);
            ++coarse_idx;
        }
    }

    return R;
}

Results simulate_episode_white_noise(const ParamsWhiteNoise &p) {
    if (p.n < 1) throw std::invalid_argument("n must be >= 1");
    if (p.dt <= 0.0) throw std::invalid_argument("dt must be > 0");
    if (p.t_sub <= 0.0) throw std::invalid_argument("t_sub must be > 0");
    if (p.T < 0.0) throw std::invalid_argument("T must be >= 0");

    const int n = p.n;
    const std::size_t N = static_cast<std::size_t>(n) + 1;
    const double n2 = static_cast<double>(n) * static_cast<double>(n);

    // Precompute constants
    const double mu_a_zeta = p.mu_a * p.zeta;
    const double one_minus_eta = 1.0 - p.eta;

    // Initialize fields
    std::vector<double> s, gamma, nplus, nminus;
    initialize_fields(n, p.eta, p.fstar, p.zeta, p.Nmotor, s, gamma, nplus, nminus);

    // Prepare result container
    Results R;
    const auto llr = [](double x){ return static_cast<long long>(std::llround(x)); };
    R.n_coarse = static_cast<std::size_t>(llr(p.T * p.t_sub)) + 1;
    R.n_nodes  = N;
    R.t_coarse.resize(R.n_coarse);
    for (std::size_t i = 0; i < R.n_coarse; ++i)
        R.t_coarse[i] = (R.n_coarse == 1) ? 0.0 : (static_cast<double>(i) / (R.n_coarse - 1)) * p.T;

    R.gamma_mat.resize(R.n_coarse * N);
    R.nplus_mat.resize(R.n_coarse * N);
    R.nminus_mat.resize(R.n_coarse * N);

    // Buffers
    std::vector<double> gamma_t(N, 0.0);
    std::vector<double> lap(N, 0.0);

    // RNG for Gaussian white noise
    std::mt19937_64 rng(p.seed);
    std::normal_distribution<double> normal(0.0, 1.0);

    const double ZETA_GAMMA_THRESHOLD = p.ZETA_GAMMA_THRESHOLD;

    // Time setup
    const std::size_t nt = static_cast<std::size_t>(llr(p.T / p.dt)) + 1;
    const std::size_t t_sub_refine = std::max<std::size_t>(1, llr(1.0 / (p.dt * p.t_sub)));

    std::size_t coarse_idx = 0;

    // Main simulation loop
    for (std::size_t it = 0; it < nt; ++it) {

        // Laplacian (inner points)
        for (int i = 1; i < n; ++i)
            lap[i] = (gamma[i - 1] - 2.0 * gamma[i] + gamma[i + 1]) * n2;

        // gamma_t (inner + boundary)
        for (int i = 1; i < n; ++i) {
            const double denom = p.beta + mu_a_zeta * (nplus[i] + nminus[i]);
            gamma_t[i] = (lap[i] - p.mu * gamma[i] + p.mu_a * (nminus[i] - nplus[i])) / denom;
        }
        gamma_t[0] = -gamma[0] / p.dt;
        {
            const std::size_t kn = static_cast<std::size_t>(n);
            const double denom_n = p.beta + mu_a_zeta * (nplus[kn] + nminus[kn]);
            if (denom_n < 1.0 || !std::isfinite(denom_n)) {
                std::cerr << "Invalid denom_n=" << denom_n << "\n";
            }

            const double lap_last = (2.0 * gamma[n - 1] - 2.0 * gamma[kn]) * n2;
            gamma_t[kn] = (lap_last - p.mu * gamma[kn] + p.mu_a * (nminus[kn] - nplus[kn])) / denom_n;
        }

        // White-noise approximation for motor dynamics
        for (std::size_t i = 0; i < N; ++i) {
            gamma[i] += gamma_t[i] * p.dt;

            const double zeta_gamma = p.zeta * gamma_t[i];
            const bool saturated =
                std::fabs(zeta_gamma) > ZETA_GAMMA_THRESHOLD;

            double exp_plus = 1.0, exp_minus = 1.0;

            if (!saturated) {
                exp_plus  = std::exp(p.fstar * (1.0 + zeta_gamma));
                exp_minus = std::exp(p.fstar * (1.0 - zeta_gamma));
            } else {
                ++R.count_large_zeta_gamma;
            }

            // --- Drift terms (deterministic reaction part) ---
            const double bind_plus   = p.eta * (1.0 - nplus[i]);
            const double bind_minus  = p.eta * (1.0 - nminus[i]);
            const double unbind_plus  = one_minus_eta * nplus[i]  * exp_plus;
            const double unbind_minus = one_minus_eta * nminus[i] * exp_minus;

            const double drift_plus = bind_plus - unbind_plus;
            const double drift_minus = bind_minus - unbind_minus;

            // --- Diffusion (white noise) ---
            const double Dp =
                ( bind_plus
                + unbind_plus) / p.Nmotor;

            const double Dm =
                ( bind_minus
                + unbind_minus) / p.Nmotor;

            const double noise_plus  = std::sqrt(Dp * p.dt) * normal(rng);
            const double noise_minus = std::sqrt(Dm * p.dt) * normal(rng);

            // Update with drift + noise
            nplus[i]  = std::clamp(nplus[i]  + p.dt*drift_plus  + noise_plus,  0.0, 1.0);
            nminus[i] = std::clamp(nminus[i] + p.dt*drift_minus + noise_minus, 0.0, 1.0);
        }

        // Record coarse data
        if (it % t_sub_refine == 0 && coarse_idx < R.n_coarse) {
            const std::size_t base = coarse_idx * N;
            std::copy(gamma.begin(),  gamma.end(),  R.gamma_mat.begin() + base);
            std::copy(nplus.begin(),  nplus.end(),  R.nplus_mat.begin() + base);
            std::copy(nminus.begin(), nminus.end(), R.nminus_mat.begin() + base);
            ++coarse_idx;
        }
    }

    return R;
}

Results simulate_episode_const_white_noise(const ParamsConstWhiteNoise &p){
    if (p.n < 1) throw std::invalid_argument("n must be >=1");
    if (p.dt <= 0.0) throw std::invalid_argument("dt must be >0");

    const int n = p.n;
    const std::size_t N = n + 1;
    const double n2 = double(n) * double(n);

    std::vector<double> s, gamma, nplus, nminus;
    initialize_fields(n, p.eta, p.fstar, p.zeta, p.Nmotor, s, gamma, nplus, nminus);

    double D_const;
    compute_constant_white_noise_amplitudes(p.eta, p.fstar, p.Nmotor,
                                            D_const);

    Results R;
    const auto llr = [](double x){ return (long long)std::llround(x); };
    R.n_coarse = llr(p.T * p.t_sub) + 1;
    R.n_nodes = N;
    R.t_coarse.resize(R.n_coarse);
    for (std::size_t i=0;i<R.n_coarse;i++)
        R.t_coarse[i] = (R.n_coarse==1)?0.0 : (double(i) / (R.n_coarse-1)) * p.T;

    R.gamma_mat.resize(R.n_coarse * N);
    R.nplus_mat.resize(R.n_coarse * N);
    R.nminus_mat.resize(R.n_coarse * N);

    std::vector<double> lap(N), gamma_t(N);
    std::mt19937_64 rng(p.seed);
    std::normal_distribution<double> normal(0.0, 1.0);

    const double mu_a_zeta = p.mu_a * p.zeta;
    const double one_minus_eta = 1 - p.eta;

    const std::size_t nt = llr(p.T / p.dt) + 1;
    const std::size_t t_sub_refine = std::max<std::size_t>(1, llr(1.0/(p.dt*p.t_sub)));

    std::size_t coarse_idx = 0;

    for (std::size_t it=0; it<nt; ++it) {

        // Laplacian (same as white-noise)
        for (int i=1;i<n;++i)
            lap[i] = (gamma[i-1] - 2*gamma[i] + gamma[i+1]) * n2;

        // gamma dynamics (same as white-noise)
        for (int i=1;i<n;++i) {
            const double denom = p.beta + mu_a_zeta*(nplus[i] + nminus[i]);
            gamma_t[i] = (lap[i] - p.mu*gamma[i] + p.mu_a*(nminus[i]-nplus[i])) / denom;
        }
        gamma_t[0] = -gamma[0] / p.dt;
        {
            const int i = n;
            const double denom = p.beta + mu_a_zeta*(nplus[i] + nminus[i]);
            const double lap_last = (2*gamma[n-1] - 2*gamma[n]) * n2;
            gamma_t[i] = (lap_last - p.mu*gamma[i] + p.mu_a*(nminus[i]-nplus[i])) / denom;
        }

        // Motor dynamics with *constant* noise
        for (std::size_t i=0;i<N;++i) {

            gamma[i] += gamma_t[i] * p.dt;

            const double zeta_gamma = p.zeta * gamma_t[i];
            const bool saturated = std::fabs(zeta_gamma) > p.ZETA_GAMMA_THRESHOLD;

            double exp_plus = 1.0, exp_minus = 1.0;
            if (!saturated) {
                exp_plus  = std::exp(p.fstar*(1+zeta_gamma));
                exp_minus = std::exp(p.fstar*(1-zeta_gamma));
            } else {
                ++R.count_large_zeta_gamma;
            }

            // Drift
            const double bind_plus   = p.eta*(1-nplus[i]);
            const double bind_minus  = p.eta*(1-nminus[i]);
            const double unbind_plus = one_minus_eta*nplus[i]*exp_plus;
            const double unbind_minus = one_minus_eta*nminus[i]*exp_minus;

            const double dNp = bind_plus - unbind_plus;
            const double dNm = bind_minus - unbind_minus;

            // Constant white noise amplitude
            const double noise_p = std::sqrt(D_const * p.dt) * normal(rng);
            const double noise_m = std::sqrt(D_const * p.dt) * normal(rng);

            nplus[i]  = std::clamp(nplus[i]  + p.dt*dNp + noise_p,  0.0, 1.0);
            nminus[i] = std::clamp(nminus[i] + p.dt*dNm + noise_m,  0.0, 1.0);
        }

        if (it % t_sub_refine == 0) {
            const std::size_t base = coarse_idx * N;
            std::copy(gamma.begin(), gamma.end(), R.gamma_mat.begin()+base);
            std::copy(nplus.begin(), nplus.end(), R.nplus_mat.begin()+base);
            std::copy(nminus.begin(), nminus.end(), R.nminus_mat.begin()+base);
            ++coarse_idx;
        }
    }

    return R;
}

Results simulate_episode_const_white_noise_periodic(const ParamsConstWhiteNoisePeriodic &p)
{
    if (p.n < 1) throw std::invalid_argument("n must be >=1");

    const int n = p.n;
    const std::size_t N = n + 1;
    const double n2 = double(n)*double(n);

    std::vector<double> s, gamma, nplus, nminus;
    initialize_fields(n, p.eta, p.fstar, p.zeta, p.Nmotor,
                      s, gamma, nplus, nminus);

    double D_const;
    compute_constant_white_noise_amplitudes(p.eta, p.fstar, p.Nmotor,
                                            D_const);

    Results R;
    const auto llr = [](double x){ return (long long)std::llround(x); };
    R.n_coarse = llr(p.T * p.t_sub) + 1;
    R.n_nodes = N;
    R.t_coarse.resize(R.n_coarse);
    for (std::size_t i=0;i<R.n_coarse;i++)
        R.t_coarse[i] = double(i)/(R.n_coarse-1)*p.T;

    R.gamma_mat.resize(R.n_coarse*N);
    R.nplus_mat.resize(R.n_coarse*N);
    R.nminus_mat.resize(R.n_coarse*N);

    std::vector<double> lap(N), gamma_t(N);
    std::mt19937_64 rng(p.seed);
    std::normal_distribution<double> normal(0.0, 1.0);

    const double mu_a_zeta = p.mu_a * p.zeta;
    const double one_minus_eta = 1 - p.eta;

    const std::size_t nt = llr(p.T / p.dt) + 1;
    const std::size_t t_sub_refine = std::max<std::size_t>(1, llr(1.0/(p.dt*p.t_sub)));

    std::size_t coarse_idx=0;

    for (std::size_t it=0; it<nt; ++it) {

        for (int i=0;i<=n;++i) {
            int im1 = (i==0 ? n : i-1);
            int ip1 = (i==n ? 0 : i+1);
            lap[i] = (gamma[im1] -2*gamma[i] + gamma[ip1]) * n2;
        }

        for (int i=0;i<=n;++i) {
            double denom = p.beta + mu_a_zeta*(nplus[i] + nminus[i]);
            gamma_t[i] = (lap[i] - p.mu*gamma[i] + p.mu_a*(nminus[i]-nplus[i]))/denom;
        }

        for (std::size_t i=0;i<N;++i){
            gamma[i] += gamma_t[i]*p.dt;

            double zeta_gamma = p.zeta * gamma_t[i];
            bool saturated = std::fabs(zeta_gamma) > p.ZETA_GAMMA_THRESHOLD;

            double exp_plus=1.0, exp_minus=1.0;
            if (!saturated){
                exp_plus  = std::exp(p.fstar*(1+zeta_gamma));
                exp_minus = std::exp(p.fstar*(1-zeta_gamma));
            } else {
                ++R.count_large_zeta_gamma;
            }

            double bind_plus   = p.eta*(1-nplus[i]);
            double bind_minus  = p.eta*(1-nminus[i]);
            double unbind_plus = one_minus_eta*nplus[i]*exp_plus;
            double unbind_minus = one_minus_eta*nminus[i]*exp_minus;

            double dNp = bind_plus - unbind_plus;
            double dNm = bind_minus - unbind_minus;

            double noise_p = std::sqrt(D_const*p.dt)*normal(rng);
            double noise_m = std::sqrt(D_const*p.dt)*normal(rng);

            nplus[i]  = std::clamp(nplus[i]  + p.dt*dNp + noise_p, 0.0,1.0);
            nminus[i] = std::clamp(nminus[i] + p.dt*dNm + noise_m, 0.0,1.0);
        }

        if (it % t_sub_refine == 0){
            const std::size_t base = coarse_idx*N;
            std::copy(gamma.begin(), gamma.end(),  R.gamma_mat.begin()+base);
            std::copy(nplus.begin(), nplus.end(),  R.nplus_mat.begin()+base);
            std::copy(nminus.begin(), nminus.end(), R.nminus_mat.begin()+base);
            ++coarse_idx;
        }
    }

    return R;
}

Results simulate_episode_const_white_noise_open(const ParamsConstWhiteNoiseOpen &p)
{
    if (p.n < 1) throw std::invalid_argument("n must be >=1");

    const int n = p.n;
    const std::size_t N = n + 1;
    const double n2 = double(n)*double(n);

    std::vector<double> s, gamma, nplus, nminus;
    initialize_fields(n, p.eta, p.fstar, p.zeta, p.Nmotor,
                      s, gamma, nplus, nminus);

    double D_const;
    compute_constant_white_noise_amplitudes(p.eta, p.fstar, p.Nmotor,
                                            D_const);

    Results R;
    const auto llr = [](double x){ return (long long)std::llround(x); };
    R.n_coarse = llr(p.T * p.t_sub) + 1;
    R.n_nodes  = N;
    R.t_coarse.resize(R.n_coarse);
    for (std::size_t i = 0; i < R.n_coarse; ++i)
        R.t_coarse[i] = double(i) / (R.n_coarse - 1) * p.T;

    R.gamma_mat.resize(R.n_coarse*N);
    R.nplus_mat.resize(R.n_coarse*N);
    R.nminus_mat.resize(R.n_coarse*N);

    std::vector<double> lap(N), gamma_t(N);
    std::mt19937_64 rng(p.seed);
    std::normal_distribution<double> normal(0.0, 1.0);

    const double mu_a_zeta = p.mu_a * p.zeta;
    const double one_minus_eta = 1 - p.eta;

    const std::size_t nt = llr(p.T / p.dt) + 1;
    const std::size_t t_sub_refine = std::max<std::size_t>(1, llr(1.0/(p.dt*p.t_sub)));

    std::size_t coarse_idx=0;

    for(std::size_t it=0; it<nt; ++it) {
        {
            double gm1 = gamma[1];
            lap[0] = (gamma[1] - 2*gamma[0] + gm1) * n2;

            for (int i=1; i<n; ++i)
                lap[i] = (gamma[i-1] -2*gamma[i] + gamma[i+1]) * n2;

            double gp1 = gamma[n-1];
            lap[n] = (gamma[n-1] - 2*gamma[n] + gp1) * n2;
        }

        for (int i=0;i<=n;++i){
            const double denom = p.beta + mu_a_zeta*(nplus[i] + nminus[i]);
            gamma_t[i] = (lap[i] - p.mu*gamma[i] + p.mu_a*(nminus[i]-nplus[i]))/denom;
        }

        for (std::size_t i=0;i<N;++i) {
            gamma[i] += gamma_t[i]*p.dt;

            double zeta_gamma = p.zeta * gamma_t[i];
            bool saturated = std::fabs(zeta_gamma) > p.ZETA_GAMMA_THRESHOLD;

            double exp_plus=1.0, exp_minus=1.0;
            if (!saturated){
                exp_plus  = std::exp(p.fstar*(1+zeta_gamma));
                exp_minus = std::exp(p.fstar*(1-zeta_gamma));
            } else {
                ++R.count_large_zeta_gamma;
            }

            double bind_plus   = p.eta*(1-nplus[i]);
            double bind_minus  = p.eta*(1-nminus[i]);
            double unbind_plus = one_minus_eta*nplus[i]*exp_plus;
            double unbind_minus = one_minus_eta*nminus[i]*exp_minus;

            double dNp = bind_plus - unbind_plus;
            double dNm = bind_minus - unbind_minus;

            double noise_p = std::sqrt(D_const*p.dt)*normal(rng);
            double noise_m = std::sqrt(D_const*p.dt)*normal(rng);

            nplus[i]  = std::clamp(nplus[i]  + p.dt*dNp + noise_p, 0.0,1.0);
            nminus[i] = std::clamp(nminus[i] + p.dt*dNm + noise_m, 0.0,1.0);
        }

        if (it % t_sub_refine == 0){
            const std::size_t base = coarse_idx*N;
            std::copy(gamma.begin(), gamma.end(), R.gamma_mat.begin()+base);
            std::copy(nplus.begin(), nplus.end(), R.nplus_mat.begin()+base);
            std::copy(nminus.begin(), nminus.end(), R.nminus_mat.begin()+base);
            ++coarse_idx;
        }
    }

    return R;
}

Results simulate_episode_cm_ps_old(const ParamsCMPS& p)
{
    if (p.n < 1) throw std::invalid_argument("n must be >= 1");
    if (p.dt <= 0.0) throw std::invalid_argument("dt must be > 0");
    if (p.t_sub <= 0.0) throw std::invalid_argument("t_sub must be > 0");
    if (p.T < 0.0) throw std::invalid_argument("T must be >= 0");

    const int n = p.n;
    const std::size_t N = static_cast<std::size_t>(n) + 1;
    const double n2 = static_cast<double>(n) * static_cast<double>(n);

    // Precompute constants
    const double mu_a_zeta = p.mu_a * p.zeta;
    const double Nm_dt = p.Nmotor * p.dt;
    const double Nm_r_dt = Nm_dt * p.ps_rate;
    const double one_minus_eta = 1.0 - p.eta;

    // Initialize fields
    std::vector<double> s, gamma, nplus, nminus;
    initialize_fields(n, p.eta, p.fstar, p.zeta, p.Nmotor, s, gamma, nplus, nminus);

    // Prepare result container
    Results R;
    const auto llr = [](double x){ return static_cast<long long>(std::llround(x)); };
    R.n_coarse = static_cast<std::size_t>(llr(p.T * p.t_sub)) + 1;
    R.n_nodes  = N;
    R.t_coarse.resize(R.n_coarse);
    for (std::size_t i = 0; i < R.n_coarse; ++i)
        R.t_coarse[i] = (R.n_coarse == 1) ? 0.0 : (static_cast<double>(i) / (R.n_coarse - 1)) * p.T;

    R.gamma_mat.resize(R.n_coarse * N);
    R.nplus_mat.resize(R.n_coarse * N);
    R.nminus_mat.resize(R.n_coarse * N);

    // Buffers
    std::vector<double> gamma_t(N, 0.0);
    std::vector<double> lap(N, 0.0);

    // RNG
    std::mt19937_64 rng(p.seed);
    std::uniform_real_distribution<double> uni(0.0, 1.0);

    // Thresholds
    const double ZETA_GAMMA_THRESHOLD = p.ZETA_GAMMA_THRESHOLD;
    const double LAMBDA_SMALL_THRESHOLD = p.LAMBDA_SMALL_THRESHOLD;

    auto sample_poisson = [&](double lambda) -> std::uint64_t {
        if (lambda <= 0.0) return 0;
        if (lambda < LAMBDA_SMALL_THRESHOLD)
            return (uni(rng) < lambda) ? 1U : 0U;

        if (lambda > 1e12) {
            throw std::runtime_error("sample_poisson(): lambda exceeds 1e12 — possible numerical instability");
        }

        ++R.count_regular_poisson;
        std::poisson_distribution<std::uint64_t> dist(lambda);
        return dist(rng);
    };

    // Time setup
    const std::size_t nt = static_cast<std::size_t>(llr(p.T / p.dt)) + 1;
    const std::size_t t_sub_refine = std::max<std::size_t>(1, llr(1.0 / (p.dt * p.t_sub)));

    std::size_t coarse_idx = 0;

    // Main simulation 
    for (std::size_t it = 0; it < nt; ++it) {

        // Laplacian (inner points)
        for (int i = 1; i < n; ++i)
            lap[i] = (gamma[i - 1] - 2.0 * gamma[i] + gamma[i + 1]) * n2;

        // gamma_t (inner + boundary)
        for (int i = 1; i < n; ++i) {
            const double exp_ps_nplus = nplus[i] * Nm_r_dt;
            const double ps_plus = static_cast<double>(sample_poisson(exp_ps_nplus))/Nm_r_dt;

            const double exp_ps_nminus = nminus[i] * Nm_r_dt;
            const double ps_minus = static_cast<double>(sample_poisson(exp_ps_nminus))/Nm_r_dt;

            const double denom = p.beta + mu_a_zeta * (ps_plus + ps_minus);
            if (denom < 1.0 || !std::isfinite(denom)) {
                std::cerr << "Small/invalid denom at i=" << i
                        << " denom=" << denom
                        << " beta=" << p.beta
                        << " mu_a_zeta=" << mu_a_zeta
                        << " n+ n-=" << ps_plus << " " << ps_minus
                        << " gamma=" << gamma[i] << "\n";
            }

            gamma_t[i] = (lap[i] - p.mu * gamma[i] + p.mu_a * (ps_minus - ps_plus)) / denom;
        }
        gamma_t[0] = -gamma[0] / p.dt;
        {
            const std::size_t kn = static_cast<std::size_t>(n);
            const double exp_ps_nplus = nplus[kn] * Nm_r_dt;
            const double ps_plus = static_cast<double>(sample_poisson(exp_ps_nplus))/Nm_r_dt;
            const double exp_ps_nminus = nminus[kn] * Nm_r_dt;
            const double ps_minus = static_cast<double>(sample_poisson(exp_ps_nminus))/Nm_r_dt;

            const double denom_n = p.beta + mu_a_zeta * (ps_plus + ps_minus);
            if (denom_n < 1.0 || !std::isfinite(denom_n)) {
                std::cerr << "Invalid denom_n=" << denom_n << "\n";
            }

            const double lap_last = (2.0 * gamma[n - 1] - 2.0 * gamma[kn]) * n2;
            gamma_t[kn] = (lap_last - p.mu * gamma[kn] + p.mu_a * (ps_minus - ps_plus)) / denom_n;
        }

        // Stochastic motor updates
        for (std::size_t i = 0; i < N; ++i) {
            gamma[i]  += gamma_t[i] * p.dt;
            
            const double zeta_gamma = p.zeta * gamma_t[i];
            const bool saturated = std::fabs(zeta_gamma) > ZETA_GAMMA_THRESHOLD;

            double exp_plus  = 1.0, exp_minus = 1.0;
            if (!saturated) {
                exp_plus  = std::exp(p.fstar * (1.0 + zeta_gamma));
                exp_minus = std::exp(p.fstar * (1.0 - zeta_gamma));
            } else {
                ++R.count_large_zeta_gamma;
            }
            // one variant there is also an other option
            double bind_plus   = p.eta * (1.0 - nplus[i])  * Nm_dt;
            double bind_minus  = p.eta * (1.0 - nminus[i]) * Nm_dt;
            double unbind_plus  = one_minus_eta * nplus[i]  * exp_plus  * Nm_dt;
            double unbind_minus = one_minus_eta * nminus[i] * exp_minus * Nm_dt;

            if (saturated) {
                // Early exit: only sample one side depending on zeta_gamma sign
                unbind_plus = unbind_minus = 0.0;

                if (zeta_gamma > 0.0) {
                    nplus[i] = 0.0;
                    bind_plus = 0.0;
                    const double dNm = static_cast<double>(sample_poisson(bind_minus));
                    nminus[i] = std::clamp(nminus[i] + dNm / p.Nmotor, 0.0, 1.0);
                } else {
                    nminus[i] = 0.0;
                    bind_minus = 0.0;
                    const double dNp = static_cast<double>(sample_poisson(bind_plus));
                    nplus[i] = std::clamp(nplus[i] + dNp / p.Nmotor, 0.0, 1.0);
                }
                continue;  // skip rest for saturated case
            }

            const double dNp = static_cast<double>(sample_poisson(bind_plus))
                            - static_cast<double>(sample_poisson(unbind_plus));
            const double dNm = static_cast<double>(sample_poisson(bind_minus))
                            - static_cast<double>(sample_poisson(unbind_minus));

            
            nplus[i]  = std::clamp(nplus[i] + dNp / p.Nmotor,  0.0, 1.0);
            nminus[i] = std::clamp(nminus[i] + dNm / p.Nmotor, 0.0, 1.0);
        }


        // Record coarse data
        if (it % t_sub_refine == 0 && coarse_idx < R.n_coarse) {
            const std::size_t base = coarse_idx * N;
            std::copy(gamma.begin(),  gamma.end(),  R.gamma_mat.begin()  + base);
            std::copy(nplus.begin(),  nplus.end(),  R.nplus_mat.begin()  + base);
            std::copy(nminus.begin(), nminus.end(), R.nminus_mat.begin() + base);
            ++coarse_idx;
        }
    }

    return R;
}

Results simulate_episode_cm_ps(const ParamsCMPS& p)
{
    if (p.n < 1) throw std::invalid_argument("n must be >= 1");
    if (p.dt <= 0.0) throw std::invalid_argument("dt must be > 0");
    if (p.t_sub <= 0.0) throw std::invalid_argument("t_sub must be > 0");
    if (p.T < 0.0) throw std::invalid_argument("T must be >= 0");

    const int n = p.n;
    const std::size_t N = static_cast<std::size_t>(n) + 1;
    const double n2 = static_cast<double>(n) * static_cast<double>(n);

    // Precompute constants
    const double mu_a_zeta = p.mu_a * p.zeta;
    const double Nm_dt = p.Nmotor * p.dt;
    const double one_minus_eta = 1.0 - p.eta;
    const double duty = p.ps_rate / (p.ps_rate + p.rest_rate);
    const double inv_duty = 1.0 / duty;

    // Initialize fields
    std::vector<double> s, gamma, nplus_rest, nminus_rest, nplus_ps, nminus_ps;
    initialize_fields(n, p.eta, p.fstar, p.zeta, p.Nmotor, s, gamma, nplus_rest, nminus_rest, nplus_ps, nminus_ps);

    // Prepare result container
    Results R;
    const auto llr = [](double x){ return static_cast<long long>(std::llround(x)); };
    R.n_coarse = static_cast<std::size_t>(llr(p.T * p.t_sub)) + 1;
    R.n_nodes  = N;
    R.t_coarse.resize(R.n_coarse);
    for (std::size_t i = 0; i < R.n_coarse; ++i)
        R.t_coarse[i] = (R.n_coarse == 1) ? 0.0 : (static_cast<double>(i) / (R.n_coarse - 1)) * p.T;

    R.gamma_mat.resize(R.n_coarse * N);
    R.nplus_mat.resize(R.n_coarse * N);
    R.nminus_mat.resize(R.n_coarse * N);

    // Buffers
    std::vector<double> gamma_t(N, 0.0);
    std::vector<double> lap(N, 0.0);

    // RNG
    std::mt19937_64 rng(p.seed);
    std::uniform_real_distribution<double> uni(0.0, 1.0);

    // Thresholds
    const double ZETA_GAMMA_THRESHOLD = p.ZETA_GAMMA_THRESHOLD;
    const double LAMBDA_SMALL_THRESHOLD = p.LAMBDA_SMALL_THRESHOLD;

    auto sample_poisson = [&](double lambda) -> std::uint64_t {
        if (lambda <= 0.0) return 0;
        if (lambda < LAMBDA_SMALL_THRESHOLD)
            return (uni(rng) < lambda) ? 1U : 0U;

        if (lambda > 1e12) {
            throw std::runtime_error("sample_poisson(): lambda exceeds 1e12 — possible numerical instability");
        }

        ++R.count_regular_poisson;
        std::poisson_distribution<std::uint64_t> dist(lambda);
        return dist(rng);
    };

    // Time setup
    const std::size_t nt = static_cast<std::size_t>(llr(p.T / p.dt)) + 1;
    const std::size_t t_sub_refine = std::max<std::size_t>(1, llr(1.0 / (p.dt * p.t_sub)));

    std::size_t coarse_idx = 0;

    // Main simulation loop
    for (std::size_t it = 0; it < nt; ++it) {

        // Laplacian (inner points)
        for (int i = 1; i < n; ++i)
            lap[i] = (gamma[i - 1] - 2.0 * gamma[i] + gamma[i + 1]) * n2;

        // gamma_t (inner + boundary)
        for (int i = 1; i < n; ++i) {
            const double denom = p.beta + mu_a_zeta * (nplus_ps[i] + nminus_ps[i]) * inv_duty;
            if (denom < 1.0 || !std::isfinite(denom)) {
                std::cerr << "Small/invalid denom at i=" << i
                        << " denom=" << denom
                        << " beta=" << p.beta
                        << " mu_a_zeta=" << mu_a_zeta
                        << " n+ n-=" << nplus_ps[i] << " " << nminus_ps[i]
                        << " gamma=" << gamma[i] << "\n";
            }
            gamma_t[i] = (lap[i] - p.mu * gamma[i] + p.mu_a * (nminus_ps[i] - nplus_ps[i]) * inv_duty) / denom;
        }
        gamma_t[0] = -gamma[0] / p.dt;
        {
            const std::size_t kn = static_cast<std::size_t>(n);
            const double denom_n = p.beta + mu_a_zeta * (nplus_ps[kn] + nminus_ps[kn]) * inv_duty;
            if (denom_n < 1.0 || !std::isfinite(denom_n)) {
                std::cerr << "Invalid denom_n=" << denom_n << "\n";
            }

            const double lap_last = (2.0 * gamma[n - 1] - 2.0 * gamma[kn]) * n2;
            gamma_t[kn] = (lap_last - p.mu * gamma[kn] + p.mu_a * (nminus_ps[kn] - nplus_ps[kn]) * inv_duty) / denom_n;
        }

        // Stochastic motor updates
        for (std::size_t i = 0; i < N; ++i) {
            gamma[i]  += gamma_t[i] * p.dt;
            
            const double zeta_gamma = p.zeta * gamma_t[i];
            const bool saturated = std::fabs(zeta_gamma) > ZETA_GAMMA_THRESHOLD;

            double exp_plus  = 1.0, exp_minus = 1.0;
            if (!saturated) {
                exp_plus  = std::exp(p.fstar * (1.0 + zeta_gamma));
                exp_minus = std::exp(p.fstar * (1.0 - zeta_gamma));
            } else {
                ++R.count_large_zeta_gamma;
            }
            // bind from unbound to rest
            double bind_plus   = p.eta * (1.0 - nplus_rest[i] - nplus_ps[i])  * Nm_dt;
            double bind_minus  = p.eta * (1.0 - nminus_rest[i] - nminus_ps[i]) * Nm_dt;
            // change from rest to ps state
            double start_plus_ps  = p.ps_rate * nplus_rest[i] * Nm_dt;
            double start_minus_ps = p.ps_rate * nminus_rest[i] * Nm_dt;
            // unbind from ps state
            double unbind_plus  = one_minus_eta * nplus_ps[i]  * exp_plus  * Nm_dt * inv_duty;
            double unbind_minus = one_minus_eta * nminus_ps[i] * exp_minus * Nm_dt * inv_duty;
            // change from ps to rest state
            double end_plus_ps  = p.rest_rate * nplus_ps[i]  * Nm_dt;
            double end_minus_ps = p.rest_rate * nminus_ps[i] * Nm_dt;
            
            if (saturated) {
                // Early exit: only sample one side depending on zeta_gamma sign
                unbind_plus = unbind_minus = 0.0;

                if (zeta_gamma > 0.0) {
                    nplus_ps[i] = 0.0;
                    bind_plus = 0.0;
                    const double dNp_rest = - static_cast<double>(sample_poisson(start_plus_ps));
                    nplus_rest[i] = std::clamp(nplus_rest[i] + dNp_rest / p.Nmotor, 0.0, 1.0);

                    const double dNm_start_ps = static_cast<double>(sample_poisson(start_minus_ps));
                    const double dNm_end_ps = static_cast<double>(sample_poisson(end_minus_ps));

                    const double dNm_rest = static_cast<double>(sample_poisson(bind_minus))
                            - dNm_start_ps
                            + dNm_end_ps;
                    const double dNm_ps = dNm_start_ps
                            - dNm_end_ps;

                    nminus_rest[i] = std::clamp(nminus_rest[i] + dNm_rest / p.Nmotor, 0.0, 1.0);
                    nminus_ps[i] = std::clamp(nminus_ps[i] + dNm_ps / p.Nmotor, 0.0, 1.0 - nminus_rest[i]);
                } else {
                    nminus_ps[i] = 0.0;
                    bind_minus = 0.0;
                    const double dNm_rest = - static_cast<double>(sample_poisson(start_minus_ps));
                    nminus_rest[i] = std::clamp(nminus_rest[i] + dNm_rest / p.Nmotor, 0.0, 1.0);

                    const double dNp_start_ps = static_cast<double>(sample_poisson(start_plus_ps));
                    const double dNp_end_ps = static_cast<double>(sample_poisson(end_plus_ps));

                    const double dNp_rest = static_cast<double>(sample_poisson(bind_plus))
                            - dNp_start_ps
                            + dNp_end_ps;
                    const double dNp_ps = dNp_start_ps
                            - dNp_end_ps;
                    nplus_rest[i] = std::clamp(nplus_rest[i] + dNp_rest / p.Nmotor, 0.0, 1.0);
                    nplus_ps[i] = std::clamp(nplus_ps[i] + dNp_ps / p.Nmotor, 0.0, 1.0 - nplus_rest[i]);
                }
                continue;  // skip rest for saturated case
            }


            const double dNp_start_ps = static_cast<double>(sample_poisson(start_plus_ps));
            const double dNm_start_ps = static_cast<double>(sample_poisson(start_minus_ps));
            const double dNp_end_ps = static_cast<double>(sample_poisson(end_plus_ps));
            const double dNm_end_ps = static_cast<double>(sample_poisson(end_minus_ps));

            const double dNp_rest = static_cast<double>(sample_poisson(bind_plus))
                            - dNp_start_ps
                            + dNp_end_ps;
            const double dNm_rest = static_cast<double>(sample_poisson(bind_minus))
                            - dNm_start_ps
                            + dNm_end_ps;
            const double dNp_ps = dNp_start_ps
                            - static_cast<double>(sample_poisson(unbind_plus))
                            - dNp_end_ps;
            const double dNm_ps = dNm_start_ps
                            - static_cast<double>(sample_poisson(unbind_minus))
                            - dNm_end_ps;


            nplus_rest[i]  = std::clamp(nplus_rest[i] + dNp_rest / p.Nmotor,  0.0, 1.0);
            nminus_rest[i] = std::clamp(nminus_rest[i] + dNm_rest / p.Nmotor, 0.0, 1.0);

            nplus_ps[i]  = std::clamp(nplus_ps[i] + dNp_ps / p.Nmotor,  0.0, 1.0 - nplus_rest[i]);
            nminus_ps[i] = std::clamp(nminus_ps[i] + dNm_ps / p.Nmotor, 0.0, 1.0 - nminus_rest[i]);
        }


        // Record coarse data
        if (it % t_sub_refine == 0 && coarse_idx < R.n_coarse) {
            const std::size_t base = coarse_idx * N;
            std::copy(gamma.begin(), gamma.end(), R.gamma_mat.begin() + base);
            for (std::size_t i = 0; i < N; ++i) {
                R.nplus_mat [base + i] = nplus_rest[i]  + nplus_ps[i];
                R.nminus_mat[base + i] = nminus_rest[i] + nminus_ps[i];
            }
            ++coarse_idx;
        }
    }

    return R;
}

Results simulate_episode_3d_deterministic(const Params3DDet& p)
{
    if (p.n < 1) throw std::invalid_argument("n must be >= 1");
    if (p.dt <= 0.0) throw std::invalid_argument("dt must be > 0");
    if (p.t_sub <= 0.0) throw std::invalid_argument("t_sub must be > 0");
    if (p.T < 0.0) throw std::invalid_argument("T must be >= 0");

    const int n = p.n;
    const std::size_t N = static_cast<std::size_t>(n) + 1;
    const double n2 = static_cast<double>(n) * static_cast<double>(n);

    // Precompute constants
    const double one_minus_eta = 1.0 - p.eta;
    const std::array<double, 4> d_tilde = {p.d_tilde_1, p.d_tilde_2, p.d_tilde_3, p.d_tilde_4};

    // Initialize fields
    std::vector<double> s, gamma_temp, nplus_temp, nminus_temp;
    initialize_fields(n, p.eta, p.fstar, p.zeta, 0.0, s, gamma_temp, nplus_temp, nminus_temp);

    // For 3D, replicate to 4 components
    std::vector<std::vector<double>> nplus(4, nplus_temp);
    std::vector<std::vector<double>> nminus(4, nminus_temp);
    std::vector<double> gamma = gamma_temp;

    // Prepare result container
    Results R;
    const auto llr = [](double x){ return static_cast<long long>(std::llround(x)); };
    R.n_coarse = static_cast<std::size_t>(llr(p.T * p.t_sub)) + 1;
    R.n_nodes  = N;
    R.t_coarse.resize(R.n_coarse);
    for (std::size_t i = 0; i < R.n_coarse; ++i)
        R.t_coarse[i] = (R.n_coarse == 1) ? 0.0 : (static_cast<double>(i) / (R.n_coarse - 1)) * p.T;

    R.gamma_mat.resize(R.n_coarse * N);
    R.nplus_mat.resize(R.n_coarse * 4 * N);
    R.nminus_mat.resize(R.n_coarse * 4 * N);

    // Buffers
    std::vector<double> gamma_t(N, 0.0);
    std::vector<double> lap(N, 0.0);

    const double ZETA_GAMMA_THRESHOLD = p.ZETA_GAMMA_THRESHOLD;

    // Time setup
    const std::size_t nt = static_cast<std::size_t>(llr(p.T / p.dt)) + 1;
    const std::size_t t_sub_refine = std::max<std::size_t>(1, llr(1.0 / (p.dt * p.t_sub)));

    std::size_t coarse_idx = 0;

    // Deterministic loop (no stochastic binding)
    for (std::size_t it = 0; it < nt; ++it) {

        // Laplacian
        for (int i = 1; i < n; ++i)
            lap[i] = (gamma[i - 1] - 2.0 * gamma[i] + gamma[i + 1]) * n2;

        // gamma_t (inner + boundary)
        for (int i = 1; i < n; ++i) {
            double active_num = 0.0;
            double active_den = 0.0;
            for (int k = 0; k < 4; ++k) {
                double dk = d_tilde[k];
                active_num += dk * (nminus[k][i] - nplus[k][i]);
                active_den += dk * dk * (nplus[k][i] + nminus[k][i]);
            }
            const double denom = p.beta + p.mu_a * p.zeta * active_den;
            if (denom <= 0.0 || !std::isfinite(denom)) {
                // Handle invalid denom
                gamma_t[i] = 0.0;
            } else {
                gamma_t[i] = (lap[i] - p.mu * gamma[i] + p.mu_a * active_num) / denom;
            }
        }
        gamma_t[0] = 0.0;  // Fixed boundary
        {
            const std::size_t kn = static_cast<std::size_t>(n);
            double active_num = 0.0;
            double active_den = 0.0;
            for (int k = 0; k < 4; ++k) {
                double dk = d_tilde[k];
                active_num += dk * (nminus[k][kn] - nplus[k][kn]);
                active_den += dk * dk * (nplus[k][kn] + nminus[k][kn]);
            }
            const double denom_n = p.beta + p.mu_a * p.zeta * active_den;
            if (denom_n <= 0.0 || !std::isfinite(denom_n)) {
                gamma_t[kn] = 0.0;
            } else {
                const double lap_last = (2.0 * gamma[n - 1] - 2.0 * gamma[kn]) * n2;
                gamma_t[kn] = (lap_last - p.mu * gamma[kn] + p.mu_a * active_num) / denom_n;
            }
        }

        // Deterministic update (mean-field ODEs)
        for (std::size_t i = 0; i < N; ++i) {
            for (int k = 0; k < 4; ++k) {
                double dk = d_tilde[k];
                double zeta_gamma = p.zeta * dk * gamma_t[i];
                const bool saturated = std::fabs(zeta_gamma) > ZETA_GAMMA_THRESHOLD;

                double exp_plus = 1.0, exp_minus = 1.0;
                if (!saturated) {
                    exp_plus  = std::exp(p.fstar * (1.0 + zeta_gamma));
                    exp_minus = std::exp(p.fstar * (1.0 - zeta_gamma));
                } else {
                    ++R.count_large_zeta_gamma;
                }

                const double bind_plus   = p.eta * (1.0 - nplus[k][i]);
                const double bind_minus  = p.eta * (1.0 - nminus[k][i]);
                const double unbind_plus  = one_minus_eta * nplus[k][i]  * exp_plus;
                const double unbind_minus = one_minus_eta * nminus[k][i] * exp_minus;

                // ODE evolution
                const double dNp = bind_plus - unbind_plus;
                const double dNm = bind_minus - unbind_minus;

                nplus[k][i]  = std::clamp(nplus[k][i] + p.dt * dNp,  0.0, 1.0);
                nminus[k][i] = std::clamp(nminus[k][i] + p.dt * dNm, 0.0, 1.0);
            }
            gamma[i] += gamma_t[i] * p.dt;
        }
        gamma[0] = 0.0;  // Enforce boundary

        // Record coarse data
        if (it % t_sub_refine == 0 && coarse_idx < R.n_coarse) {
            const std::size_t base_gamma = coarse_idx * N;
            std::copy(gamma.begin(), gamma.end(), R.gamma_mat.begin() + base_gamma);
            for (int k = 0; k < 4; ++k) {
                const std::size_t base = coarse_idx * 4 * N + k * N;
                std::copy(nplus[k].begin(), nplus[k].end(), R.nplus_mat.begin() + base);
                std::copy(nminus[k].begin(), nminus[k].end(), R.nminus_mat.begin() + base);
            }
            ++coarse_idx;
        }
    }

    R.count_regular_poisson = 0;
    R.count_poisson_overflow = 0;
    R.count_invalid_denom = 0;  // Not tracked in deterministic

    return R;
}

Results simulate_episode_3d(const Params3D& p)
{
    if (p.n < 1) throw std::invalid_argument("n must be >= 1");
    if (p.dt <= 0.0) throw std::invalid_argument("dt must be > 0");
    if (p.t_sub <= 0.0) throw std::invalid_argument("t_sub must be > 0");
    if (p.T < 0.0) throw std::invalid_argument("T must be >= 0");
    if (p.Nmotor <= 0.0) throw std::invalid_argument("Nmotor must be > 0");

    const int n = p.n;
    const std::size_t N = static_cast<std::size_t>(n) + 1;
    const double n2 = static_cast<double>(n) * static_cast<double>(n);

    // Precompute constants
    const double one_minus_eta = 1.0 - p.eta;
    const double Nm_dt = p.Nmotor * p.dt;
    const std::array<double, 4> d_tilde = {p.d_tilde_1, p.d_tilde_2, p.d_tilde_3, p.d_tilde_4};

    // Initialize fields
    std::vector<double> s, gamma_temp, nplus_temp, nminus_temp;
    initialize_fields(n, p.eta, p.fstar, p.zeta, p.Nmotor, s, gamma_temp, nplus_temp, nminus_temp);

    // For 3D, replicate to 4 components
    std::vector<std::vector<double>> nplus(4, nplus_temp);
    std::vector<std::vector<double>> nminus(4, nminus_temp);
    std::vector<double> gamma = gamma_temp;

    // Prepare result container
    Results R;
    const auto llr = [](double x){ return static_cast<long long>(std::llround(x)); };
    R.n_coarse = static_cast<std::size_t>(llr(p.T * p.t_sub)) + 1;
    R.n_nodes  = N;
    R.t_coarse.resize(R.n_coarse);
    for (std::size_t i = 0; i < R.n_coarse; ++i)
        R.t_coarse[i] = (R.n_coarse == 1) ? 0.0 : (static_cast<double>(i) / (R.n_coarse - 1)) * p.T;

    R.gamma_mat.resize(R.n_coarse * N);
    R.nplus_mat.resize(R.n_coarse * 4 * N);
    R.nminus_mat.resize(R.n_coarse * 4 * N);

    // Buffers
    std::vector<double> gamma_t(N, 0.0);
    std::vector<double> lap(N, 0.0);

    // RNG
    std::mt19937_64 rng(p.seed);
    std::uniform_real_distribution<double> uni(0.0, 1.0);

    // Thresholds
    const double ZETA_GAMMA_THRESHOLD = p.ZETA_GAMMA_THRESHOLD;
    const double LAMBDA_SMALL_THRESHOLD = p.LAMBDA_SMALL_THRESHOLD;
    const double POISSON_LAMBDA_MAX = 1e12;

    auto sample_poisson = [&](double lambda) -> std::uint64_t {
        if (lambda <= 0.0) return 0;
        if (lambda < LAMBDA_SMALL_THRESHOLD)
            return (uni(rng) < lambda) ? 1U : 0U;
        if (lambda > POISSON_LAMBDA_MAX) {
            ++R.count_poisson_overflow;
            return 0;  // or some approximation, but for now 0
        }
        ++R.count_regular_poisson;
        std::poisson_distribution<std::uint64_t> dist(lambda);
        return dist(rng);
    };

    // Time setup
    const std::size_t nt = static_cast<std::size_t>(llr(p.T / p.dt)) + 1;
    const std::size_t t_sub_refine = std::max<std::size_t>(1, llr(1.0 / (p.dt * p.t_sub)));

    std::size_t coarse_idx = 0;

    // Main simulation loop
    for (std::size_t it = 0; it < nt; ++it) {

        // Laplacian (inner points)
        for (int i = 1; i < n; ++i)
            lap[i] = (gamma[i - 1] - 2.0 * gamma[i] + gamma[i + 1]) * n2;

        // gamma_t (inner + boundary)
        for (int i = 1; i < n; ++i) {
            double active_num = 0.0;
            double active_den = 0.0;
            for (int k = 0; k < 4; ++k) {
                double dk = d_tilde[k];
                active_num += dk * (nminus[k][i] - nplus[k][i]);
                active_den += dk * dk * (nplus[k][i] + nminus[k][i]);
            }
            const double denom = p.beta + p.mu_a * p.zeta * active_den;
            if (denom <= 0.0 || !std::isfinite(denom)) {
                ++R.count_invalid_denom;
                gamma_t[i] = 0.0;
            } else {
                gamma_t[i] = (lap[i] - p.mu * gamma[i] + p.mu_a * active_num) / denom;
            }
        }
        gamma_t[0] = 0.0;  // Fixed boundary
        {
            const std::size_t kn = static_cast<std::size_t>(n);
            double active_num = 0.0;
            double active_den = 0.0;
            for (int k = 0; k < 4; ++k) {
                double dk = d_tilde[k];
                active_num += dk * (nminus[k][kn] - nplus[k][kn]);
                active_den += dk * dk * (nplus[k][kn] + nminus[k][kn]);
            }
            const double denom_n = p.beta + p.mu_a * p.zeta * active_den;
            if (denom_n <= 0.0 || !std::isfinite(denom_n)) {
                ++R.count_invalid_denom;
                gamma_t[kn] = 0.0;
            } else {
                const double lap_last = (2.0 * gamma[n - 1] - 2.0 * gamma[kn]) * n2;
                gamma_t[kn] = (lap_last - p.mu * gamma[kn] + p.mu_a * active_num) / denom_n;
            }
        }

        // Stochastic motor updates
        for (std::size_t i = 0; i < N; ++i) {
            gamma[i] += gamma_t[i] * p.dt;
            for (int k = 0; k < 4; ++k) {
                double dk = d_tilde[k];
                double zeta_gamma = p.zeta * dk * gamma_t[i];
                const bool saturated = std::fabs(zeta_gamma) > ZETA_GAMMA_THRESHOLD;

                if (saturated) {
                    ++R.count_large_zeta_gamma;
                    if (zeta_gamma > 0.0) {
                        nplus[k][i] = 0.0;
                        const double lam = p.eta * (1.0 - nminus[k][i]) * Nm_dt;
                        const double jump = static_cast<double>(sample_poisson(lam));
                        nminus[k][i] = std::clamp(nminus[k][i] + jump / p.Nmotor, 0.0, 1.0);
                    } else {
                        nminus[k][i] = 0.0;
                        const double lam = p.eta * (1.0 - nplus[k][i]) * Nm_dt;
                        const double jump = static_cast<double>(sample_poisson(lam));
                        nplus[k][i] = std::clamp(nplus[k][i] + jump / p.Nmotor, 0.0, 1.0);
                    }
                    continue;
                }

                const double exp_plus  = std::exp(p.fstar * (1.0 + zeta_gamma));
                const double exp_minus = std::exp(p.fstar * (1.0 - zeta_gamma));

                const double bind_plus   = p.eta * (1.0 - nplus[k][i])  * Nm_dt;
                const double bind_minus  = p.eta * (1.0 - nminus[k][i]) * Nm_dt;
                const double unbind_plus  = one_minus_eta * nplus[k][i]  * exp_plus  * Nm_dt;
                const double unbind_minus = one_minus_eta * nminus[k][i] * exp_minus * Nm_dt;

                const double dNp = static_cast<double>(sample_poisson(bind_plus))
                                 - static_cast<double>(sample_poisson(unbind_plus));
                const double dNm = static_cast<double>(sample_poisson(bind_minus))
                                 - static_cast<double>(sample_poisson(unbind_minus));

                nplus[k][i]  = std::clamp(nplus[k][i] + dNp / p.Nmotor,  0.0, 1.0);
                nminus[k][i] = std::clamp(nminus[k][i] + dNm / p.Nmotor, 0.0, 1.0);
            }
        }
        gamma[0] = 0.0;  // Enforce boundary

        // Record coarse data
        if (it % t_sub_refine == 0 && coarse_idx < R.n_coarse) {
            const std::size_t base_gamma = coarse_idx * N;
            std::copy(gamma.begin(), gamma.end(), R.gamma_mat.begin() + base_gamma);
            for (int k = 0; k < 4; ++k) {
                const std::size_t base = coarse_idx * 4 * N + k * N;
                std::copy(nplus[k].begin(), nplus[k].end(), R.nplus_mat.begin() + base);
                std::copy(nminus[k].begin(), nminus[k].end(), R.nminus_mat.begin() + base);
            }
            ++coarse_idx;
        }
    }

    return R;
}


} // namespace spde
