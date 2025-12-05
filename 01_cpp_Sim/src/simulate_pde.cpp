#include "simulate_pde.hpp"

#include <cmath>
#include <algorithm>
#include <random>
#include <fstream>
#include <stdexcept>
#include <limits>
#include <iostream>
#include <cstdint> 

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

// -----------------------------------------------------------------------------
// Initialization
// -----------------------------------------------------------------------------
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


// -----------------------------------------------------------------------------
// Simulation
// -----------------------------------------------------------------------------
Results simulate_episode(const Params& p)
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

    // -------------------------------------------------------------------------
    // Main simulation loop
    // -------------------------------------------------------------------------
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

Results simulate_episode_deterministic(const Params& p)
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

    // -------------------------------------------------------------------------
    // Deterministic loop (no stochastic binding)
    // -------------------------------------------------------------------------
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

Results simulate_episode_white_noise(const Params &p) {
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

    // -------------------------------------------------------------------------
    // Main simulation loop
    // -------------------------------------------------------------------------
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

        // ---------------------------------------------------------------------
        // White-noise approximation for motor dynamics
        // ---------------------------------------------------------------------
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

Results simulate_episode_const_white_noise(const Params &p){
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

Results simulate_episode_const_white_noise_periodic(const Params &p)
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

Results simulate_episode_const_white_noise_open(const Params &p)
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

        // Open BC = Neumann; mirror ghost nodes
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


} // namespace spde
