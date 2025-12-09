%% Flagellar Reaction-Diffusion model, jupyter notebook by Cass & Gadelha Nature Communications 2023
% ported to Matlab and extended by Benjamin M. Friedrich, 13.09.2024-01.10.2025; status: working

% Numerical parameters ====================================================
n = 100;    % discrete segments of arclength

I = eye(n+1);   % identity matrix
S = n^2 * toeplitz( [ [-2 1] zeros(1,n-2) 1 ] );  % second derivatives matrix

s = linspace(0,1,n+1)';   % points of normalized arclength
ds = s(2);   % arclength spacing
t_sub = 20;  % subdivide unit of time by t_sub for output, i.e., dt = 1/t_sub

nsim = 1; % number of independent stochastic realizations of simulation
isim = 1;
rng(isim) % seed random number generator for reproducible simulations

save_results = false; 
% save_results = true; 

% Model parameters ========================================================
% simulation time
T = 20; % for quick testing
% T = 2000; % total simulation time, corresponds to ~500 oscillation cycles for Chlamydomonas parameter set and Nmotor=Inf (350 cycles for wt, N=100% in Sharma et al.)
% T = 10000; % total simulation time, corresponds to ~2500 oscillation cycles for Chlamydomonas parameter set and Nmotor=Inf (350 cycles for wt, N=100% in Sharma et al.)

fstar = 2;  % detachment force ratio (from Cass et al. 2023)
% fit parameters for Chlamydomonas (Nat. Comm. 2023, Table 1, p. 7) -------
mu_a = 1570; % []
mu = 10; % [] shear resistance / bending resistance
eta = 0.096; % [] duty ratio
zeta = 0.96;
% new parameter: internal sliding friction (needed to stabilize numerics near Hopf bifurcation)
% beta = 0; % [] corresponds to original model of Cass et al. 2023
beta = 2; % []
% NOTE: for beta = 2, mu_a,crit~150, gradual increase of oscillation amplitude, consistent with super-critical Hopf bifurcation

% new parameters from SBI -------------------------------------------------
mu_a = 365; % []
mu = 12; % [] shear resistance / bending resistance
eta = 0.35; % [] duty ratio
zeta = 0.86;
beta = 2.4; % [] % internal sliding friction (needed to stabilize numerics near Hopf bifurcation)
fstar = 2.2;  % detachment force ratio (from Cass et al. 2023)
Nmotor = 1.7e4/(2*n);
fprintf(1,'Using new parameters from SBI:\n')

% Nmotor = Inf; % deterministic limit of infinite motor number
% Nmotor = 5e2; % number of motors per axonemal segment of length ds = 100nm, rho = 100'000 motors/um --> used in Figs. 2 and 3
% Nmotor = 1e2; % number of motors per axonemal segment of length ds = 100nm, rho = 20'000 motors/um = realistic, but beat pattern too noisy
% NOTE: motor number N used in manuscript corresponds to N = 2*n*Nmotor

p_remain = 1; % [] fraction of non-extracted motors, 1=no motor extraction

% Time-averaged mean shape of axoneme =====================================
C0 = -0.24; % [rad/um] mean cilia curvature of static component of Chlamydomonas cilia beat [Geyer et al. Curr Biol 2016]; sperm flagella: C0 = 0.0191 rad/um [JEB 2010]
L_in_micron = 10; % [um]
ds = L_in_micron/n; % [um]
gamma0 = pi/2 + C0*(0:ds:L_in_micron); % [rad] % tangent angle profile of static component

fprintf(1,'Nmotor=%d, mu_a=%.2f, isim=%d ...\n', Nmotor, mu_a, isim)        


%% Solve reaction diffusion PDE ===========================================

b = eta + (1-eta)*exp(fstar);
n0 = eta/b;
if Nmotor < Inf
    n0 = round(n0*Nmotor) / Nmotor; % for discrete motor number, nplus, nminus, n0 should change in units of 1/Nmotor
end

% Initial condition -------------------------------------------------------
% (uncomment your choice)
% ... small, symmetric peak
% gamma = 0.0015*exp( -(s-0.5).^2/0.1^2 ); % small gaussian initial condition
% nplus = n0*ones(n+1,1); nminus = n0*ones(n+1,1); % steady-state value for fraction of bound motors

% ... straight axoneme
% gamma = zeros(n+1,1); 
% nplus = n0*ones(n+1,1); nminus = n0*ones(n+1,1); % steady-state value for fraction of bound motors

% ... snap-shot of full-amplitude oscillation (from noise-free simulation with standard parameters)
load('x0.mat'); % --> x0; 
gamma = x0(1:n+1); 
nplus = x0(n+2:2*n+2); 
nminus = x0(2*n+3:end);

gamma_t = zeros(size(gamma)); 

% Simple Euler scheme -----------------------------------------------------
%   slightly less acurate, but usable for stochastic differential equations

% nunmerical parameters
dt = 1e-4; % fixed time-step for Euler scheme (numerics becomes unstable for dt = 5e-3)       
t_sub_refine = round(1/dt/t_sub); % how many Euler time steps by time interval between control time points (used for output)
nt = length( 0:dt:T ); % number of time steps

% for output
t = linspace(0, T, T*t_sub+1)'; % control time points
gamma_mat = nan( length(t), n+1 ); % allocate memory for results (1st dimension = time, 2nd dimension = arc-length)
nplus_mat = nan( length(t), n+1 ); nminus_mat = nan( length(t), n+1 );
% gamma_mat(1,:) = gamma; nplus_mat(1,:) = nplus; nminus_mat(1,:) = nminus; % store initial condition

% ... loop over time steps
tic
for it = 1:nt
    % regularization_constant = 0; % original code: prevent motor populations from becoming negative (should not occur anymore if bionomial random numbers for jump process are used)
    regularization_constant = 5e-3; % prevent (rare) division by zero (reduces oscillation amplitude by 10%)       
    nplus  = min( max( nplus,  regularization_constant ), 1 );
    nminus = min( max( nminus, regularization_constant ), 1 ); 
    % watch out for numerical instabilities
    if any(~isfinite(gamma)) 
        error( 'gamma became nan at t=%.2f', it*dt )
    end

    % Compute energy dissipation ------------------------------------------
    % ... see version v4_energetic_cost.m

    % keep a record every t_sub_refine-th time step =======================
    if mod(it, t_sub_refine ) == 1
        it_coarse = round( it / t_sub_refine ) + 1;  
        gamma_mat(  it_coarse, : ) = gamma; 
        nplus_mat(  it_coarse, : ) = nplus; 
        nminus_mat( it_coarse, : ) = nminus;         
    end

    % tangent angle profile: update =======================================    
    
    % NUMERICAL EXPERIMENT: stalling by external force?
    % a = 0.2; % [um] sliding length (new preprint by Cass et al. 2024)    
    L = 1; % [] normalized length
    zetapar = 0.69e-3; % [pN s/um^2], parameters resistive force theory (Friedrich et al. J. exp. Biol. 2010)
    zetaper=1.81*zetapar; 
    v_ext = 1e4; % [um/s] = 10 mm/s (cf. Klindt et al. PRL 2016)
    L_in_micron = 10; % [um]
    B = 840; % [pN um^2] bending resistance (CAUTION: Cass et al. 2023 states wrong unit)
    time_factor = 1/250; % [] --> [s] to scale time from simulation units to physical units
    length_factor = 10; % [] --> [um] to scale length from simulation units to physical units
    % zetaper*v_ext*L_in_micron --> 125 pN
    % zetaper*v_ext*L_in_micron / (B/L_in_micron^2) --> 14.9 [] (should be independent of 'time_factor')

    fext_factor = +14.9*10; % corresponding to flow speed +100 mm/s along positive y-axis; Fig. S3A
    % fext_factor = -14.9*10; % corresponding to flow speed -100 mm/s along positive y-axis; Fig. S3B
    fext = fext_factor * cos( gamma0(:) + gamma ) .* (1-s);
    % NOTE: gamma0 [rad]: time-averaged shape of cilium, see line 227 below
    
    % Default code: without external force fext:
    % gamma_t = (S * gamma - mu*gamma + mu_a*(nminus - nplus)) ./ (beta + mu_a*zeta*(nplus + nminus)); % Eq. (S.55) in Cass et al. 2023, our Eq. (1)      
    % Modified code: including external force line density in torque line density equation
    gamma_t = (S * gamma - mu*gamma + mu_a*(nminus - nplus) + fext ) ./ (beta + mu_a*zeta*(nplus + nminus)); % Eq. (S.55) in Cass et al. 2023, our Eq. (1), modified
    % NOTE: sliding friction with friction coefficient 'beta' was added
    % END OF MODIFIED CODE ================================================

    % apply boundary conditions for gamma
    % proximal end (s=0): gamma(1) = 0 --> 
    gamma_t(1) = -gamma(1)/dt; 
    % distal end (s=1): d gamma/ds|s=L = 0 --> apply Eq. (S.55) with virtual gamma(n+2) = gamma(n+1)
    gamma_t(n+1) = ( n^2*(2*gamma(n) - 2*gamma(n+1)) - mu*gamma(n+1) + mu_a*(nminus(n+1) - nplus(n+1)) ) ...
                     ./ (beta + mu_a*zeta*( nplus(n+1) + nminus(n+1) ));
    
    % bound motor fraction: deterministic update ==========================
    if Nmotor == Inf % infinite number of motors (=renormalization fixed point)
        nplus_t  = eta*(1 - nplus)  - (1-eta)*nplus .*exp( fstar*(1 + zeta*gamma_t) ); % Eq. (S.56)
        nminus_t = eta*(1 - nminus) - (1-eta)*nminus.*exp( fstar*(1 - zeta*gamma_t) ); %  same
        % Euler update step ---------------------------------------------------
        gamma  = gamma  + gamma_t *dt; 
        nplus  = nplus  + nplus_t *dt;
        nminus = nminus + nminus_t*dt;
        
        continue
    end
        
    % bound motor fraction: stochastic update =============================    
    % ... expecation values of number of binding/unbinding motors ---------
    Nplus_bind_expect   =    eta *(1 - nplus)                               *Nmotor*dt *p_remain; % Eq. (S.56)
    Nplus_unbind_expect = (1-eta)* nplus .*exp( fstar*(1 + zeta*gamma_t) )  *Nmotor*dt *p_remain;

    Nminus_bind_expect   =    eta *(1 - nminus)                             *Nmotor*dt *p_remain;
    Nminus_unbind_expect = (1-eta)* nminus.*exp( fstar*(1 - zeta*gamma_t) ) *Nmotor*dt *p_remain;

    % ... Poisson distributed random numbers with prescribed expectation values   
    Nplus_bind    = poissrnd( Nplus_bind_expect    );
    Nplus_unbind  = poissrnd( Nplus_unbind_expect  );

    Nminus_bind   = poissrnd( Nminus_bind_expect   );
    Nminus_unbind = poissrnd( Nminus_unbind_expect );
    % Note: the approximation of the numbers of binding/unbinding motors as Poissonian random numbers may cause nplus & nminus to exceed the interval [0,1] of permissable values; this is taken care of below   

    % Euler update step
    gamma  = gamma  + gamma_t*dt; % gamma = Delta/a
    nplus  = nplus  + ( Nplus_bind  - Nplus_unbind  ) / Nmotor; 
    nminus = nminus + ( Nminus_bind - Nminus_unbind ) / Nmotor; 
            
end % it

toc

if save_results
    fname = sprintf('Nmotor_%d_mu_a_%.2f_isim_%d_gamma0_peak.mat', Nmotor, mu_a, nsim );
    dname = './simdata_Hopf_bifurcation/'; 
    save( [dname fname], 'gamma_mat','nplus_mat','nminus_mat','Nmotor','p_remain', 'mu_a', 'mu', 'zeta', 'beta'); 
end % save_results?

fprintf(1,'Simulation completed!\n')

%% plot kymograph of tangent angle ----------------------------------------
figure(1), clf 
imshow(flipud(gamma_mat'),[],'InitialMagnification','fit')
colormap jet
% clim([-1 1]*pi/2)
colorbar

%% plot axonemal shapes ---------------------------------------------------
% gamma_mat : 1st dim = time, 2nd dim = arc-length
nt_coarse = size(gamma_mat,1); 
nt_skip = 200; 
xflag = cumsum( cos( gamma_mat(nt_skip+1:end,:) + repmat(gamma0, nt_coarse-nt_skip,1 ) ), 2) * ds; % [um]
yflag = cumsum( sin( gamma_mat(nt_skip+1:end,:) + repmat(gamma0, nt_coarse-nt_skip,1 ) ), 2) * ds; 
figure(1), clf, hold on
plot( xflag', yflag', 'color', 0.8*[1 1 1])
plot( xflag(end,:), yflag(end,:), 'r', 'LineWidth', 2 )
daspect([1 1 1])

quiver( 0:10, -4*ones(1,11), zeros(1,11), ones(1,11)*fext_factor/100, 0, 'color', [0,0,0.7] )

saveas(gcf,'figure_S_flow_Nmotor_500_vext_100_mm_per_sec_new_params.eps','epsc') % fext_factor = +14.9*10; % corresponding to flow speed +100 mm/s along positive y-axis
% saveas(gcf,'figure_S_flow_Nmotor_500_vext_minus_100_mm_per_sec_new_params.eps','epsc') % fext_factor = -14.9*10; % corresponding to flow speed -100 mm/s along positive y-axis