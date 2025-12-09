%% Flagellar Reaction-Diffusion model, jupyter notebook by Cass & Gadelha Nature Communications 2023
% ported to Matlab by Benjamin Friedrich, 13.09.2024; status: working

% Notes: 
% - total dissipation 200 fW (=plausible; only positive work counted)
% - total dissipation ~0 (if also negative work included); reason unclear
% - non-isochronous oscillator: amplitude fluctuations --> 2nd harmonic in phase ACF
% - weakly chaotic: not visible in 3D PCA-shape space

% Numerical parameters ====================================================-
n = 100;    % discrete segments of arclength

I = eye(n+1);   % identity matrix
S = n^2 * toeplitz( [ [-2 1] zeros(1,n-2) 1 ] );  % second derivatives matrix

s = linspace(0,1,n+1)';   % points of normalized arclength
ds = s(2);   % arclength spacing
t_sub = 20;  % subdivide unit of time by t_sub for output, i.e., dt = 1/t_sub

%  Change parameter values here ===========================================
% simulation time
T = 100; % for quick testing
% T = 2000; % total simulation time, corresponds to ~500 oscillation cycles for Chlamydomonas parameter set and Nmotor=Inf (350 cycles for wt, N=100% in Sharma et al.)
% T = 10000; % total simulation time, corresponds to ~2500 oscillation cycles for Chlamydomonas parameter set and Nmotor=Inf (350 cycles for wt, N=100% in Sharma et al.)

% fit parameters for Chlamydomonas (Nat. Comm. 2023, Table 1, p. 7) -------
mu_a = 1570; % [] motor activity parameter
mu = 10; % [] shear resistance / bending resistance
eta = 0.096; % [] motor duty ratio
zeta = 0.96; % [] normalized sliding length-scale (axoneme diameter / motor distance)
% new parameter: internal sliding friction (needed to stabilize numerics near Hopf bifurcation)
% beta = 0; % [] corresponds to original model of Cass et al. 2023
beta = 2; % [] sliding friction
fstar = 2; % [] normalized motor force (from Cass et al. 2023)
time_factor = 1/250; % [] --> [s] to scale time from simulation units to physical units
tau = 1*time_factor; % [s]=[]*[s]

% Nmotor = Inf; % deterministic limit of infinite motor number
% Nmotor = 5e2; % number of motors per axonemal segment of length ds = 100nm, rho = 100'000 motors/um --> used in Figs. 2 and 3
% Nmotor = 1e2; % number of motors per axonemal segment of length ds = 100nm, rho = 20'000 motors/um = realistic, but beat pattern too noisy

% New parameter values from SBI -------------------------------------------
mu_a = 365; % []
mu = 12; % [] shear resistance / bending resistance
eta = 0.35; % [] duty ratio
zeta = 0.86;
beta = 2.4; % [] % internal sliding friction (needed to stabilize numerics near Hopf bifurcation)
fstar = 2.2;  % detachment force ratio (from Cass et al. 2023)
Nmotor = 1.7e4/(2*n);
% Nmotor = Inf;
fprintf(1,'Using new parameters from SBI:\n')

% convert non-dimensional parameters back to dimensional parameters -------
L = 10; % [um] flagellar length
a = 0.2; % [um] sliding length (new preprint by Cass et al. 2024)
K = 2e3; % [pN/um^2], sliding stiffness (Nexin linker); note: value consistent with a^2 K = 80; % [pN/rad] 
B = 840; % [pN um^2] bending resistance (CAUTION: Cass et al. 2023 states wrong unit)
rho = 1e3; % [1/um] (new preprint by Cass et al. 2024)
% mu = a^2*K*L^2/B; % true
f0 = mu_a*B/(a*rho*L^2); % [pN]
fc = f0/fast; % [pN] 
% Note: fc = 0.5 - 2.5; % [pN] (new preprint by Cass et al. 2024) 
% Note: f0 = 1-5; % [pN] (new preprint by Cass et al. 2024) --> discrepancy: f0 = 65.9 pN
time_factor = 4.7/1000; % [] --> [s] to scale time from simulation units to physical units
tau = 1*time_factor; % [s]=[]*[s]
pi0 = eta/tau; % [1/s] binding rate
epsilon0 = (1-eta)/tau; % [1/s] basal unbinding rate (unbinding rate = epsilon0*exp(fast) for gamma_t = 0)
tau_b = ( pi0 + epsilon0*exp(f0/fc) )^(-1); % [s] characteristic time-scale of motor relaxation
v0 = a / (zeta*tau); % [um/s] characteristic motor speed 
% Note: v0 = 5-7; % [um/s] (new preprint by Casset al. 2024) --> discrepancy: v0 = 52.0 um/s
% Note: mu = a^2*K*L^2/B; % true
fprintf(1, 'f0 = %.2f pN, tau=%.2f ms, 1/pi0 = %.2f ms, tau_b = %.2f ms, v0 = %.2f um/s, v0*tau_b = %.2f nm\n', f0, tau*1e3, 1e3/pi0, 1e3*tau_b, v0, v0*1e3*tau_b )

p_remain = 1; % [] fraction of non-extracted motors, 1=no motor extraction

nsim = 1; % number of independent stochastic realizations of simulation

save_results = false;
% save_results = true;
    
isim = 1;
rng(isim) % seed random number generator for reproducible simulations

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

Rlist = nan(nt,1); % [aW] dissipation rate

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
    
    % keep a record every t_sub_refine-th time step =======================
    if mod(it, t_sub_refine ) == 1
        it_coarse = round( it / t_sub_refine ) + 1;  
        gamma_mat(  it_coarse, : ) = gamma; 
        nplus_mat(  it_coarse, : ) = nplus; 
        nminus_mat( it_coarse, : ) = nminus;         
    end

    % tangent angle profile: update =======================================
    gamma_t = (S * gamma - mu*gamma + mu_a*(nminus - nplus)) ./ (beta + mu_a*zeta*(nplus + nminus)); % Eq. (S.55)      
    % NOTE: sliding friction with friction coefficient 'beta' was added
    
    % apply boundary conditions for gamma
    % proximal end (s=0): gamma(1) = 0 --> 
    gamma_t(1) = -gamma(1)/dt; 
    % distal end (s=1): d gamma/ds|s=L = 0 --> apply Eq. (S.55) with virtual gamma(n+2) = gamma(n+1)
    gamma_t(n+1) = ( n^2*(2*gamma(n) - 2*gamma(n+1)) - mu*gamma(n+1) + mu_a*(nminus(n+1) - nplus(n+1)) ) ...
                     ./ (beta + mu_a*zeta*( nplus(n+1) + nminus(n+1) ));

    % Compute energy dissipation ------------------------------------------
       
    Fplus  = f0*( 1 + gamma_t*a/v0/time_factor ) ; % [pN] force exerted by each bound plus-motor
    Fminus = f0*( 1 - gamma_t*a/v0/time_factor ) ; % [pN] force exerted by each bound minus-motor
       
    % Presumed correction of Eq. (S.12): f = rho*( -nplus*Fplus + nminus*Fminus)
    fmotor_plus  = -rho*nplus .*Fplus; % [pN/um] force density, cf. Eq. (S.12) in SI of Cass et al. 2023
    fmotor_minus =  rho*nminus.*Fminus; 
    % ... sum all work done (including negative work, i.e. ATP hydrolysis by motors dragged backwards)    
    % R = sum( fmotor_plus  .* gamma_t*a/time_factor + fmotor_minus .* gamma_t*a/time_factor )*ds*L; % [pN um/s] = [aW] dissipation rate
    % --> 0 pW (consistency check: both motor groups give zero)

    % ... include only positive work
    R = sum( max(0, fmotor_plus .* gamma_t*a/time_factor ) + max(0, fmotor_minus .* gamma_t*a/time_factor ) )*ds*L; % [pN um/s] = [aW] dissipation rate
    % --> 226 fW (consistency check: both motor groups give same contribution)
   
    Rlist(it) = R; % keep a record
    
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

fprintf(1,'Simulation completed!\n')

%% Plot instantaneous dissipation rate as function of time
figure(1), clf, hold on
ind = abs(Rlist) < 4e5;
tlist = (0:nt-1)*dt*time_factor; % [s]
plot(tlist(ind),Rlist(ind)*1e-3)
xlabel('t [s]'), ylabel('R [fW]')
ylim([0 4e2])

return

%% Visualize results ------------------------------------------------------
figure(1), clf
% gamma
subplot(3,1,1), hold on
imshow(flipud(gamma_mat'),[],'InitialMagnification','fit')
colorbar
title('\gamma')
% xlim( t_sub*[80 100])
xlim( t_sub*[0 T])
clim([-1 1]*pi/2)
xlabel('t'), ylabel('s')
% n_plus
subplot(3,1,2), hold on
imshow(flipud(nplus_mat'),[],'InitialMagnification','fit')
colorbar
title('n^+')
xlim( t_sub*[0 T])
% xlim( t_sub*[80 100])
clim([0 1]*0.1)
xlabel('t'), ylabel('s')
% n_minus
subplot(3,1,3), hold on
imshow(flipud(nminus_mat'),[],'InitialMagnification','fit')
colormap jet
colorbar
title('n^-')
xlim( t_sub*[0 T])
% xlim( t_sub*[80 100])
clim([0 1]*0.1)
xlabel('t'), ylabel('s')

return

%% Show beat pattern
figure(1), clf, hold on
ind = 1:5:2000; % 1ms apart
plot( cumsum( cos(gamma_mat(ind,:)')*ds*L ), cumsum( sin(gamma_mat(ind,:)')*ds*L ), 'color', [0,0,0,0.1], 'LineWidth', 1 )
ind = 50; % 0 10ms
plot( cumsum( cos(gamma_mat(ind,:))*ds*L ), cumsum( sin(gamma_mat(ind,:))*ds*L ), 'color', [0,0,1,1], 'LineWidth', 2 )
ind = [1 1]; % 0 10ms
plot( cumsum( cos(gamma_mat(ind,:)')*ds*L ), cumsum( sin(gamma_mat(ind,:)')*ds*L ), 'color', [1,0,0,1], 'LineWidth', 2 )
daspect([1 1 1])
% saveas(gcf,'for_figure3.eps','epsc')

%% Power-spectral density =================================================

ind = 1001:2000; % discard transient period at beginning
gamma0 = mean(gamma_mat(ind,:)); 
df = 1/(T*length(ind)/ind(end)); % [Hz] frequency bin size
PSD_freq = (0:length(t(ind))-1) * df; % [Hz] frequencies corresponding to PSD
PSD = mean( abs(fft(gamma_mat(ind,:)-gamma0)/t_sub).^2 / T, 2 ); % phase-averaged power-spectral density; units [rad^2 Hz^-1]

% plot PSD
figure(1), clf
semilogy(PSD_freq,PSD,'b'), hold on
xlim([0 1])
ylim([1e-6 1e1])
xlabel('f [Hz]'), ylabel('PSD [rad^2/Hz]')

%% convert non-dimensional parameters back to dimensional parameters ======
L = 10; % [um] flagellar length
a = 0.2; % [um] sliding length (new preprint by Cass et al. 2024)
K = 2e3; % [pN/um^2], sliding stiffness (Nexin linker); note: value consistent with a^2 K = 80; % [pN/rad] 
B = 840; % [pN um^2] bending resistance (CAUTION: Cass et al. 2023 states wrong unit)
rho = 1e3; % [1/um] (new preprint by Cass et al. 2024)
% mu = a^2*K*L^2/B; % true
f0 = mu_a*B/(a*rho*L^2); % [pN]
fc = f0/fstar; % [pN]
% Note: fc = 0.5 - 2.5; % [pN] (new preprint by Cass et al. 2024) 
% Note: f0 = 1-5; % [pN] (new preprint by Cass et al. 2024) --> discrepancy: f0 = 65.9 pN
pi0 = eta/tau; % [1/s] binding rate
epsilon0 = (1-eta)/tau; % [1/s] basal unbinding rate (unbinding rate = epsilon0*exp(fstar) for gamma_t = 0)
epsilon_b = epsilon0*exp(f0/fc);
tau_b = ( pi0 + epsilon_b )^(-1); % [s] characteristic time-scale of motor relaxation
v0 = a / (zeta*tau); % [um/s] characteristic motor speed 
% Note: v0 = 5-7; % [um/s] (new preprint by Cass et al. 2024) --> discrepancy: v0 = 52.0 um/s
% Note: mu = a^2*K*L^2/B; % true
fprintf(1, 'f0 = %.2f pN, tau=%.2f ms, 1/pi0 = %.2f ms, 1/eps0 = %.2f ms, tau_b = %.2f ms, 1/epsb = %.2f ms, v0 = %.2f um/s, v0*tau_b = %.2f nm\n', f0, tau*1e3, 1e3/pi0, 1e3/epsilon0, 1e3*tau_b, 1e3/epsilon_b, v0, v0*1e3*tau_b )
