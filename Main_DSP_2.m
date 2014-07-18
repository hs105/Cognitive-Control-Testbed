%function [ARMSE_f1,ARMSE1] = Main2(BW);
clear classes;
clc;
close all;

global g rho_0 Qsqrt Delta  M H gamma QPtArray wPtArray nPts;

[QPtArray,wPtArray,nPts] = findSigmaPts(3);

%%%================ Parameter definitions ==================
g     = 9.8;    %gravity in m/s2
fs = 10; %% Sampling rate 10kHz;
fs_ctr = 1; %% running freuqncy of ctr; Defining the update frequency of CTR
T = 1/fs;
Delta = T;
rho_0 = 1.754;  %prop. constant in air density
gamma = 1.49e-4;   %decaying par. in air density

c   = 2.999e8;    % speed of propagation m/s
fc  = 2*pi*10.4e9; % 10.4Ghz X-band radar
tf  = 10e-9;  % rise/fall time: 10 ns;
r_snr0 = 80e3; % range when snr = 0db; 50km

lmd_min = 10e-6; % min of lambda: 10 mu seconds  duration of envelope
lmd_max = 300e-6;% max of lambda: 100 mu seconds
d_lmd   = 10e-6; % grid size for lambda
b_min   = -300e8;   % min of chirp rate: 10,000
b_max   = 300e8;   % max of chirp rate: 500,000
d_b     = 50e8;   % grid size for chirp rate
%delta_f = 15e3;  % frequency sweep
%BW      = Inf;    % bandwidth
BW =    8e6;

Lambda  = diag([1,1,1]); % weight matrix showing signaficence of state [pos, vel, ball_coef]
seed = 2e5; rand('seed',seed);


%%%============= Parameters for model ==================
q1 = 0.01;
q2 = 0.01;
Q = [q1*Delta^3/3 q1*Delta^2/2 0;
    q1*Delta^2/2 q1*Delta     0;
    0            0            q2*Delta];   %Process Noise Covariance
Qsqrt = mysqrt(Q);

nExpt = 2;                 % number of MC runs
t_sim = 5;                 % simulation time: 30s
N = floor(t_sim/Delta);            % number of time steps
M = 30e3;             % horizontal distance of the radar to object (km)
H = 30;                     % radar height (m)

discount = .5; % discount factor in RL
lf = .8;       % Learning factor in RL
epsilon = 0;   % epsilon-greedy (epsilon*100% of times will be random)

%%%================ Generate mesh point for [lmd, b] ========
lmd0 = lmd_min : d_lmd : lmd_max;
b0 = [-b_max : d_b : -b_min, b_min : d_b : b_max];
%%pick up the radar parameters that satisfies the bandwidth constraint
para = pick_theta(lmd0,b0,BW);

%% Fixed waveform selection
wf_fix_idx = round(length(para)*rand/2);
lmd_fixed = para(1,wf_fix_idx);
b_fixed   = para(2,wf_fix_idx);
wf_fix_idx = 8;



MSE     = zeros(nExpt,3,N);
estMSE  = zeros(nExpt,3,N);
MSE_f   = zeros(nExpt,3,N);
estMSE_f= zeros(nExpt,3,N);
MSE_RL   = zeros(nExpt,3,N);
estMSE_RL= zeros(nExpt,3,N);

fprintf('Grid points %d x %d = %d\n',length(lmd0),length(b0),length(lmd0)*length(b0));

%%%================ Initialize CKF ==========================
for expt = 1:nExpt,
    % Q-Learning object with 1 state, 764 actions, alpha=.5, gamma=.5, and
    % epsilon=0 (which means it is completely greedy):
    clear agent;
    agent = QL(1, 1:764, discount, lf, epsilon); 
    
    x   = [61e3; 3048; 19161];   %Initial state
    xkk = [61.5e3; 3000; 19100];%[62e3; 3400; 19100];
    Skk = sqrt(diag([(1e3)^2; (100)^2; 1e4]));
    xkk_f = xkk;
    Skk_f = Skk;
    xkk_RL = xkk;
    Skk_RL = Skk;
    preEntropy = det(Lambda*(Skk_RL*Skk_RL')); % initial entropy
    
    fprintf('Monte Carlo runs: %d out of %d\n',expt, nExpt);
    
    %%%==================== Main loop ===========================
    
    for k = 1 : N,
        %%% learning coef. update
%         agent.alpha = .8;
        
        %%% ========= State and Mst Eqs ========================
        x = StateEq(x)+ Qsqrt*randn(3,1);
        z_idl       = MstEq(x);
        
        %%% ========== Predict ================================
        [xkk1,Skk1]     = Predict(xkk,Skk);
        [xkk1_f,Skk1_f] = Predict(xkk_f,Skk_f);
        [xkk1_RL,Skk1_RL] = Predict(xkk_RL,Skk_RL);
        
        
        %%%============ Subroutine for fixed waveform =====================
        R_f           = computeR_FI(c, fc, lmd_fixed, b_fixed, r_snr0, M, xkk1_f(1)-H);
        Rsqrt_f       = mysqrt(R_f);
        z_f           = z_idl + Rsqrt_f* randn(2,1);
        [xkk_f,Skk_f] = Update(xkk1_f,Skk1_f,z_f,Rsqrt_f);
        
        %%%================================================================
        
        %%% ========== new meas. cov. as per new theta =======
        if (k==1) || (k<30) || (rem(k,1/(fs_ctr*T))==0),
            for i = 1 : length(para),
                theta  = para(:,i)'; % fetch grid point (lmd, b)
                lmd    = theta(1);
                b      = theta(2);
                
                %%% Calculate measurement covariance: R
                R = computeR_FI(c, fc, lmd, b, r_snr0, M, xkk1(1)-H);
                Rsqrt = mysqrt(R);
                %%% Approximate cost function: J(\theta)
                Skk   = ComputeSkk(xkk1,Skk1,Rsqrt);% + p_ratio*Skk_p;
                
                J(i)= trace(Lambda*(Skk*Skk'));
            end
            [min_J,idx] = min(J); % find the minimum of J
            lmd    = para(1,idx);
            b      = para(2,idx);
            wf_lmd(expt, k) = lmd;
            wf_b(expt, k) = b;
            wf_idx(expt,k) = idx;
        end
        %%% Calculate measurement covariance R using new waveform
        R = computeR_FI(c, fc, lmd, b, r_snr0, M, xkk1(1)-H);
        Rsqrt = mysqrt(R);
        mmm = randn(2,1);
        z = z_idl + Rsqrt*mmm;
        
        [xkk,Skk] = Update(xkk1,Skk1,z,Rsqrt);
        %%%================================================================
        
        %% RL:
        Entropy = zeros(1,length(para));        % init. vector to speed up
        reward = zeros(1,length(para));         % init. vector to speed up
        
        if (k==1) || (k<30) || (rem(k,1/(fs_ctr*T))==0),
            for i = 1 : length(para),
                theta  = para(:,i)'; % fetch grid point (lmd, b)
                lmd    = theta(1);
                b      = theta(2);
                
                %%% Calculate measurement covariance: R
                R_RL = computeR_FI(c, fc, lmd, b, r_snr0, M, xkk1_RL(1)-H); %%%
                Rsqrt_RL = mysqrt(R_RL); %%%
                Skk_RL = ComputeSkk(xkk1_RL,Skk1_RL,Rsqrt_RL); %%%
                Entropy(i) = det(Lambda*(Skk_RL*Skk_RL'));
                reward_rel = abs(log(abs(preEntropy - Entropy(i))));
                reward_abs = 1/abs(log(Entropy(i)));
                reward(i) = reward_abs*sign(preEntropy - Entropy(i));
%                 reward(i)
                agent = agent.learning(1,1,i,reward(i)); % there is only one state (1)
            end
            idx = agent.egAction(1); % index of eps-greedy action
            lmd    = para(1,idx);
            b      = para(2,idx);
            wf_lmd(expt, k) = lmd;
            wf_b(expt, k) = b;
            wf_idx(expt,k) = idx;
            preEntropy = Entropy(idx); % updating the previous entropy
            hhh(k) = preEntropy;
        end
        %%% Calculate measurement covariance R using new waveform
        R_RL = computeR_FI(c, fc, lmd, b, r_snr0, M, xkk1_RL(1)-H); %%%
        Rsqrt_RL = mysqrt(R_RL); %%%
        z_RL = z_idl + Rsqrt_RL*mmm; %%%
        [xkk_RL,Skk_RL] = Update(xkk1_RL,Skk1_RL,z_RL,Rsqrt_RL); %%%
        
        %%%================================================================
        
        
        
        
        %%
        xestArray_f(:,k) = xkk_f;
        xArray(:,k) = x;
        xestArray(:,k) = xkk;
        xestArray_RL(:,k) = xkk_RL;
        %    thetaArray(:,k) = theta_new';
        
        MSE(expt,:,k) = (x - xkk).^2;
        MSE_f(expt,:,k) = (x - xkk_f).^2;
        MSE_RL(expt,:,k) = (x - xkk_RL).^2;
        
    end  % time
    
    xArray_temp(:,:,expt)   = xArray;
    xestArray_temp(:,:,expt)= xestArray;
    xestArray_temp_f(:,:,expt)= xestArray_f;
    xestArray_temp_RL(:,:,expt)= xestArray_RL;
    
end % MC run


timewdw = [1:N-2];

%%======== Calculate mean =========================================
xArray      = mean(xArray_temp,3);
xestArray   = mean(xestArray_temp,3);
xestArray_f = mean(xestArray_temp_f,3);
xestArray_RL = mean(xestArray_temp_RL,3);

% MSE = MSE/nExpt;
% RMSE = MSE.^(0.5);
% MSE_f = MSE_f/nExpt;
% RMSE_f = MSE_f.^(0.5);

d_ratio = 0.25;

RMSE1 = reshape(mean(MSE),3,N).^(0.5);
errbar1 = d_ratio*reshape(std(MSE),3,N).^(0.5);

RMSE_f1 =  reshape(mean(MSE_f),3,N).^(0.5);
errbar1_f = d_ratio*reshape(std(MSE_f),3,N).^(0.5);

RMSE_RL1 =  reshape(mean(MSE_RL),3,N).^(0.5);
errbar1_RL = d_ratio*reshape(std(MSE_RL),3,N).^(0.5);

% ARMSE1 = mean(RMSE1');
% ARMSE_f1 = mean(RMSE_f1');

ARMSE1 = mean(mean(MSE,3),1).^0.5;
ARMSE_f1 = mean(mean(MSE_f,3),1).^0.5;
ARMSE_RL1 = mean(mean(MSE_RL,3),1).^0.5;


%%%=================================================================
%%%%%%%%%%%%%%%%%%RMSE plots%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure;
%subplot(221);
plot((1:N)*T,RMSE1(1,:),'b--d','LineWidth',2);
hold on;
plot((1:N)*T,RMSE_f1(1,:),'r*-','LineWidth',2);

plot((1:N)*T,RMSE_RL1(1,:),'go:','LineWidth',2);
% niceBars((1:N)*T,RMSE1(1,:),errbar1(1,:),'b',0.5);
% niceBars((1:N)*T,RMSE_f1(1,:),errbar1_f(1,:),'r',0.5);
% errorbar((1:N)*T,RMSE1(1,:),errbar1(1,:),'b');
% errorbar((1:N)*T,RMSE_f1(1,:),errbar1_f(1,:),'r');
ylabel('RMSE: Alt');
xlabel('Time (sec)');
legend('CTR','Fixed Waveform','Entropic State');
grid on;
%title('Position-error');

figure;
%subplot(222);
plot((1:N)*T,RMSE1(2,:),'b--d','LineWidth',2);
hold on;
plot((1:N)*T,RMSE_f1(2,:),'r*-','LineWidth',2);

plot((1:N)*T,RMSE_RL1(2,:),'g:o','LineWidth',2);

% niceBars((1:N)*T,RMSE1(2,:),errbar1(2,:),'b',0.5);
% niceBars((1:N)*T,RMSE_f1(2,:),errbar1_f(2,:),'r',0.5);
% errorbar((1:N)*T,RMSE1(2,:),errbar1(2,:),'b');
% errorbar((1:N)*T,RMSE_f1(2,:),errbar1_f(2,:),'r');
ylabel('RMSE: Vel');
xlabel('Time (sec)');
legend('CTR','Fixed Waveform','Entropic State');
grid on;

figure;
%subplot(223);
%semilogy((1:N)*T,RMSE1(3,:),'b:d','LineWidth',2);
plot((1:N)*T,RMSE1(3,:),'b--d','LineWidth',2);
hold on;
plot((1:N)*T,RMSE_f1(3,:),'r*-','LineWidth',2);
plot((1:N)*T,RMSE_RL1(3,:),'g:o','LineWidth',2);
% niceBars((1:N)*T,RMSE1(3,:),errbar1(3,:),'b',0.5);
% niceBars((1:N)*T,RMSE_f1(3,:),errbar1_f(3,:),'r',0.5);
% errorbar((1:N)*T,RMSE1(3,:),errbar1(3,:),'b');
% errorbar((1:N)*T,RMSE_f1(3,:),errbar1_f(3,:),'r');
%set(gca, 'YScale', 'log');
ylabel('RMSE: B.Coef.');
xlabel('Time (sec)');
legend('CTR','Fixed Waveform','Entropic State');
grid on;

% %suplabel(['Bandwidth = ',num2str(BW/1e6), 'MHz']);
%
%    %[ax,h1]=suplabel('super X label');
%    %[ax,h2]=suplabel('super Y label','y');
%    [ax,h3]=suplabel(['Bandwidth = ',num2str(BW/1e6), 'MHz'] ,'t');
%    set(h3,'FontSize',15) ;

%figure;
%subplot(224);
% [AX,H1,H2] = plotyy((1:N)*T,mean(wf_b),(1:N)*T,mean(wf_lmd));
% hold on;
% set(H1,'LineStyle',':','LineWidth',2);
% set(H2,'LineStyle','-','LineWidth',2);
% xlabel('Time (s)');
% ylabel(AX(1),'Chirp rate');
% ylabel(AX(2),'Envelope len.');
% grid on;
%

figure;
% plot((1:N)*T,mean(wf_idx),'b-','LineWidth',2);
% xlabel('Time (s)');
% ylabel('Waveform index');
% grid on;

plot((1:N)*T,mean(wf_b),'b-','LineWidth',2);
xlabel('Time (s)');
ylabel('Chirp rate');
grid on;


figure;
plot((1:N)*T,mean(wf_lmd),'b-','LineWidth',2);
xlabel('Time (s)');
ylabel('Length of envelope');
grid on;

% %plotting
%
% figure;
% plot(xArray(1,:),'r');
% hold on;
% plot(xestArray(1,:),'b:');
% plot(xestArray_f(1,:),'k*-');
% xlabel('time');
% ylabel('altutude');
% legend('True','Filtered','Fixed Waveform');
% hold off;
%
% figure;
% plot(xArray(2,:),'r');
% hold on;
% plot(xestArray(2,:),'b:');
% plot(xestArray_f(2,:),'k*-');
% xlabel('time');
% ylabel('velocity');
% legend('True','Filtered','Fixed Waveform');
% hold off;
%
% figure;
% plot(xArray(3,:),'r');
% hold on;
% plot(xestArray(3,:),'b:');
% plot(xestArray_f(3,:),'k*-');
% xlabel('time');
% ylabel('ballistic coefficient');
% legend('True','Filtered','Fixed Waveform');
% hold off;


