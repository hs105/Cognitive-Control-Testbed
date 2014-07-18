%function [ARMSE_f1,ARMSE1] = Main2(BW);
clear classes;
clc;

global g rho_0 Qsqrt Delta  M H gamma QPtArray wPtArray nPts;

[QPtArray,wPtArray,nPts] = findSigmaPts(3);

%%%================ Parameter definitions ==================
g     = 9.8; % gravity in m/s2
fs = 10;     % Sampling rate 10kHz;
fs_ctr = 1;  % running freuqncy of ctr; Defining the update frequency of CTR
T = 1/fs;
Delta = T;
rho_0 = 1.754;     %prop. constant in air density
gamma = 1.49e-4;   %decaying par. in air density

c   = 2.999e8;     % speed of propagation m/s
fc  = 2*pi*10.4e9; % 10.4Ghz X-band radar
tf  = 10e-9;       % rise/fall time: 10 ns;
r_snr0 = 80e3;     % range when snr = 0db; 50km

lmd_min = 10e-6;   % min of lambda: 10 mu seconds  duration of envelope
lmd_max = 300e-6;  % max of lambda: 100 mu seconds
d_lmd   = 10e-6;   % grid size for lambda
b_min   = -300e8;  % min of chirp rate: 10,000
b_max   = 300e8;   % max of chirp rate: 500,000
d_b     = 50e8;    % grid size for chirp rate
%delta_f = 15e3;   % frequency sweep
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

nExpt = 1200;            % number of MC runs
t_sim = 25;              % simulation time: 30s
N = floor(t_sim/Delta); % number of time steps
M = 30e3;               % horizontal distance of the radar to object (km)
H = 30;                 % radar height (m)

discount = .5;   % discount factor in RL
gamma_back = 1; % backward discount factor
lf = 1;        % Learning factor in RL
lfRL2 = lf;
lfRL3 = lf; 
epsilon = 0;     % epsilon-greedy (epsilon*100% of times will be random)
epsilonRL2 = epsilon;    % epsilon of agent2
epsilonRL3 = epsilon;    % epsilon of agent3
window = 10;
numPlanningTrials = 1;  % must NOT be greater than number of actions

%%%================ Generate mesh point for [lmd, b] ========
lmd0 = lmd_min : d_lmd : lmd_max;
b0 = [b_min : d_b : b_max];
%%pick up the radar parameters that satisfies the bandwidth constraint
para = pick_theta(lmd0,b0,BW);

% Loading executive memory
memory = load('memory.mat');     % loading the given memory
[sorted_values, sorted_idx] = sort(memory.actionBank);
actionSet = sorted_idx(end:-1:end-39); % indeces of 40 most probable actions in memory

% actionSet = ceil(382*rand(10,1));

para = para(:,actionSet);

MSE     = zeros(nExpt,3,N);
estMSE  = zeros(nExpt,3,N);
MSE_f   = zeros(3,N);
estMSE_f= zeros(nExpt,3,N);
MSE_RL2   = zeros(3,N);
estMSE_RL= zeros(nExpt,3,N);

fprintf('Grid points %d x %d = %d\n',length(lmd0),length(b0),length(lmd0)*length(b0));

%%%================ Initialize CKF ==========================

time_old = zeros(nExpt,N);
time_new = zeros(nExpt,N); % agent1
time_new2 = zeros(nExpt,N); % agent2
time_new3 = zeros(nExpt,N); % agent3

entropyBank = zeros(nExpt, N);   % placeholder for entropies for RL method (agent1)
entropyBank2 = zeros(nExpt, N);  % placeholder for entropies for RL method (agent2)
entropyBank3 = zeros(nExpt, N);  % placeholder for entropies for RL method (agent2)
entropyBankf = zeros(nExpt, N);  % placeholder for entropies for fixed-waveform
stateBank = zeros(3, N+1);

distubance_cycles = 0:74;
disturbance_multiplier = 6*exp(-((distubance_cycles-37).^2)./600); % Normally distributed

for expt = 1:nExpt
    
    %% Fixed waveform selection
    wf_fix_idx = fix(rand*length(para) + 1);
    lmd_fixed = para(1,wf_fix_idx);
    b_fixed   = para(2,wf_fix_idx);
    
    
    % Q-Learning object with 1 state, 764 actions, alpha=.5, gamma=.5, and
    % epsilon=0 (which means it is completely greedy):
    clear agent;
    clear agent2;
    clear agent3;
    
    agent = QL(1, 1:382, discount, lf, epsilon);
    agent2 = QL(1, 1:40, discount, lfRL2, epsilonRL2);
    agent3 = QL(1, 1:382, discount, lfRL3, epsilonRL3);
    
    % Injecting memory to agent2
%       agent2.Q = memory.actionBank;
%     agent2.Q = rand(1, 382);
%     agent2.Q = 100*memory.actionBank'./sum(memory.actionBank);
    
    x   = [61e3; 3048; 19161];   %Initial state
    stateBank(:,1) = x;
    xkk = [61.5e3; 3000; 19100]; %[62e3; 3400; 19100];
    Skk = sqrt(diag([(1e3)^2; (100)^2; 1e4]));
    xkk_f = xkk;
    Skk_f = Skk;
    xkk_RL = xkk;
    Skk_RL = Skk;
    preEntropy = det(Lambda*(Skk_RL*Skk_RL')); % initial entropy
    
    xkk_RL2 = xkk;
    Skk_RL2 = Skk;
    preEntropy2 = det(Lambda*(Skk_RL2*Skk_RL2')); % initial entropy
    
    xkk_RL3 = xkk;
    Skk_RL3 = Skk;
    preEntropy3 = det(Lambda*(Skk_RL3*Skk_RL2')); % initial entropy
    
    fprintf('Monte Carlo runs: %d out of %d\n',expt, nExpt);
    
    %%%==================== Main loop ===========================
    
    for k = 1 : N,
        %%% learning coef. update
        %         agent.alpha = .8;
        
        if k>=75 && k<75+length(disturbance_multiplier)
            Qsqrt1 = Qsqrt + disturbance_multiplier(k-74)* Qsqrt;
        else
            Qsqrt1 = Qsqrt;
        end
        
        %%% ========= State and Mst Eqs ========================
        x = StateEq(x)+ Qsqrt1*randn(3,1);
        stateBank(:,k+1) = x;
        z_idl = MstEq(x);
        
        %%% ========== Predict ================================
        [xkk1,Skk1]     = Predict(xkk,Skk);
        [xkk1_f,Skk1_f] = Predict(xkk_f,Skk_f);
        [xkk1_RL,Skk1_RL] = Predict(xkk_RL,Skk_RL);
        [xkk1_RL2,Skk1_RL2] = Predict(xkk_RL2,Skk_RL2);
        [xkk1_RL3,Skk1_RL3] = Predict(xkk_RL3,Skk_RL3);
        
        
        %%%============ Subroutine for fixed waveform =====================
        R_f           = computeR_FI(c, fc, lmd_fixed, b_fixed, r_snr0, M, xkk1_f(1)-H);
        Rsqrt_f       = mysqrt(R_f);
        z_f           = z_idl + Rsqrt_f* randn(2,1);
        [xkk_f,Skk_f] = Update(xkk1_f,Skk1_f,z_f,Rsqrt_f);
        fEntropy = det(Lambda*(Skk_f*Skk_f'));
        entropyBankf(expt,k) = fEntropy;
        MSE_f(:,k) = MSE_f(:,k) + (x - xkk_f).^2;  
        
        %%%================================================================
        
        %%% ========== new meas. cov. as per new theta =======
%         tic
%         if (k==1) || (k<30) || (rem(k,1/(fs_ctr*T))==0),
%             for i = 1 : length(para),
%                 theta  = para(:,i)'; % fetch grid point (lmd, b)
%                 lmd    = theta(1);
%                 b      = theta(2);
%                 
%                 %%% Calculate measurement covariance: R
%                 R = computeR_FI(c, fc, lmd, b, r_snr0, M, xkk1(1)-H);
%                 Rsqrt = mysqrt(R);
%                 %%% Approximate cost function: J(\theta)
%                 Skk   = ComputeSkk(xkk1,Skk1,Rsqrt);% + p_ratio*Skk_p;
%                 
%                 J(i)= trace(Lambda*(Skk*Skk'));
%             end
%             [min_J,idx] = min(J); % find the minimum of J
%             lmd    = para(1,idx);
%             b      = para(2,idx);
%             wf_lmd(expt, k) = lmd;
%             wf_b(expt, k) = b;
%             wf_idx(expt,k) = idx;
%         end
%         %%% Calculate measurement covariance R using new waveform
%         R = computeR_FI(c, fc, lmd, b, r_snr0, M, xkk1(1)-H);
%         Rsqrt = mysqrt(R);
        mmm = randn(2,1);
%         z = z_idl + Rsqrt*mmm;
%         
%         [xkk,Skk] = Update(xkk1,Skk1,z,Rsqrt);
%         entropyDyOpBank(expt,k) = det(Lambda*(Skk_RL*Skk_RL'));
%         time_old(expt,k) = toc;
        %%%================================================================
        
        %% RL:
%         predictedEntropy = zeros(1,length(para)); % init. vector to speed up
%         predictedReward = zeros(1,length(para));  % init. vector to speed up
%         reward = zeros(1,N);
%         tic
%         if (k==1) || (k<30) || (rem(k,1/(fs_ctr*T))==0),
%             %%%%% Planning %%%%%
             tester = randGenerator(numPlanningTrials, length(para));
%             for count = 1 : length(tester) %length(para),
%                 i = tester(count);
%                 theta  = para(:,i)'; % fetch grid point (lmd, b)
%                 lmd    = theta(1);
%                 b      = theta(2);
%                 
%                 %%% Calculate measurement covariance: R
%                 R_RL = computeR_FI(c, fc, lmd, b, r_snr0, M, xkk1_RL(1)-H); %%%
%                 Rsqrt_RL = mysqrt(R_RL); %%%
%                 Skk_RL = ComputeSkk(xkk1_RL,Skk1_RL,Rsqrt_RL); %%%
%                 predictedEntropy(count) = det(Lambda*(Skk_RL*Skk_RL'));
%                 reward_rel = abs(log(abs(preEntropy - predictedEntropy(count))));
% %                 reward_abs = 1/abs(log(predicted Entropy(count)));
%                 predictedReward(i) = reward_rel*sign(preEntropy - predictedEntropy(count));
%                 % learing from prediction
%                 agent = agent.learning(1,1,i,predictedReward(i)); % there is only one state (1)
%             end
%             
%             %%%%% Learning %%%%%
%             % trigering epsilon-greedy action
%             idx = agent.egAction(1); % index of eps-greedy action
%             if idx == 0 % no learning yet
%                 idx = fix(rand*length(para) + 1); % random selection
%             end
%             lmd    = para(1,idx);
%             b      = para(2,idx);
%             wf_lmd(expt, k) = lmd;
%             wf_b(expt, k) = b;
%             wf_idx(expt,k) = idx;
%             
%             % computing R as the result of the applied action
%             R_RL = computeR_FI(c, fc, lmd, b, r_snr0, M, xkk1_RL(1)-H); %%%
%             Rsqrt_RL = mysqrt(R_RL); %%%
%         end
%         
%         z_RL = z_idl + Rsqrt_RL*mmm; %%%
%         [xkk_RL,Skk_RL] = Update(xkk1_RL,Skk1_RL,z_RL,Rsqrt_RL); %%%
%         
%         % computing the new entropic state
%         newEntropy = det(Lambda*(Skk_RL*Skk_RL'));
%         
%         % computing the reward
%         reward_rel = abs(log(abs(preEntropy - newEntropy)));
%         reward_abs = 1/abs(log(newEntropy));
%         reward(k) = reward_rel*sign(preEntropy - newEntropy);
%         %                 reward(i)
%         
%         % learning by Q-learning method
%         agent = agent.learning(1,1,idx,reward(k)); % there is only one state (1)
%         
%         % updating the previous entropy
%         preEntropy = newEntropy;
%         time_new(expt,k) = toc;
%         entropyBank(expt,k) = preEntropy;
        %%%================================================================
        
        %% RL2:
        predictedEntropy2 = zeros(1,length(para)); % init. vector to speed up
        predictedReward2 = zeros(1,length(para));  % init. vector to speed up
        reward2 = zeros(1,N);
        tic
        if (k==1) || (k<30) || (rem(k,1/(fs_ctr*T))==0),
            %%%%% Planning %%%%%
%             tester = randGenerator(numPlanningTrials, length(para));
            for count = 1 : length(tester)%length(para),
                i = tester(count);
                theta  = para(:,i)'; % fetch grid point (lmd, b)
                lmd    = theta(1);
                b      = theta(2);
                
                %%% Calculate measurement covariance: R
                R_RL2 = computeR_FI(c, fc, lmd, b, r_snr0, M, xkk1_RL2(1)-H); %%%
                Rsqrt_RL2 = mysqrt(R_RL2); %%%
                Skk_RL2 = ComputeSkk(xkk1_RL2,Skk1_RL2,Rsqrt_RL2); %%%
                predictedEntropy2(count) = det(Lambda*(Skk_RL2*Skk_RL2'));
                reward_rel2 = abs(log(abs(preEntropy2 - predictedEntropy2(count))));
%                 reward_abs = 1/abs(log(predicted Entropy(count)));
                predictedReward2(i) = reward_rel2*sign(preEntropy2 - predictedEntropy2(count));
                % learing from prediction
%                 agent2.histAct = memory.actionBank./sum(memory.actionBank);      % updating the probability distribution
                agent2 = agent2.dpLearning(1,1,i,predictedReward2(i)); % there is only one state (1)
            end
            
            %%%%% Learning %%%%%
            % trigering epsilon-greedy action
            idx = agent2.egAction(1); % index of eps-greedy action
            if idx == 0 % no learning yet
                idx = fix(rand*length(para) + 1); % random selection
            end
            
            lmd    = para(1,idx);
            b      = para(2,idx);
            wf_lmd(expt, k) = lmd;
            wf_b(expt, k) = b;
            wf_idx(expt,k) = idx;
            
            % computing R as the result of the applied action
            R_RL2 = computeR_FI(c, fc, lmd, b, r_snr0, M, xkk1_RL2(1)-H); %%%
            Rsqrt_RL2 = mysqrt(R_RL2); %%%
        end
        
        z_RL2 = z_idl + Rsqrt_RL2*mmm; %%%
        [xkk_RL2,Skk_RL2] = Update(xkk1_RL2,Skk1_RL2,z_RL2,Rsqrt_RL2); %%%
        
        % computing the new entropic state
        newEntropy2 = det(Lambda*(Skk_RL2*Skk_RL2'));
        
        % computing the reward
        reward_rel2 = abs(log(abs(preEntropy2 - newEntropy2)));
        reward_abs2 = 1/abs(log(newEntropy2));
        reward2(k) = reward_rel2*sign(preEntropy2 - newEntropy2);
        %                 reward(i)
        
        % learning by Q-learning method
%         agent2.histAct = memory.actionBank./sum(memory.actionBank); % updating the probability distribution
        agent2 = agent2.dpLearning(1,1,idx,reward2(k));         % there is only one state (1)
        
        % updating the previous entropy
        preEntropy2 = newEntropy2;
        time_new2(expt,k) = toc;
        entropyBank2(expt,k) = preEntropy2;
        MSE_RL2(:,k) = MSE_RL2(:,k) + (x - xkk_RL2).^2;
        %%%================================================================
        
        
        %% RL3:
%         predictedEntropy3 = zeros(1,length(para)); % init. vector to speed up
%         predictedReward3 = zeros(1,length(para));  % init. vector to speed up
%         reward3 = zeros(1,N);
%         tic
%         if (k==1) || (k<30) || (rem(k,1/(fs_ctr*T))==0),
%             %%%%% Planning %%%%%
% %             tester = randWindowGenerator(numPlanningTrials, agent3.egAction(1), window, length(para));
%             for count = 1 : length(tester)%length(para),
%                 i = tester(count);
%                 theta  = para(:,i)'; % fetch grid point (lmd, b)
%                 lmd    = theta(1);
%                 b      = theta(2);
%                 
%                 %%% Calculate measurement covariance: R
%                 R_RL3 = computeR_FI(c, fc, lmd, b, r_snr0, M, xkk1_RL3(1)-H); %%%
%                 Rsqrt_RL3 = mysqrt(R_RL3); %%%
%                 Skk_RL3 = ComputeSkk(xkk1_RL3,Skk1_RL3,Rsqrt_RL3); %%%
%                 predictedEntropy3(count) = det(Lambda*(Skk_RL3*Skk_RL3'));
%                 reward_rel3 = abs(log(abs(preEntropy3 - predictedEntropy3(count))));
%                 predictedReward3(i) = reward_rel3*sign(preEntropy3 - predictedEntropy3(count));
%                 % learing from prediction
%                 agent3 = agent3.dpLearning(1,1,i,predictedReward3(i)); % there is only one state (1)
%             end
%             
%             %%%%% Learning %%%%%
%             % trigering epsilon-greedy action
%             idx = agent3.egAction(1); % index of eps-greedy action
%             if idx == 0 % no learning yet
%                 idx = fix(rand*length(para) + 1); % random selection
%             end
%             lmd    = para(1,idx);
%             b      = para(2,idx);
%             wf_lmd(expt, k) = lmd;
%             wf_b(expt, k) = b;
%             wf_idx(expt,k) = idx;
%             
%             % computing R as the result of the applied action
%             R_RL3 = computeR_FI(c, fc, lmd, b, r_snr0, M, xkk1_RL3(1)-H); %%%
%             Rsqrt_RL3 = mysqrt(R_RL3); %%%
%         end
%         
%         z_RL3 = z_idl + Rsqrt_RL3*mmm; %%%
%         [xkk_RL3,Skk_RL3] = Update(xkk1_RL3,Skk1_RL3,z_RL3,Rsqrt_RL3); %%%
%         
%         % computing the new entropic state
%         newEntropy3 = det(Lambda*(Skk_RL3*Skk_RL3'));
%         
%         % computing the reward
%         reward_rel3 = abs(log(abs(preEntropy3 - newEntropy3)));
%         reward_abs3 = 1/abs(log(newEntropy3));
%         reward3(k) = reward_rel3*sign(preEntropy3 - newEntropy3);
%         %                 reward(i)
%         
%         % learning by Q-learning method
%         agent3 = agent3.dpLearning(1,1,idx,reward3(k)); % there is only one state (1)
%         
%         % updating the previous entropy
%         preEntropy3 = newEntropy3;
%         time_new3(expt,k) = toc;
%         entropyBank3(expt,k) = preEntropy3;
        %%%================================================================        
        
        
    end  % time
    
end % MC run

%MonteCarlo_J = MonteCarlo_J/nExpt;
% save('memory.mat','MonteCarlo_J')

% timewdw = [1:N-2];

% %%======== Calculate mean =========================================
% xArray      = mean(xArray_temp,3);
% xestArray   = mean(xestArray_temp,3);
% xestArray_f = mean(xestArray_temp_f,3);
% xestArray_RL = mean(xestArray_temp_RL,3);
% 
% % MSE = MSE/nExpt;
% % RMSE = MSE.^(0.5);
% % MSE_f = MSE_f/nExpt;
% % RMSE_f = MSE_f.^(0.5);
% 
% d_ratio = 0.25;
% 
% RMSE1 = reshape(mean(MSE),3,N).^(0.5);
% errbar1 = d_ratio*reshape(std(MSE),3,N).^(0.5);
% 
% RMSE_f1 =  reshape(mean(MSE_f),3,N).^(0.5);
% errbar1_f = d_ratio*reshape(std(MSE_f),3,N).^(0.5);
% 
% RMSE_RL1 =  reshape(mean(MSE_RL),3,N).^(0.5);
% errbar1_RL = d_ratio*reshape(std(MSE_RL),3,N).^(0.5);
% 
% % ARMSE1 = mean(RMSE1');
% % ARMSE_f1 = mean(RMSE_f1');
% 
% ARMSE1 = mean(mean(MSE,3),1).^0.5;
% ARMSE_f1 = mean(mean(MSE_f,3),1).^0.5;
% ARMSE_RL1 = mean(mean(MSE_RL,3),1).^0.5;


%%%=================================================================
%%%%%%%%%%%%%%%%%%RMSE plots%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% meanEntropy = mean(entropyBank);
% semilogy(1:N,meanEntropy,'g-d','linewidth',1)
% 
% hold on
% 
% meanEntropy2 = mean(entropyBank2);
% semilogy(1:N,meanEntropy2,'b:','linewidth',2)
% hold on
% % meanEntropy3 = mean(entropyBank3);
% % semilogy(1:N,meanEntropy3,'b-o','linewidth',1)
% 
% % meanEntropyDyOpBank = mean(entropyDyOpBank);
% % semilogy(1:N,meanEntropyDyOpBank,'b-*','linewidth',1)
% % hold on
% 
% meanEntropyf = mean(entropyBankf);
% semilogy(1:N,meanEntropyf,'g','linewidth',2)
% 
% xlabel('Perception-action cycles');
% ylabel('Entropic-state');
% hold off

% figure(2)
semilogy(sqrt(MSE_RL2(2,:)/nExpt)', ':r', 'linewidth', 2)
hold on
semilogy(sqrt(MSE_f(2,:)/nExpt)', ':g', 'linewidth', 2)

plot([75, 75], [.1 100], '--k', 'linewidth', 1)
plot([150, 150], [.1 100], '--k', 'linewidth', 1)
hold off

xlabel('Global Perception-action Cycles')
ylabel('RMSE of x_{2} (m/s)')

set(gcf, 'Position', [0 0 700 700])
set(findall(gca,'type','text'),'fontSize',20)
a = get(gcf, 'CurrentAxes');
set(a,'FontSize',20)

