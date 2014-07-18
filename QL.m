classdef QL
    %%
    properties
        S       % state set (vector of all the states):
        A       % action set (vector of all the actions):
        pol     % greedy policy vector (= pi(s))
        gamma   % discount factor
        alpha   % learning factor
        Q       % action-value set (matrix of Q(s,a))
        epsilon % eps-Greedy factor, \in [0,1]
        histAct % probability distribution of the action-space
    end
    %%
    methods
        
        function obj = QL(S, A, gamma, alpha, epsilon)
            %
            % class constructor
            %
            obj.S = S;
            obj.A = A;
            obj.pol = zeros(length(obj.S),1);
            obj.histAct = (1/length(obj.A))*ones(length(obj.A),1); % uniform distribution by default
            obj.Q = zeros(length(obj.S),length(obj.A));
            obj.gamma = gamma;
            obj.alpha = alpha;
            obj.epsilon = epsilon;
        end
        
        function obj = learning(obj, s, s_next, a, R)
            %
            % This method implements the Q-Learning algorithm
            % s and s_next are "index" of current and next states
            %
            delta = R + obj.gamma * max(obj.Q(s_next,:)) - obj.Q(s,a);
            obj.Q(s,a) = obj.Q(s,a) + obj.alpha*delta; % updating Q
            [~,obj.pol(s)] = max(obj.Q(s,:));          % updating greedy policy
        end
        
        function obj = dpLearning(obj, s, s_next, a, R)
            %
            % This method implements the CC learning algorithm
            % s and s_next are "index" of current and next states
            %
            delta = R + obj.gamma * mean(obj.Q(s_next,:)) - obj.Q(s,a);
            obj.Q(s,a) = obj.Q(s,a) + obj.alpha*delta; % updating Q
            [~,obj.pol(s)] = max(obj.Q(s,:));          % updating greedy policy
        end
        
        function obj = dp(obj, s, s_next, a, R)
            %
            % This method implements the DP algorithm
            % s and s_next are "index" of current and next states
            %
            
            delta = R + obj.gamma * (obj.Q(s_next,:)*obj.histAct) - obj.Q(s,a);
            obj.Q(s,a) = obj.Q(s,a) + obj.alpha*delta; % updating Q
            [~,obj.pol(s)] = max(obj.Q(s,:));          % updating greedy policy
        end
         
        function a = egAction(obj, s)
            %
            % This method implements the epsilon-greedy policy
            %
            r = rand;
            if r < obj.epsilon
                a = fix(length(obj.A)*rand) + 1; % selecting a random state, regardless of greedy policy
            else
                a = obj.pol(s); % greedy selection of state
            end
        end
        
        %% Setter functions:
        function obj = set.alpha(obj, value)
            if ~(value >= 0)
                error('alpha must be nonnegetive')
            else
                obj.alpha = value;
            end
        end
        
        function obj = set.gamma(obj, value)
            if ~(value >= 0 && value <= 1)
                error('gamma must be in [0,1]')
            else
                obj.gamma = value;
            end
        end
        
        function obj = set.epsilon(obj, value)
            if ~(value >= 0 && value <= 1)
                error('epsilon must be in [0,1]')
            else
                obj.epsilon = value;
            end
        end
    end
    
end