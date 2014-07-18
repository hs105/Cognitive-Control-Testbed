function R = ComputeR(xkk,Skk)

global xxx;




for i = 1: length(lmd0),
        for j = 1: length(b0),
            theta  = [lmd0(i),b0(j)]; % fetch grid point (lmd, b)
            lmd    = theta(1);
            b      = theta(2);

            %%% Calculate measurement covariance: R
            Gamma = diag([c/2,c/(2*fc)]);
            I11 = 1/(2*(2*pi)^2*lmd^2) + 2*b^2*lmd^2;
            I12 = 2*b*lmd^2;
            I22 = lmd^2/2;
            I   = (2*pi)^2*snr*[I11, I12; I12, I22];

            R   = Gamma*inv(I)*Gamma';   % warning may appear for singular I?? Eq.(7)

            %%% Approximate cost function: J(\theta)
%             z_p = mvg(z_k,P_zz,N_z);
%             
%             for m = 1: N_z;
%                 %% Calculate P_kk
%                 
%                  [xkk,Skk] = Update(xkk1,Skk1,z_p,R);
% 
%                 
%                  J_temp(m)= trace(Lambda*Skk*Skk');
%                  
%             end
            
            J(i,j)= trace(Lambda*Skk*Skk');
            
            J(i,j) = mean(J_temp); % store the cost for each grid point
            [ind_x,ind_y] = find(J == min(min(J))); % find the minimum of J

        end
    end
    
    theta_new = [lmd0(ind_x),b0(ind_y)]; % using theta_new as the new waveform
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Compute R according to theta_new again
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
