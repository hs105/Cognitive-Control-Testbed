function R = computeR_FI(c, fc, lmd, b, r_snr0, range1, range2)

%         %%% Calculate measurement covariance: R
%         Gamma = diag([c/2,c/(2*fc)]);
%         I11   = 1/(2*(2*pi)^2*lmd^2) + 2*b^2*lmd^2;
%         I12   = 2*b*lmd^2;
%         I22   = lmd^2/2;
%         r_mes = sqrt(range1^2 + range2^2); % predicted range
%         snr   = (r_snr0/r_mes)^4;               % calculate snr          
%         I     = (2*pi)^2*snr*[I11, I12; I12, I22];
%         R     = Gamma*pinv(I)*Gamma';   % warning may appear for singular I?? Eq.(7)


        %%% Calculate measurement covariance: R
        r_mes = sqrt(range1^2 + range2^2); % predicted range
        snr   = (r_snr0/r_mes)^4;               % calculate snr          
      
        R11   = lmd^2/2;
        R12   = -b*lmd^2/(fc);
        R22   = 1/fc^2 * (1/(2*lmd^2) + 2*b^2*lmd^2);
        con   = c^2/snr;
        R     = con*[R11 R12; R12 R22];