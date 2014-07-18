function Xout = StateEq(X)

global Delta gamma rho_0 g;

n = size(X,2);

rho = rho_0*exp(-gamma*X(1,:));

D = (g*rho.*(X(2,:)).^2)./(2.*X(3,:));

Phi = [1  -Delta  0; 0   1  0; 0 0 1];

G = [0 Delta 0]';

Xout = Phi*X - G*(D - repmat(g,1,n));


