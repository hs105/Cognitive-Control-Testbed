function hOut = MstEq(x)

global M H;

r = sqrt(M^2 +(x(1,:)-H).^2);

rdot = ( x(2,:).*(x(1,:)-H) )./r;

hOut = [r; rdot];

