function Msqrt = mysqrt(M)

[s v d] = svd(M);

Msqrt = s*sqrt(v);

%Msqrt = chol(M)';