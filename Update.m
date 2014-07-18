
function [xkk,Skk] = Update(xkk1,Skk1,z, Rsqrt)

global nz QPtArray  nPts;

Xi =  repmat(xkk1,1,nPts) + Skk1*QPtArray;
    
Zi = MstEq(Xi);
    
zkk1 = sum(Zi,2)/nPts; 

X = (Xi-repmat(xkk1,1,nPts))/sqrt(nPts);
    
Z = (Zi-repmat(zkk1,1,nPts))/sqrt(nPts);  

nx = 3;

nz = 2;


[foo,S] = qr([Z Rsqrt; X zeros(nx,nz)]',0);

S = S';

Z = S(1:nz,1:nz);

Y = S(nz+1:end,1:nz);

X = S(nz+1:end,nz+1:end);

G = Y/Z;

Skk = X;

xkk = xkk1 + G*(z-zkk1);  
    

