
function Skk = ComputeSkk(xkk1,Skk1, Rsqrt)

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

Skk = S(nz+1:end,nz+1:end);

    

