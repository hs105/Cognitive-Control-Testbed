function [xkk1,Skk1] = Predict(xkk,Skk)

global Qsqrt  QPtArray nPts;

Xi = repmat(xkk,1,nPts) + Skk*QPtArray;

Xi = StateEq(Xi);

xkk1 = sum(Xi,2)/nPts; 

X = (Xi-repmat(xkk1,1,nPts))/sqrt(nPts);

[foo,Skk1] = qr([ X Qsqrt]',0);

Skk1 = Skk1'; 


