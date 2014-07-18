

function [xPts,wPts,nPts]= findSigmaPts(n)

nPts = 2*n;
wPts = ones(1, nPts)/(2*n);
xPts = sqrt(n)*eye(n);
xPts = [xPts -xPts];


