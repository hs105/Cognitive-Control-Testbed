function output = randGenerator(N, upperBound)
%
% output = randGenerator(N, upperBound)
% 
% This function generates N interger random numbers with uniform dist.
% between 1 and upperBound, including both 1 and upperBound'.
%
% Powered by: Mehdi Fatemi
% CSL, McMaster University, August 17, 2011 
%

if N > upperBound % impossible situation
    error('The first argument must NOT be greater than the second one.');
end

output = zeros(N,1);
for i = 1:N
    temp = fix(rand*upperBound + 1);
    while sum(output == temp)
        temp = fix(rand*upperBound + 1);
    end
    output(i) = temp;
end