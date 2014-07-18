function output = randWindowGenerator(N, center, redius, upperBound)
%
% output = randWindowGenerator(N, center, redius, upperBound)
% 
% This function generates N interger random numbers with uniform dist.
% between center-redius and center+redius, including both endpoints.
% If the window [center-redius,center+redius] exits [1,upperBound], then
% at the side of exit the bound is considered insted. 
%
% Powered by: Mehdi Fatemi
% CSL, McMaster University, April 17, 2012 
%

% output = zeros(N,1);
% for i = 1:N
%     temp = fix(rand*upperBound + 1);
%     while sum(output == temp)
%         temp = fix(rand*upperBound + 1);
%     end
%     output(i) = temp;
% end

% Finding the upper and lower bounds of integer randoms
if ( center-redius>=1 && (center+redius)<=upperBound )
    a = center - redius;
    b = center + redius;
elseif ( center-redius<1 && (center+redius)<=upperBound )
    a = 1;
    b = center + redius;
elseif ( center-redius>=1 && (center+redius)>upperBound )
    a = center - redius;
    b = upperBound;
else
    a = 1;
    b = upperBound;
end

% Computing random in the proper interval
alpha = b-a+1;
beta = a;
output = fix( alpha*rand(N,1) + beta );
