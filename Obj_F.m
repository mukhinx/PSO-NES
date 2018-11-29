function[costFunction, gradF, cost2]= Obj_F(sig, bestSolution, MTZ_new_1D, bestFunction,rr_true, sqT)

%% NES Parameters
d = length(sig); % dimension of target vector
A = sqrtm(sig);
Cov = sig;
% logCov = log(Cov); % log Covariance for objective function
detCov = det(Cov);
eps = 1.1; % difference coef
n = round(4+3*log(d)); %number of samples fron MC
%% Calculating CostFunction
xbest = bestSolution.xbest';
a =  max(Cov(:)); % max function multiplier
% a = 30;
fit = zeros(1,n);
Z = randn(d, n);% 
X = repmat(xbest,1,n);
U = A*Z;
X = X + U;

for i = 1 : n
    [fit(i), ~] = MTZ_new_1D(X(:,i), rr_true, sqT);
end

MeanFit = mean(fit);
first = sqrt(detCov);
% second = max(0, (MeanFit-eps*bestFunction)^2);
second = exp(a*(MeanFit-eps*bestFunction)^2);
costFunction = first - second; %objective function
cost2 = second;
%% Calculating gradient
dV = 0.5 * first * inv(Cov)';
% dP = 2*a*max(0, MeanFit - eps*bestFunction); %dP/dE, where P is the Penaulty function
dP = 2 * a * second * (MeanFit-eps*bestFunction);
M = dV;

dE = zeros(d);
for i = 1 : n
    dE = dE + fit(i) * (0.5 * M * ( X(:,i) - xbest )*( X(:,i) - xbest)' * M - 0.5 * M);
end

dE = dE / n;

gradF = dV + dP*dE;

end
