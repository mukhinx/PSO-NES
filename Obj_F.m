function[costFunction, gradF]= Obj_F(sig, bestSolution, MTZ_new_1D, bestFunction)

%% NES Parameters
d = length(sig); % dimension of target vector
sig = diag(diag(sig));
A = sqrtm(sig);
% Cov = diag(sig.^2); % Covariance
Cov = sig;
logCov = diag(log(diag(sig))); % log Covariance for objective function
a = 1000; % max function multiplier
eps = 1.01; % difference coef
n = round(4+3*log(d)); %number of samples fron MC
%% Calculating CostFunction
xbest = bestSolution.xbest';
fit = zeros(1,n);
Z = randn(d, n);% 
X = repmat(xbest,1,n);
% E = 0; %expected value of mean from MC samples
% U = expm(A)*Z;
U = A*Z;
X = X + U;

for i = 1 : n
    fit(i) = MTZ_new_1D(X(:,i));
end

MeanFit = mean(fit);
costFunction = trace(logCov)+ a*(max(0, MeanFit-eps*bestFunction))^2 %objective function
% costFunction = costFunction

%% Calculating gradient
dV = inv(Cov);

dP = 2*a*max(0, MeanFit - eps*bestFunction); %dP/dE, where P is the Penaulty function


% dE = zeros(1, d); %dE/dsig, where sig is diagonal elements of matrix A
% gradF = zeros(1, d); %gradient of objective function

M = dV;

% Zdif = X - repmat(bestSolution.xbest',1,n);
dE = zeros(d);
for i = 1 : n
    dE = dE + fit(i) * (0.5 * M * ( X(:,i) - xbest )*( X(:,i) - xbest)' * M - 0.5 * M);
end

dE = dE / n;

gradF = dV + dP*dE;

% gradF = diag(diag(gradF));

end
