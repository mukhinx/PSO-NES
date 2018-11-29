function[costFunction, gradF, dmu]= Obj_F_NES(sig, mu, MTZ_new_1D, bestFunction,rr_true, sqT)

%% NES Parameters
A = sqrtm(sig);
Cov = sig;
% logCov = log(Cov); % log Covariance for objective function
detCov = det(Cov);
eps = 1.1; % difference coef


%% Calculating CostFunction
a =  max(Cov(:)); % max function multiplier
% a = 30;

%% xNES
d = length(sig);

n = round(4+3*log(d)); %number of samples fron MC
% nmu = 1;
% nsB = 3/5 * (3 + log(d))/(d*sqrt(d));
% u = max(0.0, log(n/2+1.0)-log(1:n)); 
% u = u / sum(u) - 1/n;

% sigma = (abs(det(A)))^(1/d);
% B = A / sigma;

fit = zeros(1,n);
Z = randn(d, n);% 
Mu = repmat(mu,1,n);
% sBz = sigma * B * Z;
X = Mu + A*Z;

for i = 1 : n
    [fit(i), ~] = MTZ_new_1D(X(:,i), rr_true, sqT);
end

% [~, idx] = sort(fit);
% X = X(idx); Z = Z(idx);
% Gd = sum(u*Z, 2);
% GM = sum(u*(Z*Z' - eye(d)),2);
% Gs = trace(GM) / d;
% GB = GM - Gs * eye(d);

MeanFit = mean(fit);
first = sqrt(detCov);
% second = max(0, (MeanFit-eps*bestFunction)^2);
second = exp(a*(MeanFit-eps*bestFunction)^2);
costFunction = first - second; %objective function
% cost2 = second;
%% Calculating gradient
iCov = inv(Cov);
dV = 0.5 * first * iCov;
% dP = 2*a*max(0, MeanFit - eps*bestFunction); %dP/dE, where P is the Penaulty function
dP = 2 * a * second * (MeanFit-eps*bestFunction);

dE = zeros(d);
FS = zeros(d);
dmu = zeros(size(mu));
Fmu = zeros(d);

% nablaS = (0.5 * iCov * ( X - Mu )*( X - Mu)' * iCov - 0.5 * iCov);
% nablamu = iCov * (X - Mu);

% dE = fit * nablaS / n;
% dmu = fit * nablamu / n;
for i = 1 : n
    nablaS = (0.5 * iCov * ( X(:,i) - mu )*( X(:,i) - mu)' * iCov - 0.5 * iCov);
    dE = dE + fit(i) * nablaS;
    FS = FS + nablaS * nablaS' / n;
    
    nablamu = iCov * (X(:,i) - mu);
    dmu = dmu + fit(i) * nablamu;
    Fmu = Fmu + nablamu * nablamu';
end

dE = dE/n; dmu = dmu/n;
FS = FS/n; Fmu = Fmu/n;

dE = inv(FS) * dE;
dmu = inv(Fmu) * dmu;

gradF = dV - dP*dE;

end
