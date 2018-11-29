clear all;
x_true = [300 50 600 100 1000 300 600];
tic
%%
Params.NOD = length(x_true); % number of dimensions
Params.callsMax = 1e4; % number of calls

PSOParams = Params;
PSOParams.w = 0.7298; % inertia coefficient, 0.7298
PSOParams.c1 = 1.4962; % cognitive direction multiplier, 1.4962
PSOParams.c2 = 1.4962; % social direction multiplier, 1.4962
PSOParams.BC_v = -0.5; % boundary conditions for v: 0 - adhesion, -1 - reflecting, ...
PSOParams.NOP = 20; % number of particles
PSOParams.NON = 1; % number of neighborhoods
PSOParams.Vstart = 0; % max value of initial velocity / (bordmax-bordmin)
%%
borders.min = 10*ones(1, Params.NOD);
borders.max = 1000*ones(1, Params.NOD);
borders.Vmax = zeros(1, Params.NOD);
%%
lT = 100;
sqT(1:lT) = 0;
sqT(1) = 0.01;
for i = 2:lT
    sqT(i) = sqT(i-1)*1.1;
end
rr_true = MTZ(x_true, sqT);

%% Getting best solution and corresponding function value
MTZ_blackbox = @(x) MTZ_new_1D(x, rr_true, sqT);
[bestSolution, ~, ~] = PSO(MTZ_blackbox, borders, PSOParams);
xbest = bestSolution.xbest;
bestFunction = MTZ_new_1D(bestSolution.xbest, rr_true, sqT);




%% Gradient descent
call = 0;
step0 = 10; % Learning rate
sig_prev = 0.1 * eye(length(x_true)); % Starting Sigma
mu_prev = bestSolution.xbest';
% [obj_best, grad_prev, cost2] = Obj_F(sig_prev, bestSolution, @MTZ_new_1D, bestFunction, rr_true, sqT);
[obj_best, grad_prev, dmu] = Obj_F_NES(sig_prev, mu_prev, @MTZ_new_1D, bestFunction, rr_true, sqT);
step = step0;
sig_new = zeros(size(sig_prev));
max_call = 1000;
while call < max_call
    sig_new = sig_prev + step*grad_prev;
    mu_new = mu_prev + step*dmu;
    [obj_new, grad_new, dmu] = Obj_F_NES(sig_new, mu_new, @MTZ_new_1D, bestFunction, rr_true, sqT);
    
    if norm(sig_new - sig_prev) < 1e-8
        disp('gain is less than 1e-8')
        break
    end
    
    if obj_new > obj_best
        obj_best = obj_new;
        grad_prev = grad_new;
        sig_prev = sig_new;
        sig_best = sig_new;
        mu_prev = mu_new;
        mu_best = mu_new;
%         cost_glob = cost2;
        step = step0;
    else
        step = step/2;
    call = call + 1;
    end;
end;
% toc
%     