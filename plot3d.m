SigmaL = diag(sig_best);
SigmaL = real(SigmaL);
mu = real(mu_best);

xmin = min(mu);
xmax = max(mu);
sigmax = max(SigmaL);
x = [xmin-sigmax:1:xmax+sigmax];

norm = zeros(length(x), length(sig_best));
for i=1:length(x_true)
    norm(:,i) = normpdf(x, real(mu_best(i)), SigmaL(i));
end

zprog = [10:10:(length(x_true)*10)];
zbig = repmat(zprog, [length(norm),1]);


figure;
p = plot3(zbig, x, norm);
box on;

xlabel('Глубина слоя, м')
ylabel('Величина сопротивления, Ом)')
zlabel('Вероятность значения')
title('Resulting impedance distribution')

% load(johnyf-fig2u3d-74fe75d)
% saveas(p, 'Impedance_distr', 'pdf');
% ax = gca;
% fig2u3d(ax, 'tet', 'pdf');