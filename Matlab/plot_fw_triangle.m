function plot_fw_triangle(Z, marker,f)
%%
xx = linspace(-1,1,100);
yy = linspace(-.25,1.25,100);
[x,y] = meshgrid(xx,yy);
FF = x*0;
for i = 1:size(x,1)
    for j = 1:size(x,2)
        FF(i,j) = f([x(i,j),y(i,j)]);
    end
end

Z = Z + randn(size(Z))*eps;
Z = Z(:,1)+1i*Z(:,2);

% Simple test for Frank-Wolfe algorithm


% rep = '../results/frank-wolfe/';
% [~,~] = mkdir(rep);


% square
%K = [-1-1i; -1+1i; 1+1i; 1-1i];
% % polygon
% m = 3;
%K = exp( 1i*pi/m + 2i*pi * (0:m-1)'/m );
K = [-1-0i;0+1i;1+0i];


hold on

contour(x,y,FF, 15,'k');
% colormap(parula(r+1));
%

plot(K([1:end 1]), 'k.-', 'MarkerSize', 25, 'LineWidth', 2);

% plot(p0(1), p0(2), 'r.',  'MarkerSize', 25);
% plot(zopt(1), zopt(2), 'r.',  'MarkerSize', 25);
for i=1:length(Z)-1
    t = (i-1)/(length(Z)-2);
    if marker
        plot(Z(i:i+1), '.-', 'color', [t 0.5 1-t], 'LineWidth', 2,'markersize',25);    
    else
        plot(Z(i:i+1), '-', 'color', [t 0.5 1-t], 'LineWidth', 2);    
    end
end

plot(real(Z(end)), imag(Z(end)), 'r.',  'MarkerSize', 25);
axis off; axis equal
%ylim([-0.01,0.001])
%xlim([-0.1,0.1])
% saveas(gcf, [rep 'fw-' num2str(test) '.png']);
end