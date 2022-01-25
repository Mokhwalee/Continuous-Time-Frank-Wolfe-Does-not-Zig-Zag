% Plot the triangle trajectories based on FW method

% Inputs
% Z : set of points
% f : loss function

function plot_fw_triangle(Z, f)

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

K = [-1-0i;0+1i;1+0i];


hold on

contour(x,y,FF, 15,'k');

plot(K([1:end 1]), 'k.-', 'MarkerSize', 25, 'LineWidth', 2);

for i=1:length(Z)-1
    t = (i-1)/(length(Z)-2);
    plot(Z(i:i+1), '.-', 'color', [t 0.5 1-t], 'LineWidth', 2,'markersize',25);
end

plot(real(Z(end)), imag(Z(end)), 'r.',  'MarkerSize', 25);
axis off; axis equal
end