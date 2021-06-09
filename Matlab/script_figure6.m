%%%%% paper figure 6 %%%%%
clc
clear

rng('default')

figure(1)
clf
figure(2)
clf

dt = 1;
linesearch = 0;
gamma_fact = 2;

time_norm = 0;
accel = 'nesterovacc';

p0 = [0 1];
zopt = [0,0];


f = @(z)1/2*sum((z-zopt).^2,2);
df = @(z)(z-zopt);

Tmax = 400;
Tp = 15;

figure(1)
subplot(1,3,1)
disc_type = 'vanilla';
[Z,F,T] = frank_wolfe(f,df,p0,Tmax,dt,linesearch,disc_type,accel,time_norm, gamma_fact);
plot_fw_triangle(Z(1:Tp/dt,:),1,f)
title('FW')


figure(2)
loglog(T,F,'linewidth',1.5)
hold on

figure(1)
subplot(1,3,2)
disc_type = 'midpoint';
[Z,F,T] = frank_wolfe(f,df,p0,Tmax,dt,linesearch,disc_type,accel,time_norm, gamma_fact);
plot_fw_triangle(Z(1:Tp/dt,:),1,f)
title('FW-MD')


figure(2)
semilogy(T,F,'linewidth',1.5)
hold on

figure(1)
subplot(1,3,3)
disc_type = 'rk44';
[Z,F,T] = frank_wolfe(f,df,p0,Tmax,dt,linesearch,disc_type,accel,time_norm, gamma_fact);
plot_fw_triangle(Z(1:Tp/dt,:),1,f)
title('FW-RK44')

figure(2)
semilogy(T,F,'linewidth',1.5)
hold on
axis tight
legend('FW','FW-MD','FW-RK44','location','southwest')
ylabel('gap')
xlabel('iter (k)')
