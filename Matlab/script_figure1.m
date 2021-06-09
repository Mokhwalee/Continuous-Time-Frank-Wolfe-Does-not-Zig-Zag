%%%% paper figure 1 %%%%%

figure(1)
clf

time_norm = 1;

disc_type = 'vanilla';
accel = 'none';

linesearch = 0;
Tmax = 1000;

p0 = [0 1];
zopt = [0,0];
f = @(z)1/2*sum((z-zopt).^2,2);
df = @(z)(z-zopt);

gf_vec = [1,2,10];
for i = 1:length(gf_vec)
gamma_fact = gf_vec(i);
subplot(2,3,i)

dt = 1;
[Z,F,T] = frank_wolfe(f,df,p0,Tmax,dt,linesearch,disc_type,accel,time_norm, gamma_fact);
loglog(T,F,'linewidth',1.5)
hold on
dt = .1;
[Z,F,T] = frank_wolfe(f,df,p0,Tmax,dt,linesearch,disc_type,accel,time_norm, gamma_fact);
semilogy(T,F,'linewidth',1.5)


dt = .01;
[Z,F,T] = frank_wolfe(f,df,p0,Tmax,dt,linesearch,disc_type,accel,time_norm, gamma_fact);
semilogy(T,F,'linewidth',1.5)


dt = .001;
[Z,F,T] = frank_wolfe(f,df,p0,Tmax,dt,linesearch,disc_type,accel,time_norm, gamma_fact);
semilogy(T,F,'linewidth',1.5)

tfix = logspace(-3,log10(Tmax),100);
loglog(tfix,(gamma_fact./(tfix+gamma_fact)).^gamma_fact,'k','linewidth',2)
hold on

ylim([1e-5,10])
xlim([.01,1000])

xlabel('time')
ylabel('gap')

title(sprintf('$c = %d$',gamma_fact),'interpreter','latex','fontsize',14)
legend('$\Delta = 1$','$\Delta = 0.1$','$\Delta = 0.01$','$\Delta = 0.001$','$(\frac{c}{c+t})^c$','interpreter','latex','location','southwest','fontsize',12)
end