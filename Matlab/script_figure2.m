%%%%% paper figure 2 %%%%%

figure(1)
clf

figure(2)
clf

time_norm = 1;

disc_type = 'vanilla';
accel = 'none';
gamma_fact = 2;
linesearch = 0;
Tmax = 1000;
Tp = 15;

rnglist = [1,26,9,3,15];

%
for k = 1:3
    if k == 1
        p0 = [0 1];
        zopt = [0,0];
    else
        
        
        rng(rnglist(k))
        
        p0 = rand(1,2);
        p0 = p0 / sum(p0);
        p0(1) = p0(1)*sign(randn(1));
        zopt = randn(1,2)/2;
        
    end
    
    
    f = @(z)1/2*sum((z-zopt).^2,2);
    df = @(z)(z-zopt);
    
    
    dt = 1;
    figure(1)
    subplot(3,3,(k-1)*3+1)
    [Z,F,T] = frank_wolfe(f,df,p0,Tmax,dt,linesearch,disc_type,accel,time_norm, gamma_fact);
    plot_fw_triangle(Z(1:Tp/dt,:),1,f)
    title('$\Delta = 1$','interpreter','latex','fontsize',12)
    
    figure(2)
    subplot(3,1,k)
    loglog(T,F,'linewidth',1.5)
    hold on
    
    
    dt = .25;
    figure(1)
    subplot(3,3,(k-1)*3+2)
    [Z,F,T] = frank_wolfe(f,df,p0,Tmax,dt,linesearch,disc_type,accel,time_norm, gamma_fact);
    plot_fw_triangle(Z(1:Tp/dt,:),1,f)
    title('$\Delta = 0.25$','interpreter','latex','fontsize',12)
    
    figure(2)
    subplot(3,1,k)
    loglog(T,F,'linewidth',1.5)
    hold on
    
    
    dt = .01;
    figure(1)
    subplot(3,3,(k-1)*3+3)
    [Z,F,T] = frank_wolfe(f,df,p0,Tmax,dt,linesearch,disc_type,accel,time_norm, gamma_fact);
    plot_fw_triangle(Z(1:Tp/dt,:),1,f)
    title('$\Delta = 0.01$','interpreter','latex','fontsize',12)
    
    figure(2)
    subplot(3,1,k)
    loglog(T,F,'linewidth',1.5)
    hold on
    ylim([1e-6,10])
    axis tight
    set(gca,'xtick',[.01,.1,1,10,100,1000])
    set(gca,'ytick',[.0000001,.00001,.001,.1,10])
    if k == 3
    xlabel('Time (t)','fontsize',12)
    end
    ylabel('Gap','fontsize',12)
    
    set(gca,'fontsize',12)
    legend('$\Delta = 1$','$\Delta = 0.1$','$\Delta = 0.01$','$\Delta = 0.001$','$(\frac{c}{c+t})^c$','interpreter','latex','location','southwest','fontsize',12)
end

