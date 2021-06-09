% Main function for frank wolfe method for triangle example

% Input explanations : 
% f : main loss function
% df : main loss gradient
% zstart : starting point
% Tmax : designated time limit
% dt : time step
% linesearch : use linesearch(1) or not(0)
% disc_type : type of multi-step discretization method
% accel : use nesterov accelerated method('nesterovacc') or not('vanilla')
% gamma_fact : constant c in gamma=c/(c+t)

% Output explanations : 
% Z : List of points 
% F : Objective values
% T : Time

function [Z,F,T] = frank_wolfe(f,df,zstart,Tmax,dt,linesearch, disc_type, accel, gamma_fact)


if linesearch > 0 && dt < 1
    fprintf('warning: doesn''t make sense to use line search in continuous time mode\n')
end

K = [-1,0;0,1;1,0]; % vertices of the triangle

niter = ceil(Tmax/dt);

    function u = get_lmo0(d)
        [~,j] = min(K*d',[],1);
        u = K(j,:);
    end












    function b = betafun(it)
        b = gamma_fact/(gamma_fact+it);
    end

% dotp = @(u,v)real(u*conj(v));

Z = zeros(niter,2);
F = zeros(niter,1);
T = zeros(niter,1);

z = zstart;

if strcmpi(accel,'nesterovacc')
    theta = 0*z;
    y = 0*z;
    v = 0*z;
    gamma_nest = 0;
end









    function u = get_lmo(x)
        if strcmpi(accel,'nesterovacc')
            y = x * (1.-gamma_nest) + gamma_nest * v;
            dy = df(y);
            theta = (1-gamma_nest)*theta+gamma_nest*dy;
            u = get_lmo0(theta);
        else
            d = df(x);
            u = get_lmo0(d);
        end
    end

    function [step] = mdpt_step(z)
        
        u = get_lmo(z);
        gamma1 = betafun(i*dt);
        gamma_nest = gamma1;
        k1 = gamma1*(u-z)*dt;
        
        u = get_lmo(z+k1);
        d_z = u-(z+k1);
        gamma2 = betafun(i*dt+dt);
        gamma_nest = gamma2;
        k2 = gamma2*d_z*dt;
        step = ((k2+k1)/2) / gamma1 / dt;
    end

    function [step] = vanilla_step(z)
        gamma_nest = betafun(i*dt);
        u = get_lmo(z);
        step = (u-z);
    end

    function [step] = rk44_step(z)
        d = df(z);
        u = get_lmo(z);
        % update
        gamma1 = betafun(i*dt);
        gamma_nest = gamma1;
        k1 = gamma1*(u-z) *dt;
        u = get_lmo(z+(dt/2)*k1);
        
        d_z = u-(z+(dt/2)*k1);
        gamma2 = betafun(i*dt+dt/2);
        gamma_nest = gamma2;
        k2 = gamma2*d_z *dt;
        u = get_lmo(z+(dt/2)*k2);
        
        d_z = u-(z+(dt/2)*k2);
        gamma3 = betafun(i*dt+dt/2);
        gamma_nest = gamma3;
        k3 = gamma3*d_z *dt ;
        u = get_lmo(z+dt*k3);
        
        d_z = u-dt*k3;
        gamma4 = betafun(i*dt+dt);
        gamma_nest = gamma4;
        k4 = gamma4*d_z *dt;
        
        step = (k1+2*k2+2*k3+k4)/gamma1/6/dt;
    end

    function [step] = rk45_step(z)
        CH = [47/450,0,12/25,32/225,1/30,6/25];
        A = [0,2/9,1/3,3/4,1,5/6];
        B = [0,0,0,0,0;...
            2/9,0,0,0,0;...
            1/12,1/4,0,0,0;...
            67/128,-243/128,135/64,0,0;
            -17/12,27/4,-27/5,16/15,0;...
            65/432,-5/16,13/16,4/27,5/144];
        
        
        u = get_lmo(z);
        gamma1 = betafun(i*dt);
        gamma_nest = gamma1;
        k1 = gamma1*(u-z) *dt;
        
        
        z2 = A(2)*dt*B(2,1)*k1;
        u = get_lmo(z+ z2);
        d_z = u-(z+z2);
        gamma2 = betafun(i*dt+dt*A(2));
        gamma_nest = gamma2;
        k2 = gamma2*d_z *dt;
        
        
        z3 = A(3)*dt*(B(3,1)*k1+B(3,2)*k2);
        u = get_lmo(z+ z3);
        d_z = u-(z+z3);
        gamma3 = betafun(i*dt+dt*A(3));
        gamma_nest = gamma3;
        k3 = gamma3*d_z *dt;
        
        
        z4 = A(4)*dt*(B(4,1)*k1+B(4,2)*k2+B(4,3)*k3);
        u = get_lmo(z+ z4);
        d_z = u-(z+z4);
        gamma4 = betafun(i*dt+dt*A(4));
        gamma_nest = gamma4;
        k4 = gamma4*d_z *dt;
        
        
        z5 = A(5)*dt*(B(5,1)*k1+B(5,2)*k2+B(5,3)*k3+B(5,4)*k4);
        u = get_lmo(z+ z5);
        d_z = u-(z+z5);
        gamma5 = betafun(i*dt+dt*A(5));
        gamma_nest = gamma5;
        k5 = gamma5*d_z *dt;
        
        
        
        z6 = A(6)*dt*(B(6,1)*k1+B(6,2)*k2+B(6,3)*k3+B(6,4)*k4+B(6,5)*k5);
        u = get_lmo(z+ z6);
        d_z = u-(z+z6);
        gamma6 = betafun(i*dt+dt*A(6));
        gamma_nest = gamma6;
        k6 = gamma6*d_z *dt;
        
        
        step = CH(1)*k1+CH(2)*k2+CH(3)*k3+CH(4)*k4+CH(5)*k5+CH(6)*k6;
        
    end

    function [time,gap,step] = take_step(z)
        d = df(z);
        gap = d*(z-get_lmo0(d))';
        time = i*dt;
        if strcmpi(disc_type,'vanilla')
            [step] = vanilla_step(z);
            time = i*dt;
        elseif strcmpi(disc_type,'midpoint')
            [step] = mdpt_step(z);
    
        elseif strcmpi(disc_type, 'rk44')
            [step] = rk44_step(z);
            
        elseif strcmpi(disc_type, 'rk45')
            [step] = rk45_step(z);
            
        end
    end

for i=1:niter
    Z(i,:) = z;
    
    if strcmpi(accel,'none')
        [T(i),F(i),step] = take_step(z);
    elseif strcmpi(accel,'extragradient')
        [T(i),F(i),step] = take_step(z);
        gamma1 = betafun(i*dt);
        keg = gamma1*step*dt;
        [~,~,step] = take_step(z+keg);
        
    elseif strcmpi(accel,'nesterovacc')
        gamma_nest = betafun(i*dt);
        [T(i),F(i),step] = take_step(z);
        
    end
    
    
    
    
    
    if linesearch == 0
        gamma =betafun(i*dt);
    elseif linesearch > 0
        gamma = 1;
        while(f((1-gamma)*z+gamma*step)>f(z)-eps) && (gamma > 1e-10)
            gamma = gamma/2;
        end
        
        gamma = max(betafun(i*dt),gamma);
        
    end
    z = z + gamma*step*dt;
    
end


end