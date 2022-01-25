import numpy as np
from numpy.linalg import norm

def betaFun(t):
    return 2. / (t+2.)

def FW(objFun, dFun, dFun_mom, m, alpha, T, n, disc_type, line_search = False, momentum = False, tol=1e-8, logistic = False, quadratic = False):
    x_t = np.zeros((m, 1))
    dt = T/(n+0.)
    oracles  = np.zeros(n)
    trace = np.zeros(n)
    obj = np.zeros(n)
    if momentum: 
        theta = x_t[:,0]*0.
        y = x_t*0.
        v = x_t*0.
    for it in range(n):
        d,grad = dFun(x_t,alpha)
        g_t = - d.T.dot(grad).ravel()
        if momentum:
            d,grad, theta,y,v = dFun_mom(x_t,alpha,theta,y,v,betaFun(it*dt))
            
        if disc_type == 'FE':
            step = d
            oracle = 1
        elif disc_type == 'midpoint':
            gamma1 = betaFun(it*dt)
            k1 = gamma1*d
            if momentum:
                d,grad, theta,y,v = dFun_mom(x_t+dt*k1,alpha,theta,y,v,gamma1)
            else:
                d,grad = dFun(x_t+(dt/2)*k1,alpha)
            
            gamma2 = betaFun(it*dt+dt/2)
            k2 = gamma2 *d
            #step = (k1+k2)/2./gamma1
            step = (k2)/1./gamma1
            oracle = 2
        elif disc_type == 'rk44':
            gamma1 = betaFun(it*dt)
            k1 = gamma1*d
            if momentum:
                d,grad, theta,y,v = dFun_mom(x_t+(dt/2)*k1,alpha,theta,y,v,gamma1)
            else:
                d,grad = dFun(x_t+(dt/2)*k1,alpha)
            
            gamma2 = betaFun(it*dt+dt/2)
            k2 = gamma2*d
            if momentum:
                d,grad, theta,y,v = dFun_mom(x_t+(dt/2)*k2,alpha,theta,y,v,gamma2)
            else:
                d,grad = dFun(x_t+(dt/2)*k2,alpha)
            
            gamma3 = betaFun(it*dt+dt/2)
            k3 = gamma3 * d
            if momentum:
                d,grad, theta,y,v = dFun_mom(x_t+dt*k3,alpha,theta,y,v,gamma3)
            else:
                d,grad = dFun(x_t+dt*k3,alpha)
                
            gamma = betaFun(it*dt+dt)
            k4 = gamma*d
            step = (k1+2.*k2+2.*k3+k4)/6./gamma1
            oracle = 4
        
        elif disc_type == 'rk4':
            gamma1 = betaFun(it*dt)
            k1 = gamma1*d
            
            
            if momentum:
                d,grad, theta,y,v = dFun_mom(x_t+(dt/3)*k1,alpha,theta,y,v,gamma1)
            else:
                d,grad = dFun(x_t+(dt/3)*k1,alpha)
                
            gamma2 = betaFun(it*dt+dt/3)
            k2 = gamma2*d
            
            if momentum:
                d,grad, theta,y,v = dFun_mom(x_t+(dt)*(k2-k1/3),alpha,theta,y,v,gamma2)
            else:
                d,grad = dFun(x_t+(dt)*(k2-k1/3),alpha)
                
            gamma3 = betaFun(it*dt+2*dt/3)
            k3 = gamma3*d
            
            if momentum:
                d,grad, theta,y,v = dFun_mom(x_t+dt*(k1-k2+k3),alpha,theta,y,v,gamma3)
            else:
                d,grad = dFun(x_t+dt*(k1-k2+k3),alpha)
                
            gamma = betaFun(it*dt+dt)
            k4 = gamma*d
            step = (k1+3.*k2+3.*k3+k4)/8./gamma1
            oracle = 4
        elif disc_type == 'rk5':

            gamma1 = betaFun(it*dt)
            k1 = gamma1*d
        
            if momentum:
                d,grad, theta,y,v = dFun_mom(x_t+(dt*1/4.)*k1,alpha,theta,y,v,gamma1)
            else:
                d,grad = dFun(x_t+(dt*1/4.)*k1,alpha)
                
            gamma2 = betaFun(it*dt+dt*1/4.)
            k2 = gamma2*d           
            if momentum:
                d,grad, theta,y,v = dFun_mom(x_t+(dt/8)*(k2+k1),alpha,theta,y,v,gamma2)
            else:
                d,grad = dFun(x_t+(dt/8)*(k2+k1),alpha)
            
            gamma3 = betaFun(it*dt+dt*1/4.)
            k3 = gamma3*d
            if momentum:
                d,grad, theta,y,v = dFun_mom(x_t+(dt)*(-k2/2+k3),alpha,theta,y,v,gamma3)
            else:
                d,grad = dFun(x_t+(dt)*(-k2/2+k3),alpha)
                
            gamma4 = betaFun(it*dt+dt*1/2.)
            k4 = gamma4*d            
            if momentum:
                d,grad, theta,y,v = dFun_mom(x_t+(dt/16.)*(3*k1+9*k4),alpha,theta,y,v,gamma4)
            else:
                d,grad = dFun(x_t+(dt/16.)*(3*k1+9*k4),alpha)
                
            gamma5 =  betaFun(it*dt+dt*3/4.)
            k5 = gamma5*d            
            if momentum:
                d,grad, theta,y,v = dFun_mom(x_t+(dt/7.)*(-3*k1+2*k2+12*k3-12*k4+8*k5),alpha,theta,y,v,gamma5)
            else:
                d,grad = dFun(x_t+(dt/7.)*(-3*k1+2*k2+12*k3-12*k4+8*k5),alpha)
                
            gamma = betaFun(it*dt+dt)
            k6 = gamma*d
            step =(7.*k1+32.*k3+12.*k4+32.*k5+7.*k6)/90./gamma
            oracle = 6
        
        beta = betaFun(it*dt)
        
        if line_search:
            if logistic:              
                if it == 0:
                    gamma_ls =1.
                else: 
                    if gamma_ls < 1:
                        gamma_ls *= 2.
                while(objFun(x_t+gamma_ls*step)>objFun(x_t) and gamma_ls > beta):
                    gamma_ls = gamma_ls/2.
                gamma_ls = max(gamma_ls,beta)
                x_t = x_t + gamma_ls*step
            if quadratic:
                gamma =1.
                while(objFun(x_t+gamma*step)>objFun(x_t)+.00001):
                    gamma = gamma/2.
                gamma = max(gamma,beta)
                x_t = x_t + gamma*step
        else: 
            x_t += dt * beta * step
        
        obj[it] = objFun(x_t)
        trace[it] = g_t
        if it == 0: oracles[0] = oracle
        else: oracles[it] = oracles[it-1]+oracle
        
    return x_t,oracles,trace,obj




def FW_matfact(objFun, dFun, dFun_mom, dim, alpha,rho, T, n, disc_type, line_search = False, momentum = False, tol=1e-8):
    x_t = np.zeros(dim)
    dt = T/(n+0.)
    oracles  = np.zeros(n)
    trace = np.zeros(n)
    obj = np.zeros(n)
    if momentum: 
        theta = x_t*0.
        y = x_t*0.
        v = x_t*0.
        
    for it in range(n):
        d,grad = dFun(x_t,alpha,rho)
        g_t = - np.sum(np.multiply(d,grad)[:])
        
        if momentum:
            d,grad, theta,y,v = dFun_mom(x_t,alpha,rho,theta,y,v,betaFun(it*dt))
             
        if disc_type == 'FE':
            step = d
            oracle = 1
        elif disc_type == 'midpoint':
            gamma1 = betaFun(it*dt)
            k1 = gamma1*d
            if momentum:
                d,grad, theta,y,v = dFun_mom(x_t+dt*k1,alpha,rho,theta,y,v,gamma1)
            else:
                #d,grad = dFun(x_t+dt*k1,alpha,rho)
                d,grad = dFun(x_t+(dt/2)*k1,alpha,rho)
            
            #gamma2 = betaFun(it*dt+dt)
            gamma2 = betaFun(it*dt+dt/2)
            k2 = gamma2 *d
            #step = (k1+k2)/2./gamma1
            step = (k2)/1./gamma1
            oracle = 2
        elif disc_type == 'rk44':
            gamma1 = betaFun(it*dt)
            k1 = gamma1*d
            if momentum:
                d,grad, theta,y,v = dFun_mom(x_t+(dt/2)*k1,alpha,rho,theta,y,v,gamma1)
            else:
                d,grad = dFun(x_t+(dt/2)*k1,alpha,rho)
            
            gamma2 = betaFun(it*dt+dt/2)
            k2 = gamma2*d
            if momentum:
                d,grad, theta,y,v = dFun_mom(x_t+(dt/2)*k2,alpha,rho,theta,y,v,gamma2)
            else:
                d,grad = dFun(x_t+(dt/2)*k2,alpha,rho)
            
            gamma3 = betaFun(it*dt+dt/2)
            k3 = gamma3 * d
            if momentum:
                d,grad, theta,y,v = dFun_mom(x_t+dt*k3,alpha,rho,theta,y,v,gamma3)
            else:
                d,grad = dFun(x_t+dt*k3,alpha,rho)
                
            gamma = betaFun(it*dt+dt)
            k4 = gamma*d
            step = (k1+2.*k2+2.*k3+k4)/6./gamma1
            oracle = 4
        
        elif disc_type == 'rk4':
            gamma1 = betaFun(it*dt)
            k1 = gamma1*d
            
            if momentum:
                d,grad, theta,y,v = dFun_mom(x_t+(dt/3)*k1,alpha,rho,theta,y,v,gamma1)
            else:
                d,grad = dFun(x_t+(dt/3)*k1,alpha,rho)
                
            gamma2 = betaFun(it*dt+dt/3)
            k2 = gamma2*d
            
            if momentum:
                d,grad, theta,y,v = dFun_mom(x_t+(dt)*(k2-k1/3),alpha,rho,theta,y,v,gamma2)
            else:
                d,grad = dFun(x_t+(dt)*(k2-k1/3),alpha,rho)
                
            gamma3 = betaFun(it*dt+2*dt/3)
            k3 = gamma3*d
            
            if momentum:
                d,grad, theta,y,v = dFun_mom(x_t+dt*(k1-k2+k3),alpha,rho,theta,y,v,gamma3)
            else:
                d,grad = dFun(x_t+dt*(k1-k2+k3),alpha,rho)
                
            gamma = betaFun(it*dt+dt)
            k4 = gamma*d
            step = (k1+3.*k2+3.*k3+k4)/8./gamma1
            oracle = 4
        elif disc_type == 'rk5':

            gamma1 = betaFun(it*dt)
            k1 = gamma1*d
        
            if momentum:
                d,grad, theta,y,v = dFun_mom(x_t+(dt*1/4.)*k1,alpha,rho,theta,y,v,gamma1)
            else:
                d,grad = dFun(x_t+(dt*1/4.)*k1,alpha,rho)
                
            gamma2 = betaFun(it*dt+dt*1/4.)
            k2 = gamma2*d           
            if momentum:
                d,grad, theta,y,v = dFun_mom(x_t+(dt/8)*(k2+k1),alpha,rho,theta,y,v,gamma2)
            else:
                d,grad = dFun(x_t+(dt/8)*(k2+k1),alpha,rho)
            
            gamma3 = betaFun(it*dt+dt*1/4.)
            k3 = gamma3*d
            if momentum:
                d,grad, theta,y,v = dFun_mom(x_t+(dt)*(-k2/2+k3),alpha,rho,theta,y,v,gamma3)
            else:
                d,grad = dFun(x_t+(dt)*(-k2/2+k3),alpha,rho)
                
            gamma4 = betaFun(it*dt+dt*1/2.)
            k4 = gamma4*d            
            if momentum:
                d,grad, theta,y,v = dFun_mom(x_t+(dt/16.)*(3*k1+9*k4),alpha,rho,theta,y,v,gamma4)
            else:
                d,grad = dFun(x_t+(dt/16.)*(3*k1+9*k4),alpha,rho)
                
            gamma5 =  betaFun(it*dt+dt*3/4.)
            k5 = gamma5*d            
            if momentum:
                d,grad, theta,y,v = dFun_mom(x_t+(dt/7.)*(-3*k1+2*k2+12*k3-12*k4+8*k5),alpha,rho,theta,y,v,gamma5)
            else:
                d,grad = dFun(x_t+(dt/7.)*(-3*k1+2*k2+12*k3-12*k4+8*k5),alpha,rho)
                
            gamma = betaFun(it*dt+dt)
            k6 = gamma*d
            step =(7.*k1+32.*k3+12.*k4+32.*k5+7.*k6)/90./gamma
            oracle = 6
       
        beta = betaFun(it*dt)
        
        if line_search:            
            if it == 0:
                gamma_ls =1.
            else: 
                if gamma_ls < 1:
                    gamma_ls *= 2.
            while(objFun(x_t+gamma_ls*step)>objFun(x_t) and gamma_ls > beta):
                gamma_ls = gamma_ls/2.
            gamma_ls = max(gamma_ls,beta)
            x_t = x_t + gamma_ls*step
        else: 
            x_t += dt * beta * step
        
        obj[it] = objFun(x_t)
        trace[it] = g_t
        if it == 0: oracles[0] = oracle
        else: oracles[it] = oracles[it-1]+oracle
    
    return x_t,oracles,trace,obj


