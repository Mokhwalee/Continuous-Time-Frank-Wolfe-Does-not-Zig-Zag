import numpy as np
from numpy.linalg import norm
import pylab as plt
import pickle


def obj_fun(A,b,z):
    return ((norm(z.T.dot(A.T)-b)**2.))/2.
def betaFun(t):
    return 2. / (t+2.)

def dFun(A,x,alpha,Atb):

    Ax = x.T.dot(A.T).ravel()
    grad = (A.T.dot(Ax) - Atb)
    
    idx_oracle = np.argmax(np.abs(grad))
    mag_oracle = alpha * np.sign(-grad[idx_oracle])
    d = -x.copy()
    d[idx_oracle] += mag_oracle
    return d, grad

def dFun_mom(A,x,alpha,Atb, theta,y,v, gamma):
    y = x * (1.-gamma) + gamma * v
    
    
    Ay = y.T.dot(A.T).ravel()
    grady = (A.T.dot(Ay) - Atb)
    
    theta = theta * (1.-gamma) + gamma * grady
    
    
    idx_oracle = np.argmax(np.abs(theta))
    mag_oracle = alpha * np.sign(-theta[idx_oracle])
    v = x*0.
    v[idx_oracle,0] = mag_oracle
    
    d = v-x
    return d, grady, theta,y,v

def FW(A,b,m,alpha,i,T,n,disc_type, line_search = False, momentum = False, tol=1e-8):
    def f(z): return obj_fun(A,b,z)
    x_t = np.zeros((m, 1))
    dt = T/(n+0.)
    Atb = A.T.dot(b)
    oracles  = np.zeros(n)
    trace = np.zeros(n)
    obj = np.zeros(n)
    if momentum: 
        theta = x_t[:,0]*0.
        y = x_t*0.
        v = x_t*0.
    for it in range(n):
        #gamma_nest = betaFun(i*dt+dt)
        d,grad = dFun(A,x_t,alpha,Atb)
        g_t = - d.T.dot(grad).ravel()
        if momentum:
            d,grad, theta,y,v = dFun_mom(A,x_t,alpha,Atb,theta,y,v,betaFun(i*dt))
            
        
        if disc_type == 'FE':
            step = d
            oracle = 1
        elif disc_type == 'midpoint':
            gamma1 = betaFun(i*dt)
            k1 = gamma1*d
            if momentum:
                d,grad, theta,y,v = dFun_mom(A,x_t+dt*k1,alpha,Atb,theta,y,v,gamma1)
            else:
                d,grad = dFun(A,x_t+dt*k1,alpha,Atb)
            
            gamma2 = betaFun(i*dt+dt)
            k2 = gamma2 *d
            step = (k1+k2)/2./gamma1
            oracle = 2
        elif disc_type == 'rk44':
            gamma1 = betaFun(it*dt)
            k1 = gamma1*d
            if momentum:
                d,grad, theta,y,v = dFun_mom(A,x_t+(dt/2)*k1,alpha,Atb,theta,y,v,gamma1)
            else:
                d,grad = dFun(A,x_t+(dt/2)*k1,alpha,Atb)
            
            gamma2 = betaFun(it*dt+dt/2)
            k2 = gamma2*d
            if momentum:
                d,grad, theta,y,v = dFun_mom(A,x_t+(dt/2)*k2,alpha,Atb,theta,y,v,gamma2)
            else:
                d,grad = dFun(A,x_t+(dt/2)*k2,alpha,Atb)
            
            gamma3 = betaFun(it*dt+dt/2)
            k3 = gamma3 * d
            if momentum:
                d,grad, theta,y,v = dFun_mom(A,x_t+dt*k3,alpha,Atb,theta,y,v,gamma3)
            else:
                d,grad = dFun(A,x_t+dt*k3,alpha,Atb)
                
            gamma = betaFun(it*dt+dt)
            k4 = gamma*d
            step = (k1+2.*k2+2.*k3+k4)/6./gamma1
            oracle = 4
        
        elif disc_type == 'rk4':
            gamma1 = betaFun(it*dt)
            k1 = gamma1*d
            
            
            if momentum:
                d,grad, theta,y,v = dFun_mom(A,x_t+(dt/3)*k1,alpha,Atb,theta,y,v,gamma1)
            else:
                d,grad = dFun(A,x_t+(dt/3)*k1,alpha,Atb)
                
            gamma2 = betaFun(it*dt+dt/3)
            k2 = gamma2*d
            
            if momentum:
                d,grad, theta,y,v = dFun_mom(A,x_t+(dt)*(k2-k1/3),alpha,Atb,theta,y,v,gamma2)
            else:
                d,grad = dFun(A,x_t+(dt)*(k2-k1/3),alpha,Atb)
                
            gamma3 = betaFun(it*dt+2*dt/3)
            k3 = gamma3*d
            
            if momentum:
                d,grad, theta,y,v = dFun_mom(A,x_t+dt*(k1-k2+k3),alpha,Atb,theta,y,v,gamma3)
            else:
                d,grad = dFun(A,x_t+dt*(k1-k2+k3),alpha,Atb)
                
            gamma = betaFun(it*dt+dt)
            k4 = gamma*d
            step = (k1+3.*k2+3.*k3+k4)/8./gamma1
            oracle = 4
        elif disc_type == 'rk5':

            gamma1 = betaFun(it*dt)
            k1 = gamma1*d
        
            if momentum:
                d,grad, theta,y,v = dFun_mom(A,x_t+(dt*1/4.)*k1,alpha,Atb,theta,y,v,gamma1)
            else:
                d,grad = dFun(A,x_t+(dt*1/4.)*k1,alpha,Atb)
                
            gamma2 = betaFun(it*dt+dt*1/4.)
            k2 = gamma2*d           
            if momentum:
                d,grad, theta,y,v = dFun_mom(A,x_t+(dt/8)*(k2+k1),alpha,Atb,theta,y,v,gamma2)
            else:
                d,grad = dFun(A,x_t+(dt/8)*(k2+k1),alpha,Atb)
            
            gamma3 = betaFun(it*dt+dt*1/4.)
            k3 = gamma3*d
            if momentum:
                d,grad, theta,y,v = dFun_mom(A,x_t+(dt)*(-k2/2+k3),alpha,Atb,theta,y,v,gamma3)
            else:
                d,grad = dFun(A,x_t+(dt)*(-k2/2+k3),alpha,Atb)
                
            gamma4 = betaFun(it*dt+dt*1/2.)
            k4 = gamma4*d            
            if momentum:
                d,grad, theta,y,v = dFun_mom(A,x_t+(dt/16.)*(3*k1+9*k4),alpha,Atb,theta,y,v,gamma4)
            else:
                d,grad = dFun(A,x_t+(dt/16.)*(3*k1+9*k4),alpha,Atb)
                
            gamma5 =  betaFun(it*dt+dt*3/4.)
            k5 = gamma5*d            
            if momentum:
                d,grad, theta,y,v = dFun_mom(A,x_t+(dt/7.)*(-3*k1+2*k2+12*k3-12*k4+8*k5),alpha,Atb,theta,y,v,gamma5)
            else:
                d,grad = dFun(A,x_t+(dt/7.)*(-3*k1+2*k2+12*k3-12*k4+8*k5),alpha,Atb)
                
            gamma = betaFun(it*dt+dt)
            k6 = gamma*d
            step =(7.*k1+32.*k3+12.*k4+32.*k5+7.*k6)/90./gamma
            oracle = 6
        elif disc_type == 'rk8':
            d,grad = dFun(A,x_t,alpha,Atb)
        

            gamma1 = betaFun(it*dt)
            k1 = gamma1*d
            d,grad = dFun(A,x_t+(dt*4/27)*k1,alpha,Atb)
            k2 = betaFun(it*dt+dt*4/27)*d
            d,grad = dFun(A,x_t+(dt/18)*(3*k2+k1),alpha,Atb)
            k3 = betaFun(it*dt+dt*2/9)*d
            d,grad = dFun(A,x_t+(dt/12)*(k1+3*k3),alpha,Atb)
            k4 = betaFun(it*dt+dt*1/3)*d
            d,grad = dFun(A,x_t+(dt/8)*(k1+3*k4),alpha,Atb)
            k5 = betaFun(it*dt+dt*1/2)*d
            d,grad = dFun(A,x_t+(dt/54)*(13*k1-27*k3+42*k4+8*k5),alpha,Atb)
            k6 = betaFun(it*dt+dt*2/3)*d
            d,grad = dFun(A,x_t+(dt/4320)*(389*k1-54*k3+966*k4-824*k5+243*k6),alpha,Atb)
            k7 = betaFun(it*dt+dt*1/6)*d   
            d,grad = dFun(A,x_t+(dt/20)*(-234*k1+81*k3-1164*k4+656*k5-122*k6+800*k7),alpha,Atb)
            k8 = betaFun(it*dt+dt)*d 
            d,grad = dFun(A,x_t+(dt/288)*(-127*k1+18*k3-678*k4+456*k5-9*k6+576*k7+4*k8),alpha,Atb)
            k9 = betaFun(it*dt+dt*5/6)*d          
            d,grad = dFun(A,x_t+(dt/820)*(1481*k1-81*k3+7104*k4-3376*k5+72*k6-5040*k7-60*k8+720*k9),alpha,Atb)
            gamma = betaFun(it*dt+dt)
            k10 = gamma*d
            step = (41*k1+27*k4+272*k5+27*k6+216*k7+216*k9+41*k10)/840./gamma
            oracle = 10
        
        beta = betaFun(it*dt)
        
        if line_search:
            gamma =1.
            while(f(x_t+gamma*step)>f(x_t)+.00001):
                gamma = gamma/2.
            gamma = max(gamma,beta)
            x_t = x_t + gamma*step
        else: 
            x_t += dt * beta * step
        
        
        obj[it] =f(x_t)
        trace[it] = g_t
        if it == 0: oracles[0] = oracle
        else: oracles[it] = oracles[it-1]+oracle
        
    plt.figure(100)
    plt.plot(x_t)
    return x_t,oracles,trace,obj