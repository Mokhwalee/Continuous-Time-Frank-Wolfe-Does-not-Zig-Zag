'''
Logistic function bundles
If you didn't download autograd : pip install autograd
'''

import numpy as np
from numpy.linalg import norm
import pylab as plt
from scipy import sparse
import pandas as pd
from sklearn.preprocessing import StandardScaler
from autograd import grad
import pickle

def sigmoid(z):
    return np.divide(1.,1.+np.exp(-z))
def obj_fun(z, m, bA):
    return np.sum(np.log(1.+np.exp(-np.dot(bA, z))))/(m+0.)
def betaFun(t):
    return 2. / (t+2.)

def dFun(x, alpha, bA):
    grad = np.dot(bA.T, sigmoid(np.dot(bA,x)) - 1.)    
    idx_oracle = np.argmax(np.abs(grad))
    mag_oracle = alpha * np.sign(-grad[idx_oracle])
    d = -x.copy()
    d[idx_oracle] += mag_oracle
    return d, grad

def dFun_mom(bA, x,alpha, theta,y,v, gamma):
    y = x * (1.-gamma) + gamma * v
    
    grady = np.dot(bA.T,sigmoid(np.dot(bA,y)) - 1.)[:,0]
    theta = theta * (1.-gamma) + gamma * grady
    
    idx_oracle = np.argmax(np.abs(theta))
    mag_oracle = alpha * np.sign(-theta[idx_oracle])
    v = x*0.
    v[idx_oracle,0] = mag_oracle
    
    d = v-x
    return d, grady, theta,y,v

def FW(alpha, T, n, m, i, bA, disc_type, line_search = False, momentum = False, tol=1e-8):
    def f(z): return obj_fun(z, m, bA)
    x_t = np.zeros((m, 1))
    dt = T/(n+0.)
    oracles  = np.zeros(n)
    trace = np.zeros(n)
    obj = np.zeros(n)
    if momentum: 
        theta = x_t[:,0]*0.
        y = x_t*0.
        v = x_t*0.
    print(disc_type,line_search,momentum)
    for it in range(n):
        if it % int(n/10) == 0:
            print('.'),
        #gamma_nest = betaFun(i*dt+dt)
        d,grad = dFun(x_t,alpha, bA)
        g_t = - d.T.dot(grad).ravel()
        if momentum:
            d,grad, theta,y,v = dFun_mom(bA, x_t,alpha,theta,y,v,betaFun(i*dt))
            
        if disc_type == 'FE':
            step = d
            oracle = 1
        elif disc_type == 'midpoint':
            gamma1 = betaFun(i*dt)
            k1 = gamma1*d
            if momentum:
                d,grad, theta,y,v = dFun_mom(bA, x_t+dt*k1,alpha,theta,y,v,gamma1)
            else:
                d,grad = dFun(x_t+dt*k1, alpha, bA)
            
            gamma2 = betaFun(i*dt+dt)
            k2 = gamma2 *d
            step = (k1+k2)/2./gamma1
            oracle = 2
        elif disc_type == 'rk44':
            gamma1 = betaFun(it*dt)
            k1 = gamma1*d
            if momentum:
                d,grad, theta,y,v = dFun_mom(bA, x_t+(dt/2)*k1,alpha,theta,y,v,gamma1)
            else:
                d,grad = dFun(x_t+(dt/2)*k1,alpha, bA)
            
            gamma2 = betaFun(it*dt+dt/2)
            k2 = gamma2*d
            if momentum:
                d,grad, theta,y,v = dFun_mom(bA, x_t+(dt/2)*k2,alpha,theta,y,v,gamma2)
            else:
                d,grad = dFun(x_t+(dt/2)*k2,alpha, bA)
            
            gamma3 = betaFun(it*dt+dt/2)
            k3 = gamma3 * d
            if momentum:
                d,grad, theta,y,v = dFun_mom(bA, x_t+dt*k3,alpha,theta,y,v,gamma3)
            else:
                d,grad = dFun(x_t+dt*k3,alpha, bA)
                
            gamma = betaFun(it*dt+dt)
            k4 = gamma*d
            step = (k1+2.*k2+2.*k3+k4)/6./gamma1
            oracle = 4
        
        elif disc_type == 'rk4':
            gamma1 = betaFun(it*dt)
            k1 = gamma1*d
            
            
            if momentum:
                d,grad, theta,y,v = dFun_mom(bA, x_t+(dt/3)*k1,alpha,theta,y,v,gamma1)
            else:
                d,grad = dFun(x_t+(dt/3)*k1, alpha, bA)
                
            gamma2 = betaFun(it*dt+dt/3)
            k2 = gamma2*d
            
            if momentum:
                d,grad, theta,y,v = dFun_mom(bA, x_t+(dt)*(k2-k1/3),alpha,theta,y,v,gamma2)
            else:
                d,grad = dFun(x_t+(dt)*(k2-k1/3), alpha, bA)
                
            gamma3 = betaFun(it*dt+2*dt/3)
            k3 = gamma3*d
            
            if momentum:
                d,grad, theta,y,v = dFun_mom(bA, x_t+dt*(k1-k2+k3),alpha,theta,y,v,gamma3)
            else:
                d,grad = dFun(x_t+dt*(k1-k2+k3), alpha, bA)
                
            gamma = betaFun(it*dt+dt)
            k4 = gamma*d
            step = (k1+3.*k2+3.*k3+k4)/8./gamma1
            oracle = 4
        elif disc_type == 'rk5':

            gamma1 = betaFun(it*dt)
            k1 = gamma1*d
        
            if momentum:
                d,grad, theta,y,v = dFun_mom(bA, x_t+(dt*1/4.)*k1,alpha,theta,y,v,gamma1)
            else:
                d,grad = dFun(x_t+(dt*1/4.)*k1, alpha, bA)
                
            gamma2 = betaFun(it*dt+dt*1/4.)
            k2 = gamma2*d           
            if momentum:
                d,grad, theta,y,v = dFun_mom(bA, x_t+(dt/8)*(k2+k1),alpha,theta,y,v,gamma2)
            else:
                d,grad = dFun(x_t+(dt/8)*(k2+k1), alpha, bA)
            
            gamma3 = betaFun(it*dt+dt*1/4.)
            k3 = gamma3*d
            if momentum:
                d,grad, theta,y,v = dFun_mom(bA, x_t+(dt)*(-k2/2+k3),alpha,theta,y,v,gamma3)
            else:
                d,grad = dFun(x_t+(dt)*(-k2/2+k3), alpha, bA)
                
            gamma4 = betaFun(it*dt+dt*1/2.)
            k4 = gamma4*d            
            if momentum:
                d,grad, theta,y,v = dFun_mom(bA, x_t+(dt/16.)*(3*k1+9*k4),alpha,theta,y,v,gamma4)
            else:
                d,grad = dFun(x_t+(dt/16.)*(3*k1+9*k4), alpha, bA)
                
            gamma5 =  betaFun(it*dt+dt*3/4.)
            k5 = gamma5*d            
            if momentum:
                d,grad, theta,y,v = dFun_mom(bA, x_t+(dt/7.)*(-3*k1+2*k2+12*k3-12*k4+8*k5),alpha,theta,y,v,gamma5)
            else:
                d,grad = dFun(x_t+(dt/7.)*(-3*k1+2*k2+12*k3-12*k4+8*k5), alpha, bA)
                
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
            while(f(x_t+gamma_ls*step)>f(x_t) and gamma_ls > beta):
                gamma_ls = gamma_ls/2.
            gamma_ls = max(gamma_ls,beta)
            x_t = x_t + gamma_ls*step
        else: 
            x_t += dt * beta * step
        
        
        #obj[it] =f(x_t)
        trace[it] = g_t
        if it == 0: oracles[0] = oracle
        else: oracles[it] = oracles[it-1]+oracle
        
    plt.figure(100)
    plt.plot(x_t)
    return x_t,oracles,trace,obj
