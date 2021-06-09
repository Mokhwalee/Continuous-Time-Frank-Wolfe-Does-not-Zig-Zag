##### Figure 7 script file #####
'''
Quadratic functions Frank Wolfe
Figure 7
'''

from quadratic_functions import *

sola = [[] for k in range(5)]
solb = [[] for k in range(5)]
for trial in range(10):
    n,m,sparsity = 500, 100, 0.1 # rows, columns, matrix sparsity
    alpha = 5000  # L1 norm bound

    n_samples = n
    n_features = m
    A = np.random.randn(n_samples,n_features)
    x_gt = np.random.randn(n_features)
    for j in range(len(x_gt)):
        if np.random.rand(1) > sparsity:
            x_gt[j] = 0
    noise = np.random.normal(0,0.05,n_samples) ## Guassian noise
    b = x_gt.T.dot(A.T) + noise

    T=5000
    i = T

    sol = FW(A,b,m,alpha, i, T=T, n=i, disc_type ='FE',line_search = False)
    sola[0].append(sol)
    sol = FW(A,b,m,alpha, i, T=T, n=i, disc_type ='FE',line_search = True)
    solb[0].append(sol)
    

    sol = FW(A,b,m,alpha, i, T=T, n=i, disc_type ='midpoint',line_search = False)
    sola[1].append(sol)
    sol = FW(A,b,m,alpha, i, T=T, n=i, disc_type ='midpoint',line_search = True)
    solb[1].append(sol)

    sol = FW(A,b,m,alpha, i, T=T, n=i, disc_type ='rk44',line_search = False)
    sola[2].append(sol)
    sol = FW(A,b,m,alpha, i, T=T, n=i, disc_type ='rk44',line_search = True)
    solb[2].append(sol)

    sol = FW(A,b,m,alpha, i, T=T, n=i, disc_type ='rk4',line_search = False)
    sola[3].append(sol)
    sol = FW(A,b,m,alpha, i, T=T, n=i, disc_type ='rk4',line_search = True)
    solb[3].append(sol)

    sol = FW(A,b,m,alpha, i, T=T, n=i, disc_type ='rk5',line_search = False)
    sola[4].append(sol)
    sol = FW(A,b,m,alpha, i, T=T, n=i, disc_type ='rk5',line_search = True)
    solb[4].append(sol)

#############################################################################

sola_plot = [[] for k in range(5)]

for k in range(5):
   
    s = np.vstack([s[2] for s in sola[k]])
    
    sola_plot[k].append(np.mean(s,axis=0))
    sola_plot[k].append(np.min(s,axis=0))
    sola_plot[k].append(np.max(s,axis=0))
solb_plot = [[] for k in range(5)]
for k in range(5):
    s = np.vstack([s[2] for s in solb[k]])
    solb_plot[k].append(np.mean(s,axis=0))
    solb_plot[k].append(np.min(s,axis=0))
    solb_plot[k].append(np.max(s,axis=0))

#############################################################################

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

plt.figure(figsize=(9,3))

plt.subplot(1,2,1)

xxx = range(len(sola_plot[0][1]))

plt.plot(xxx,sola_plot[0][0],label="FW")
plt.fill_between(xxx,sola_plot[0][1],sola_plot[0][2],alpha=.5)
plt.plot(xxx,sola_plot[1][2],label="FW-MD")
plt.fill_between(xxx,sola_plot[1][1],sola_plot[1][2],alpha=.5)
plt.plot(xxx,sola_plot[2][0],label="FW-RK44")
plt.fill_between(xxx,sola_plot[2][1],sola_plot[2][2],alpha=.5)
plt.plot(xxx,sola_plot[3][0],label="FW-RK4")
plt.fill_between(xxx,sola_plot[3][1],sola_plot[3][2],alpha=.5)
plt.plot(xxx,sola_plot[4][0],label="FW_RK5")
plt.fill_between(xxx,sola_plot[4][1],sola_plot[4][2],alpha=.5)

plt.subplot(1,2,2)

xxx = sola[0][0][1]
plt.plot(xxx[1:],sola_plot[0][0][1:],label="FW")
plt.fill_between(xxx[1:],sola_plot[0][1][1:],sola_plot[0][2][1:],alpha=.5)

xxx = sola[1][0][1]
plt.plot(xxx[1:],sola_plot[1][2][1:],label="FW-MD")
plt.fill_between(xxx[1:],sola_plot[1][1][1:],sola_plot[1][2][1:],alpha=.5)

xxx = sola[2][0][1]
plt.plot(xxx[1:],sola_plot[2][0][1:],label="FW-RK44")
plt.fill_between(xxx[1:],sola_plot[2][1][1:],sola_plot[2][2][1:],alpha=.5)

xxx = sola[3][0][1]
plt.plot(xxx[1:],sola_plot[3][0][1:],label="FW-RK4")
plt.fill_between(xxx[1:],sola_plot[3][1][1:],sola_plot[3][2][1:],alpha=.5)

xxx = sola[4][0][1]
plt.plot(xxx[1:],sola_plot[4][0][1:],label="FW_RK5")
plt.fill_between(xxx[1:],sola_plot[4][1][1:],sola_plot[4][2][1:],alpha=.5)

#############################################################################

plt.subplot(1,2,1)
xxx = range(len(solb_plot[0][1]))

plt.plot(xxx,solb_plot[0][0],label="FW",color=colors[0],linestyle="dashed")
plt.fill_between(xxx,solb_plot[0][1],solb_plot[0][2],alpha=.3,color=colors[0])
plt.plot(xxx,solb_plot[1][2],label="FW-MD",color=colors[1],linestyle="dashed")
plt.fill_between(xxx,solb_plot[1][1],solb_plot[1][2],alpha=.3,color=colors[1])
plt.plot(xxx,solb_plot[2][0],label="FW-RK44",color=colors[2],linestyle="dashed")
plt.fill_between(xxx,solb_plot[2][1],solb_plot[2][2],alpha=.3,color=colors[2])
plt.plot(xxx,solb_plot[3][0],label="FW-RK4",color=colors[3],linestyle="dashed")
plt.fill_between(xxx,solb_plot[3][1],solb_plot[3][2],alpha=.3,color=colors[3])
plt.plot(xxx,solb_plot[4][0],label="FW_RK5",color=colors[4],linestyle="dashed")
plt.fill_between(xxx,solb_plot[4][1],solb_plot[4][2],alpha=.3,color=colors[4])

plt.subplot(1,2,2)

xxx = solb[0][0][1]
plt.plot(xxx[1:],solb_plot[0][0][1:],label="FW (L)",linestyle="dashed",color=colors[0])
plt.fill_between(xxx[1:],solb_plot[0][1][1:],solb_plot[0][2][1:],alpha=.3,color=colors[0])

xxx = solb[1][0][1]
plt.plot(xxx[1:],solb_plot[1][2][1:],label="FW-MD (L)",linestyle="dashed",color=colors[1])
plt.fill_between(xxx[1:],solb_plot[1][1][1:],solb_plot[1][2][1:],alpha=.3,color=colors[1])

xxx = solb[2][0][1]
plt.plot(xxx[1:],solb_plot[2][0][1:],label="FW-RK44 (L)",linestyle="dashed",color=colors[2])
plt.fill_between(xxx[1:],solb_plot[2][1][1:],solb_plot[2][2][1:],alpha=.3,color=colors[2])

xxx = solb[3][0][1]
plt.plot(xxx[1:],solb_plot[3][0][1:],label="FW-RK4 (L)",linestyle="dashed",color=colors[3])
plt.fill_between(xxx[1:],solb_plot[3][1][1:],solb_plot[3][2][1:],alpha=.3,color=colors[3])

xxx = solb[4][0][1]
plt.plot(xxx[1:],solb_plot[4][0][1:],label="FW_RK5 (L)",linestyle="dashed",color=colors[4])
plt.fill_between(xxx[1:],solb_plot[4][1][1:],solb_plot[4][2][1:],alpha=.3,color=colors[4])

plt.subplot(1,2,1)
plt.yscale('log')
plt.xscale('log')
plt.xlabel('Iterations')
plt.ylabel('Gap')
plt.subplot(1,2,2)

plt.yscale('log')
plt.xscale('log')
plt.xlabel('Gradient/LMO calls')
plt.ylabel('Gap')
plt.legend(bbox_to_anchor=(1.8, 1), loc='upper right', ncol=1)

plt.tight_layout()
plt.savefig('quadratic1.png')
plt.savefig('quadratic1.eps')