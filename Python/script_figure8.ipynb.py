##### Figure 8 script file #####
'''
Logistic functions Frank Wolfe
Figure 8
'''

import logistic_functions
from logistic_functions import *

A = pd.read_csv('gisette_data.csv',header=None)
b = pd.read_csv('gisette_labels.csv',header=None)[0].values

scaler = StandardScaler() #Standardize features by removing the mean and scaling to unit variance
A = scaler.fit_transform(A)
bA = (A.T*b).T
# print(bA, bA.shape)
m = A.shape[1]

alpha = 250.
T=200
i = T

sol1a = FW(alpha, T, i, m, i, bA, 'FE')
pickle.dump({'sol':sol1a},open('experiments_logistic/sol1a.pkl','wb'))
sol1a = pickle.load(open('experiments_logistic/sol1a.pkl','rb'))['sol']

sol1b = FW(alpha, T, i, m, i, bA, 'FE',line_search = True)
pickle.dump({'sol':sol1b},open('experiments_logistic/sol1b.pkl','wb'))
sol1b = pickle.load(open('experiments_logistic/sol1b.pkl','rb'))['sol']

sol1c = FW(alpha, T, i, m, i, bA, 'FE', momentum = True)
pickle.dump({'sol':sol1c},open('experiments_logistic/sol1c.pkl','wb'))
sol1c = pickle.load(open('experiments_logistic/sol1c.pkl','rb'))['sol']

sol2a = FW(alpha, T, i, m, i, bA, 'midpoint',line_search = False)
pickle.dump({'sol':sol2a},open('experiments_logistic/sol2a.pkl','wb'))
sol2a = pickle.load(open('experiments_logistic/sol2a.pkl','rb'))['sol']

sol2b = FW(alpha, T, i, m, i, bA, 'midpoint',line_search = True)
pickle.dump({'sol':sol2b},open('experiments_logistic/sol2b.pkl','wb'))
sol2b = pickle.load(open('experiments_logistic/sol2b.pkl','rb'))['sol']

sol2c = FW(alpha, T, i, m, i, bA, 'midpoint', momentum = True)
pickle.dump({'sol':sol2c},open('experiments_logistic/sol2c.pkl','wb'))
sol2c = pickle.load(open('experiments_logistic/sol2c.pkl','rb'))['sol']

sol3a = FW(alpha, T, i, m, i, bA, 'rk44')
pickle.dump({'sol':sol3a},open('experiments_logistic/sol3a.pkl','wb'))
sol3a = pickle.load(open('experiments_logistic/sol3a.pkl','rb'))['sol']

sol3b = FW(alpha, T, i, m, i, bA, 'rk44',line_search = True)
pickle.dump({'sol':sol3b},open('experiments_logistic/sol3b.pkl','wb'))
sol3b = pickle.load(open('experiments_logistic/sol3b.pkl','rb'))['sol']

sol3c = FW(alpha, T, i, m, i, bA, 'rk44',momentum = True)
pickle.dump({'sol':sol3c},open('experiments_logistic/sol3c.pkl','wb'))
sol3c = pickle.load(open('experiments_logistic/sol3c.pkl','rb'))['sol']

sol4a = FW(alpha, T, i, m, i, bA, 'rk4')
pickle.dump({'sol':sol4a},open('experiments_logistic/sol4a.pkl','wb'))
sol4a = pickle.load(open('experiments_logistic/sol4a.pkl','rb'))['sol']

sol4b = FW(alpha, T, i, m, i, bA, 'rk4',line_search = True)
pickle.dump({'sol':sol4b},open('experiments_logistic/sol4b.pkl','wb'))
sol4b = pickle.load(open('experiments_logistic/sol4b.pkl','rb'))['sol']


sol4c = FW(alpha, T, i, m, i, bA, 'rk4', momentum = True)
pickle.dump({'sol':sol4c},open('experiments_logistic/sol4c.pkl','wb'))
sol4c = pickle.load(open('experiments_logistic/sol4c.pkl','rb'))['sol']


sol5a = FW(alpha, T, i, m, i, bA, 'rk5')
pickle.dump({'sol':sol5a},open('experiments_logistic/sol5a.pkl','wb'))
sol5a = pickle.load(open('experiments_logistic/sol5a.pkl','rb'))['sol']

sol5b = FW(alpha, T, i, m, i, bA, 'rk5',line_search = True)
pickle.dump({'sol':sol5b},open('experiments_logistic/sol5b.pkl','wb'))
sol5b = pickle.load(open('experiments_logistic/sol5b.pkl','rb'))['sol']


sol5c = FW(alpha, T, i, m, i, bA, 'rk5', momentum = True)
pickle.dump({'sol':sol5c},open('experiments_logistic/sol5c.pkl','wb'))
sol5c = pickle.load(open('experiments_logistic/sol5c.pkl','rb'))['sol']

##########################################################################

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

plt.figure(figsize=(9,3))

plt.subplot(1,2,1)
plt.plot(sol1a[2],linestyle="-",label="FW")
plt.plot(sol2a[2],linestyle="-",label="FW-MD")
plt.plot(sol3a[2],linestyle="-",label="FW-RK44")
plt.plot(sol4a[2],linestyle="-",label="FW-RK4")
plt.plot(sol5a[2],linestyle="-",label="FW-RK5")

plt.subplot(1,2,2)
plt.plot(sol1a[1][1:],sol1a[2][1:] ,linestyle="-",label="FW")
plt.plot(sol2a[1][1:],sol2a[2][1:] ,linestyle="-",label="FW-MD")
plt.plot(sol3a[1][1:],sol3a[2][1:],linestyle="-",label="FW-RK44")
plt.plot(sol4a[1][1:],sol4a[2][1:] ,linestyle="-",label="FW-RK4")
plt.plot(sol5a[1][1:],sol5a[2][1:],linestyle="-",label="FW_RK5")


plt.subplot(1,2,1)
plt.plot(sol1c[2],linestyle="dashed",label="FW (L)",color = colors[0])
plt.plot(sol2c[2],linestyle="dashed",label="FW-MD (L)",color = colors[1])
plt.plot(sol3c[2],linestyle="dashed",label="FW-RK44 (L)",color = colors[2])
plt.plot(sol4c[2],linestyle="dashed",label="FW-RK4 (L)",color = colors[3])
plt.plot(sol5c[2],linestyle="dashed",label="FW-RK5 (L)",color = colors[4])


plt.subplot(1,2,2)
plt.plot(sol1c[1][1:],sol1c[2][1:],linestyle="dashed",label="FW (M)",color = colors[0])
plt.plot(sol2c[1][1:],sol2c[2][1:] ,linestyle="dashed",label="FW-MD (M)",color = colors[1])
plt.plot(sol3c[1][1:],sol3c[2][1:],linestyle="dashed",label="FW-RK44 (M)",color = colors[2])
plt.plot(sol4c[1][1:],sol4c[2][1:] ,linestyle="dashed",label="FW-RK4 (M)",color = colors[3])
plt.plot(sol5c[1][1:],sol5c[2][1:],linestyle="dashed",label="FW-RK5 (M)",color = colors[4])

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
plt.savefig('logistic.png')
plt.savefig('logistic.eps')


