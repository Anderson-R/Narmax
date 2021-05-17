# system identification - UnB
# Anderson Rodrigues - makiander.abr@gmail.com

# example 3.3 - billings 2013

import numpy as np
import libs.sysid as si

sysid = si.sysid()

# table 3.1
Mat = np.matrix([[9, -5, 5, -1.53, 9.08],
                 [1, -1, 8, -0.39, 7.87],
                 [2, -5, 6, -3.26, 3.01],
                 [8, -2, 0, 0.36, 5.98],
                 [0, 0, 9, 0.13, 9.05]])

P = Mat[:, 0:4]
Y = Mat[:, 4]

# ordinary least squares solution
# see text in example 3.3, below eq. 3.22
th_ls = np.matmul(np.linalg.pinv(P), Y)

# orthogonal least squares

# -- remove the predictors here

print('Executing for example 3.3 in billings 2013 book')
print('Insert which line (1, 2 or 3) in table 3.2 you want to check')
a = 2  # int(input())
i = [2, 0, 1, 3]
P = P[:, i]
if a == 1:
    P = P[:, 0:2]
elif a == 2:
    P = P[:, 0:3]
elif a > 3 or a < 1:
    exit("wrong value imputed")

niter = P.shape[1]

# out = CGS(P) # classical Gram-Schmidt
A, W = sysid.MGS(P)  # modified Gram-Schmidt

# print(W)
# print(A)

Alpha = np.matmul(W.T, W)

g = np.zeros(niter)  # rep(0,niter)
Y_1 = Y.reshape(Y.shape[0])
for i in range(niter):
    # (Y %*% W[,i]) / (W[,i] %*% W[,i])
    g[i] = (np.matmul(Y_1, W[:, i]) /
            np.matmul(W[:, i], W[:, i]))

SA = np.linalg.solve(Alpha, np.diag([1 for x in range(Alpha.shape[1])]))
g2 = np.matmul(np.matmul(SA, W.T), Y)

ERR = np.zeros(niter)  # rep(0,niter)
for i in range(niter):
    # ( (Y %*% W[,i])^2 )/ ( (Y %*% Y) * (W[,i] %*% W[,i]) )
    ERR[i] = ((np.matmul(Y_1, W[:, i])**2) /
              (np.matmul(Y_1, Y) * np.matmul(W[:, i], W[:, i])))

ESR = 1 - sum(ERR)

th_OLS = np.linalg.solve(A, g)

print('OLS estimated parameters')
print(th_OLS)
print('ERR estimated parameters')
print(ERR)
print('ESR')
print(ESR)
