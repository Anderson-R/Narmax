# system identification - UnB
# Anderson Rodrigues - makiander.abr@gmail.com

# example 3.6 - billings 2013

import numpy as np
import libs.sysid as si

# code begins

# generate simulation data ------------------------------------------------
N = 200
# u = runif(N,min = -1,max = 1)
u = np.random.uniform(-1, 1, N)
# e = rnorm(N,mean = 0,sd=0.1)
e = np.random.normal(loc=0, scale=0.1, size=N)
# y = rep(0,length(u))
y = np.zeros(u.size)
for k in range(3, N):
    y[k] = -0.605*y[k-1] - 0.163*y[k-2]**2 + 0.588*u[k-1] - 0.24*u[k-2] + e[k]

# model parameters --------------------------------------------------------
rho = 0.04
nu = 2
ny = 2
ne = 0
L = 3
n = nu + ny + ne
p = max(nu, ny, ne) + 1
sysid = si.sysid()

# create regression and target matrices -----------------------------------
P = sysid.regMatNARX(u, y, nu, ny, L)
Y = sysid.targetVec(y, p)

M = P.shape[1]
NP = P.shape[0]
