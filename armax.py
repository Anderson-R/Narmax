# system identification - UnB
# Anderson Rodrigues - makiander.abr@gmail.com

from libs.sysid import sysid as sd
import numpy as np
import plotly.graph_objects as go


# creates the sysid object
sysid = sd()

# allows reproducibility
np.random.seed(42)

# model parameters
na = 4
nb = 3
nc = 2
p = 1 + max(na, nb, nc)
sd_noise = 1e-2

# generate input-output data ----------------------------------------------
N = 1000  # number of samples
cutoff = 0.1  # normalized frequency cutoff
th = np.array([0.3, 0.5, 0.1, 0.4, 0.2, 0.32, 0.1])
# create input signal
ue = sysid.multisine(N, cutoff)
uv = sysid.randnoise(N, cutoff)

ye = np.zeros((N,))
yv = np.zeros((N,))

for k in range(p, N):
    phie = np.array([-ye[k-1], -ye[k-2], -ye[k-3], -ye[k-4],
                     ue[k-1], ue[k-2], ue[k-3]])
    phiv = np.array([-yv[k-1], -yv[k-2], -yv[k-3], -yv[k-4],
                     uv[k-1], uv[k-2], uv[k-3]])
    ye[k] = np.matmul(phie, th)
    yv[k] = np.matmul(phiv, th)

yeor = ye
# ye = rnorm(N,mean=ye,sd=sd_noise)
ye = np.random.normal(loc=ye, scale=sd_noise, size=N)
yvor = yv
yv = np.random.normal(loc=yv, scale=sd_noise, size=N)

Phie = sysid.regMatrix_ARX(ye, ue, na, nb, p)
Ye = sysid.targetVec(ye, p)

Phiv = sysid.regMatrix_ARX(yv, uv, na, nb, p)
Yv = sysid.targetVec(yv, p)

Phie_np = Phie.to_numpy()
Phiv_np = Phiv.to_numpy()
Ye_np = Ye.to_numpy()
Yv_np = Yv.to_numpy()

# estimate parameters -----------------------------------------------------
niter = 10
Th_ARMAX_hat = np.zeros((na+nb+nc, niter))
th_ARX_hat = np.matmul(np.linalg.pinv(Phie_np), Ye_np)
th_ARX_hat0 = th_ARX_hat
th_ARMAX_hat0 = np.append(th_ARX_hat0, np.array([0 for x in range(nc)]))
# ee_s1 = c(rep(0,p-1),Ye - (Phie %*% th_ARX_hat))
ee_s1 = np.append(np.array([0 for x in range(p)]),
                  (Ye_np - (np.matmul(Phie_np, th_ARX_hat))))
dlt1 = np.array([0 for x in range(niter)])
dlt2 = np.array([0 for x in range(niter)])

for i in range(niter):
    Phie_ext = sysid.regMatrix_MA(ye, ue, ee_s1, na, nb, nc, p).to_numpy()

    th_ARMAX_hat = np.matmul(np.linalg.pinv(Phie_ext), Ye_np)

    # --- stop conditions
    dlt1[i] = np.sqrt(np.sum((th_ARMAX_hat - th_ARMAX_hat0)**2))
    th_ARMAX_hat0 = th_ARMAX_hat

    # calculate error and pad zeros for the initial conditions
    ee_s = ee_s1
    # ee_s1 = c(rep(0,p-1),Ye - (Phie_ext %*% th_ARMAX_hat)[,])
    ee_s1 = np.append(np.array([0 for x in range(p)]),
                      (Ye_np - (np.matmul(Phie_ext, th_ARMAX_hat))))
    dlt2[i] = np.sqrt(np.sum((ee_s1 - ee_s)**2))

    # save estimated vectors
    Th_ARMAX_hat[:, i] = th_ARMAX_hat.reshape((th_ARMAX_hat.shape[0],))
    th_ARX_hat = th_ARMAX_hat[:(na+nb)]
print(dlt1, dlt2)
fig = go.Figure()
fig.add_trace(go.Scatter(mode="markers",
                         x=[x for x in range(dlt1.size)],
                         y=dlt1,
                         name="armax theta convergence (log scale)"))
fig.update_yaxes(type="log")
fig.write_html("graphs/armax/dlt1.html")

fig = go.Figure()
fig.add_trace(go.Scatter(mode="markers",
                         x=[x for x in range(dlt2.size)],
                         y=dlt2,
                         name="armax error convergence (log scale)"))
fig.update_yaxes(type="log")
fig.write_html("graphs/armax/dlt2.html")

# calculate predictions ---------------------------------------------------
ye_osa = sysid.calcOSA_ARMAX(ye, ue, na, nb, nc, p, th_ARMAX_hat)
ye_osa = np.append(ye[:ye.size - ye_osa.size], ye_osa)

ye_fr = sysid.calcFR_ARX(ye, ue, na, nb, p, th_ARX_hat)
ye_fr = np.append(ye[:ye.size - ye_fr.size], ye_fr)

yv_osa = sysid.calcOSA_ARMAX(yv, uv, na, nb, nc, p, th_ARMAX_hat)
yv_osa = np.append(ye[:ye.size - yv_osa.size], yv_osa)

yv_fr = sysid.calcFR_ARX(yv, uv, na, nb, p, th_ARX_hat)
yv_fr = np.append(ye[:ye.size - yv_fr.size], yv_fr)

# prediction performance
R2e_osa = sysid.calcR2(Ye_np, ye_osa)
R2e_fr = sysid.calcR2(Ye_np, ye_fr)
R2v_osa = sysid.calcR2(Yv_np, yv_osa)
R2v_fr = sysid.calcR2(Yv_np, yv_fr)

# Plots
time = np.array([x for x in range(N)])
# ye, ye_osa e ye_fr
fig = go.Figure()
fig.add_trace(go.Scatter(y=ye, x=time,
              mode='lines',
              name='ye'))
fig.add_trace(go.Scatter(y=ye_osa, x=time,
              mode='lines',
              name='ye_osa'))
fig.add_trace(go.Scatter(y=ye_fr, x=time,
              mode='lines',
              name='ye_fr'))
fig.write_html("graphs/armax/yTraining.html")

# yv, yv_osa e yv_fr
fig = go.Figure()
fig.add_trace(go.Scatter(y=yv, x=time,
              mode='lines',
              name='yv'))
fig.add_trace(go.Scatter(y=yv_osa, x=time,
              mode='lines',
              name='yv_osa'))
fig.add_trace(go.Scatter(y=yv_fr, x=time,
              mode='lines',
              name='yv_fr'))
fig.write_html("graphs/armax/yValidation.html")

# ee_osa e ee_fr
ee_osa = (ye - ye_osa)
ee_fr = (ye - ye_fr)
fig = go.Figure()
fig.add_trace(go.Scatter(y=ee_osa, x=time,
              mode='lines',
              name='ee_osa'))
fig.add_trace(go.Scatter(y=ee_fr, x=time,
              mode='lines',
              name='ee_fr'))
fig.write_html("graphs/armax/ee_e.html")

# ev_osa e ev_fr
ev_osa = (yv - yv_osa)
ev_fr = (yv - yv_fr)
fig = go.Figure()
fig.add_trace(go.Scatter(y=ev_osa, x=time,
              mode='lines',
              name='ev_osa'))
fig.add_trace(go.Scatter(y=ee_fr, x=time,
              mode='lines',
              name='ee_fr'))
fig.write_html("graphs/armax/ee_v.html")
