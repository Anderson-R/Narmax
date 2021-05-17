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
p = 1 + np.maximum(na, nb)
sd_noise = 1e-1

# generate input-output data ----------------------------------------------
N = 1000  # number of samples
cutoff = 0.1  # normalized frequency cutoff

# create input signal
ue = sysid.multisine(N, cutoff)
uv = sysid.randnoise(N, cutoff)

# M_spec(ue,'ue')
# M_spec(uv,'uv')

ye = np.zeros((N,))
yv = np.zeros((N,))

for k in range(p, N):
    ye[k] = -0.3*ye[k-1] - 0.5*ye[k-2] - 0.1*ye[k-3] - 0.4*ye[k-4] + \
        0.2*ue[k-1] + 0.32*ue[k-2] + 0.1*ue[k-3]
    yv[k] = -0.3*yv[k-1] - 0.5*yv[k-2] - 0.1*yv[k-3] - 0.4*yv[k-4] + \
        0.2*uv[k-1] + 0.32*uv[k-2] + 0.1*uv[k-3]

yeor = ye
# rnorm(N,mean=ye,sd=sd_noise)
ye = np.random.normal(loc=ye, scale=sd_noise, size=N)
yvor = yv
# rnorm(N,mean=yv,sd=sd_noise)
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
th_hat = np.matmul(np.linalg.pinv(Phie_np), Ye_np)

# calculate predictions ---------------------------------------------------
# training input
ye_osa = np.matmul(Phie_np, th_hat)
ye_osa = ye_osa.reshape((ye_osa.shape[0],))
ye_osa = np.append(ye[:ye.size - ye_osa.size], ye_osa)

ye_fr = sysid.calcFR_ARX(ye, ue, na, nb, p, th_hat)
ye_fr = np.append(ye[:ye.size - ye_fr.size], ye_fr)

# validation input
yv_osa = np.matmul(Phiv_np, th_hat)
yv_osa = yv_osa.reshape((yv_osa.shape[0],))
yv_osa = np.append(yv[:yv.size - yv_osa.size], yv_osa)

yv_fr = sysid.calcFR_ARX(yv, uv, na, nb, p, th_hat)
yv_fr = np.append(yv[:yv.size - yv_fr.size], yv_fr)

# Plots
time = np.array([x for x in range(N)])
# ue e uv
fig = go.Figure()
fig.add_trace(go.Scatter(y=ue, x=time,
              mode='lines',
              name='ue'))
fig.add_trace(go.Scatter(y=uv, x=time,
              mode='lines',
              name='uv'))

# fig.show()
fig.write_html("graphs/arx/u.html")

# ye e yeor
fig = go.Figure()
fig.add_trace(go.Scatter(y=ye, x=time,
              mode='lines',
              name='ye'))
fig.add_trace(go.Scatter(y=yeor, x=time,
              mode='lines',
              name='yeor'))

# fig.show()
fig.write_html("graphs/arx/ye.html")

# yv e yvor
fig = go.Figure()
fig.add_trace(go.Scatter(y=yv, x=time,
              mode='lines',
              name='yv'))
fig.add_trace(go.Scatter(y=yvor, x=time,
              mode='lines',
              name='yvor'))

# fig.show()
fig.write_html("graphs/arx/yv.html")

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

# fig.show()
fig.write_html("graphs/arx/yTraining.html")

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
fig.write_html("graphs/arx/yValidation.html")

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

# fig.show()
fig.write_html("graphs/arx/ee_e.html")

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

# fig.show()
fig.write_html("graphs/arx/ee_v.html")
