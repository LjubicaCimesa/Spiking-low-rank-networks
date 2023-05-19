from Rate_INPUTS_C import *
import matplotlib.pyplot as plt
import matplotlib.colorbar
from mpl_toolkits.mplot3d import axes3d
import pickle
import numpy as np
import matplotlib.gridspec as gridspec


filename = 'Results_low_dim'

plt.rcParams['font.size'] = 24
plt.rcParams["font.family"] = "serif"
plt.rcParams['lines.linewidth'] =2
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False


H = 3
W = 5
# W = 7.2 the one at overleaf in the moment

plt.rcParams['figure.figsize'] = W, H
color_a = 'steelblue'
color_o = 'gray'
color_e = 'purple'

redo_rate = False
redo_spikes = False
ind_c = 0 #for c approximately 0.7
# c_arr_spikes = np.linspace(0,0.2,11)
c_arr_rates = np.linspace(0, 0.5, 10)
c_arr_rates = np.asarray([0.5+0.5])
N = 1000

res_e = rate_C(c_array = c_arr_rates, inp_D='const', N_neurons = N, seed_val = 10)
res_a = rate_C(c_array = c_arr_rates, inp_D='along', N_neurons = N, seed_val = 1)
res_o = rate_C(c_array = c_arr_rates, inp_D='orth', N_neurons = N, seed_val = 10)


t1 = 0.5
t2 = 1.5
t_tr = 0.5
t_arr = np.linspace(0, 5, 5000)
# seed 4 za orth
# t = spikes_res_along['time']
t = np.linspace(0,5,5000)
np.random.seed(10)
ind1 = int(t1*1000)
ind2 = int(t2*1000)
ind_tr = int(t_tr*1000)
# ======================= activity ==========================


f1 = plt.figure(figsize= (5,2.5))
f2 = plt.figure(figsize= (5,2.5))
f3 = plt.figure(figsize=(5,2.5))
f4 = plt.figure()
f5 = plt.figure()
f6 = plt.figure()
f7 = plt.figure()
f8 = plt.figure()
f9 = plt.figure()
f10 = plt.figure()
f11 = plt.figure()
f12 = plt.figure()
f13 = plt.figure()
f14 = plt.figure()
f15 = plt.figure()
f16 = plt.figure()
f17 = plt.figure()
f18 = plt.figure()


I = np.zeros(len(t_arr))
I[1000:] = 1
plt.figure(1)
plt.plot(t_arr[ind1:ind2], I[ind1:ind2], color = color_e)
# plt.xlabel('time (s)')
# plt.ylabel(r'$u(t)$')
# ======

I = np.zeros(len(t_arr))
I[1000:] = 1
plt.figure(2)
plt.plot(t_arr[ind1:ind2], I[ind1:ind2], color = color_a)
# plt.xlabel('time (s)')
# plt.ylabel(r'$u(t)$')

# =====
I = np.zeros(len(t_arr))
I[1000:] = 1
plt.figure(3)
plt.plot(t_arr[ind1:ind2], I[ind1:ind2], color = color_o)
# plt.xlabel('time (s)')
# plt.ylabel(r'$u(t)$')


#  ===========  ACTIVITY ===========

N_r = 10
N_ind = np.random.randint(0, N, N_r)

# ==========
plt.figure(4)
plt.plot(t_arr[ind1:ind2], phi_shif_tanh(res_e['x_all_shif_c'][0])[ind1:ind2,N_ind],  color = color_e)
# plt.xlabel('time (s)')
# plt.ylabel(r'$x$')
# ax6.set_ylim([-1,1])
plt.yticks([0,1,2])

# =======
plt.figure(5)
plt.plot(t_arr[ind1:ind2], phi_shif_tanh(res_a['x_all_shif_c'][0])[ind1:ind2,N_ind],  color = color_a)
# plt.xlabel('time (s)')
# plt.ylabel(r'$x$')
plt.yticks([0,1,2])

N_ind = np.random.randint(0, N, N_r)

# ======
plt.figure(6)
plt.plot(t_arr[ind1:ind2], phi_shif_tanh(res_o['x_all_shif_c'][0])[ind1:ind2,N_ind],  color = color_o)
# plt.xlabel('time (s)')
# plt.ylabel(r'$x$')
# ax5.set_ylim([-1,1])
plt.yticks([0,1,2])

# ============  POPULATION FIRING RATE ======

plt.figure(7)
plt.plot(t_arr[ind1:ind2], np.mean(phi_shif_tanh(res_e['x_all_shif_c'][0])[ind1:ind2,:], axis = 1),  color = color_e)
# plt.xlabel('time (s)')
# plt.ylabel('pop. rate')
plt.yticks([0,1])
plt.ylim([0,2])

# ==========
plt.figure(8)
plt.plot(t_arr[ind1:ind2], np.mean(phi_shif_tanh(res_a['x_all_shif_c'][0])[ind1:ind2,:], axis = 1),  color = color_a)
# plt.xlabel('time (s)')
# plt.ylabel('pop. rate')
plt.yticks([0,1])
plt.ylim([0,2])

# ======

plt.figure(9)
plt.plot(t_arr[ind1:ind2], np.mean(phi_shif_tanh(res_o['x_all_shif_c'][0])[ind1:ind2,:], axis = 1),  color = color_o)
# plt.xlabel('time (s)')
# plt.ylabel('pop. rate')
plt.yticks([0,1])
plt.ylim([0,2])

# ======= PROJECTIONS onto (I,m) plane
plt.figure(10)
rates_e = phi_shif_tanh(res_e['x_all_shif_c'][0])

proj_I_3 = np.dot(rates_e, res_e['vector I'])/N
proj_m_3 = np.dot(rates_e, res_e['vector m'])/N
plt.plot(proj_I_3[ind_tr:], proj_m_3[ind_tr:], color = color_e)
# ax12.set_xlabel(r'$I$')
# ax12.set_ylabel(r'$m$')
# plt.xlabel('proj. I')
# plt.ylabel('proj. m')
plt.yticks([0,0.2])
plt.ylim([-0.1, 0.4])

# =====
plt.figure(11)
rates_a = phi_shif_tanh(res_a['x_all_shif_c'][0])

proj_I_1 = np.dot(rates_a, res_a['vector I'])/N
proj_m_1 = np.dot(rates_a, res_a['vector m'])/N
plt.plot(proj_I_1[ind_tr:], proj_m_1[ind_tr:], color = color_a)
# ax10.set_xlabel(r'$I$')
# ax10.set_ylabel(r'$m$')
# plt.xlabel('proj. I')
# plt.ylabel('proj. m')
# ax10.set_ylim([-2,12])
plt.yticks([0,0.2])
plt.ylim([-0.1, 0.4])

# =====
plt.figure(12)
rates_o = phi_shif_tanh(res_o['x_all_shif_c'][0])

proj_I_2 = np.dot(rates_o, res_o['vector I'])/N
proj_m_2 = np.dot(rates_o, res_o['vector m'])/N
plt.plot(proj_I_2[ind_tr:], proj_m_2[ind_tr:], color = color_o)
# ax11.set_xlabel(r'$I$')
# ax11.set_ylabel(r'$m$')
# plt.xlabel('proj. I')
# plt.ylabel('proj. m')
plt.yticks([0,0.2])
plt.ylim([-0.1, 0.4])


# ================ PCA COMPONENTS ===================
plt.figure(13)
X = (rates_e - np.mean(rates_e, axis = 0))[ind_tr:]
C = np.cov(X.transpose())  # Compute covariance matrix
C = np.dot(X.T, X) / (X.shape[0] - 1)

Lambda, eigvec = np.linalg.eig(C)
Lambda = Lambda.real
eigvec = eigvec.real

Lambda_sort = Lambda[(np.argsort(-Lambda))]
E_e = eigvec[:,np.argsort(-Lambda)]
X_prim = np.dot(X,E_e)
C_prim = np.dot(X_prim.transpose(), X_prim)

PCA_1 = []
for i in range(0,8):
    PCA_1.append(C_prim[i,i]/np.trace(C_prim))
plt.bar(np.linspace(0,8,8), PCA_1, color = color_e)
# plt.xlabel('PC number')
# plt.ylabel('std explained')
plt.yticks([0,1])

vector_g = np.ones(len(res_o['vector I']))

g_norm = vector_g/np.linalg.norm(vector_g)
I_norm_e = res_e['vector I']/np.linalg.norm(res_e['vector I'])
m_norm_e = res_e['vector m']/np.linalg.norm(res_e['vector m'])

PCA_projs_e = []

for i in range(0,2):
    PCA_g = np.dot(g_norm, E_e[:,i])

    PCA_I = np.dot(I_norm_e, E_e[:,i])

    PCA_m = np.dot(m_norm_e, E_e[:,i])

    PCA_projs_e.append([PCA_g, PCA_I, PCA_m])

PCA_projs = np.absolute(np.asarray(PCA_projs_e))
IM1 = PCA_projs.transpose()

plt.figure(14)
plt.imshow(IM1, cmap = 'Greys')
plt.colorbar()
plt.clim([0,1])

plt.xticks([0,1])
plt.yticks([])

# =======

plt.figure(15)
X = (rates_a - np.mean(rates_a, axis = 0))[ind_tr:]
C = np.cov(X.transpose())  # Compute covariance matrix
C = np.dot(X.T, X) / (X.shape[0] - 1)

Lambda, eigvec = np.linalg.eig(C)
Lambda = Lambda.real
eigvec = eigvec.real

Lambda_sort = Lambda[(np.argsort(-Lambda))]
E_a = eigvec[:,np.argsort(-Lambda)]
X_prim = np.dot(X,E_a)
C_prim = np.dot(X_prim.transpose(), X_prim)

PCA_1 = []
for i in range(0,8):
    PCA_1.append(C_prim[i,i]/np.trace(C_prim))
plt.bar(np.linspace(0,8,8), PCA_1, color = color_a)
# plt.xlabel('PC number')
# plt.ylabel('std explained')
plt.yticks([0,1])

vector_g = np.ones(len(res_a['vector I']))

g_norm = vector_g/np.linalg.norm(vector_g)
I_norm_a = res_a['vector I']/np.linalg.norm(res_a['vector I'])
m_norm_a = res_a['vector m']/np.linalg.norm(res_a['vector m'])

PCA_projs_a = []

for i in range(0,2):
    PCA_g = np.dot(g_norm, E_a[:,i])

    PCA_I = np.dot(I_norm_a, E_a[:,i])

    PCA_m = np.dot(m_norm_a, E_a[:,i])

    PCA_projs_a.append([PCA_g, PCA_I, PCA_m])

PCA_projs = np.absolute(np.asarray(PCA_projs_a))
IM2 = PCA_projs.transpose()

plt.figure(16)
plt.imshow(IM2, cmap = 'Greys')
plt.colorbar()
plt.clim([0,1])
plt.xticks([0,1])
plt.yticks([])

# ============== ORTH
plt.figure(17)
X = (rates_o - np.mean(rates_o, axis = 0))[ind_tr:]
C = np.cov(X.transpose())  # Compute covariance matrix
C = np.dot(X.T, X) / (X.shape[0] - 1)

Lambda, eigvec = np.linalg.eig(C)
Lambda = Lambda.real
eigvec = eigvec.real

Lambda_sort = Lambda[(np.argsort(-Lambda))]
E_o = eigvec[:,np.argsort(-Lambda)]
X_prim = np.dot(X,E_o)
C_prim = np.dot(X_prim.transpose(), X_prim)

PCA_1 = []
for i in range(0,8):
    PCA_1.append(C_prim[i,i]/np.trace(C_prim))
plt.bar(np.linspace(0,8,8), PCA_1, color = color_o)
plt.yticks([0,1])

vector_g = np.ones(len(res_o['vector I']))

g_norm = vector_g/np.linalg.norm(vector_g)
I_norm_o = res_o['vector I']/np.linalg.norm(res_o['vector I'])
m_norm_o = res_o['vector m']/np.linalg.norm(res_o['vector m'])

PCA_projs_o = []

for i in range(0,2):
    PCA_g = np.dot(g_norm, E_o[:,i])

    PCA_I = np.dot(I_norm_o, E_o[:,i])

    PCA_m = np.dot(m_norm_o, E_o[:,i])

    PCA_projs_o.append([PCA_g, PCA_I, PCA_m])

PCA_projs = np.absolute(np.asarray(PCA_projs_o))
IM3 = PCA_projs.transpose()

plt.figure(18)
plt.imshow(IM3, cmap = 'Greys')
plt.colorbar()
plt.clim([0,1])
plt.xticks([0,1])
plt.yticks([])


f1.set_tight_layout(True)
f2.set_tight_layout(True)
f3.set_tight_layout(True)
f4.set_tight_layout(True)
f5.set_tight_layout(True)

f6.set_tight_layout(True)
f7.set_tight_layout(True)
f8.set_tight_layout(True)
f9.set_tight_layout(True)
f10.set_tight_layout(True)


f11.set_tight_layout(True)
f12.set_tight_layout(True)
f13.set_tight_layout(True)
f14.set_tight_layout(True)
f15.set_tight_layout(True)

f16.set_tight_layout(True)
f17.set_tight_layout(True)
f18.set_tight_layout(True)

