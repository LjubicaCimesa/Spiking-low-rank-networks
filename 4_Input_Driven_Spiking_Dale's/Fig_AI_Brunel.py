from Spiking_rank_one_INPUTS_C import *
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle
import matplotlib.gridspec as gridspec

def firing_rates(spikes_t_all = np.zeros(0), spikes_i_all = np.zeros(0), time_array = np.zeros(0), tau_s = 100*ms, N_total = 1000):

    n_steps_rates = len(time_array)
    aver_rates = np.zeros((n_steps_rates, N_total))
    N_trials = len(spikes_t_all)
    print(N_trials)
    dt = 1*ms
    sd_firing = []
    rates_all = []
    for trial in range(0, N_trials):
        print('Iteracija: '+str(trial))
        spike_t = spikes_t_all[trial]*second
        spike_i = spikes_i_all[trial]
        inst_rates = np.zeros((len(time_array), N_total))

        for k in range(n_steps_rates-1):
            t_k = time_array[k]
            t_k_plus_one = time_array[k+1]
            #list of neurons that spiked between t_k and t_k_plus_one
            spiking_neurons = spike_i[(t_k <= spike_t) & (spike_t<t_k_plus_one)]
            # print(spiking_neurons)
            for i in range(N_total):
                if i in spiking_neurons : ######################§#############################
                    inst_rates[k+1][i] = inst_rates[k][i]*(1-dt/tau_s) + dt/tau_s
                else: ######################################################################
                    inst_rates[k+1][i] = inst_rates[k][i]*(1-dt/tau_s)
                # if (i==0): print(inst_rates[k+1][0])

        inst_rates = 1000*inst_rates
        rates_all.append(inst_rates)
        # inst_rates_all.append(inst_rates)
        aver_rates = (aver_rates*trial + inst_rates)/(trial+1)
        pop_rate = np.mean(aver_rates,axis=1)

        sd_firing.append(np.std(pop_rate))

    return [rates_all, aver_rates, sd_firing]

H = 3
W = 5
# W = 7.2 the one at overleaf in the moment

plt.rcParams['figure.figsize'] = W,H
plt.rcParams['font.size'] = 24
# plt.rcParams["font.family"] = "serif"
# "font.sans-serif": ["Helvetica"] an example from matplotlib documentation
plt.rcParams['lines.linewidth'] = 2
# 8 fmar
# plt.rcParams['figure.figsize'] = 4,4
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

# ======== Firing rates ========
factor_mn = 1

I_e = 45*mV
I_o = 1*mV*250
sigma = 0.1
mu_noise = 40

J_syn = 0.1*mV

Sm = 2
Sn = 2*0.01

g = 5
# =============== filter time constant ==========
tau_D_min = 1.5
tau_D_max = 1.5

N_trials = 1
#  for 10000 neurons,  5 trials
syn_Del = 'const'
c_arr = np.asarray([0.14])
c_arr = np.asarray([0.5])
#  possible for inp_D which is input direction : 'const', 'along' or 'rand'
color_a = 'steelblue'
color_o = 'gray'
color_e = 'purple'

N = 12500
C_sparsity = int(0.1*N)
# C=C_sparsity,
# set generate_results to True until all the code is run. After that, the results data will be saved in the file
generate_res = False
tau_short = 1*ms
tau_long = 100*ms
raster_trials_plot = [0, 0, 0]

t_run = 2*second
t_arr = np.linspace(0, t_run, int(t_run*1000))*second
tau_all = np.asarray([1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])*ms
tau_all = np.asarray([100])*ms


filename='Results'

if generate_res:

    spikes_res_along = SPIKES_C(Sm = Sm, Sn = Sn, J_value =J_syn, factor_mn = factor_mn, I_ampl = I_o, ind_c = 0, c_array = c_arr,
                    mu_noise = mu_noise*mV, sigma_noise = sigma*mV, nmb_trials = N_trials, N_total= N, C=C_sparsity,  syn_D = syn_Del,
                    inp_D = 'along', tau_s = tau_long, tau_min = tau_D_min, tau_max = tau_D_max, inhib_g = g, seed_start=1, t_run = t_run )

    spikes_res_orth = SPIKES_C(Sm = Sm, Sn = Sn, J_value = J_syn, factor_mn = factor_mn, I_ampl =  I_o, ind_c = 0, c_array = c_arr,
                    mu_noise = mu_noise*mV, sigma_noise = sigma*mV, nmb_trials = N_trials, N_total= N, C=C_sparsity, syn_D = syn_Del,
                    inp_D = 'orth', tau_s= tau_long, tau_min = tau_D_min, tau_max = tau_D_max, inhib_g = g, seed_start=5, t_run = t_run)

    spikes_res_equal = SPIKES_C(Sm = Sm, Sn = Sn, J_value = J_syn, factor_mn = factor_mn, I_ampl = I_e, ind_c = 0, c_array = c_arr,
                    mu_noise = mu_noise*mV, sigma_noise = sigma*mV, nmb_trials = N_trials, N_total= N, C=C_sparsity, syn_D = syn_Del,
                    inp_D = 'const', tau_s= tau_long, tau_min = tau_D_min, tau_max = tau_D_max, inhib_g = g, seed_start=3, t_run = t_run)


    f = open(filename, 'wb')
    I_dir = ['along', 'orth', 'equal']
    spikes_res = [spikes_res_along, spikes_res_orth, spikes_res_equal]
    for d in range(0,3):

        S = {}
        S['I_'+I_dir[d]+'; J='+str(J_syn)+'; spikes_t'] = spikes_res[d]['spike_times']
        S['I_'+I_dir[d]+'; J='+str(J_syn)+'; spikes_i'] = spikes_res[d]['neuron ind.']
        S['I_'+I_dir[d]+'; J='+str(J_syn)+'; vector_I'] = spikes_res[d]['vector_I']
        S['I_'+I_dir[d]+'; J='+str(J_syn)+'; vector_m'] = spikes_res[d]['vector_m']
        S['I_'+I_dir[d]+'; J='+str(J_syn)+'; vector_n'] = spikes_res[d]['vector_n']
        S['I_'+I_dir[d]+'; J='+str(J_syn)+'; input_I'] = spikes_res[d]['I input'][0]
        S['I_'+I_dir[d]+'; J='+str(J_syn)+'; t_arr'] = spikes_res[d]['t_arr']
        S['I_'+I_dir[d]+'; J='+str(J_syn)+'; J_matrix'] = spikes_res[d]['J_matrix']
        S['I_'+I_dir[d]+'; J='+str(J_syn)+'; EI_matrix'] = spikes_res[d]['EI_matrix']
        S['I_'+I_dir[d]+'; J='+str(J_syn)+'; rank1_matrix'] = spikes_res[d]['rank1_matrix']

        pickle.dump(S, f)

    f.close()


    spikes_t_a = spikes_res_along['spike_times']
    spikes_i_a = spikes_res_along['neuron ind.']
    print('soikes res along t_arr:')
    print(spikes_res_along['t_arr'])
    R1 = {}
    rates_1_tau_all = []

    for tau in tau_all:
        rates_1_tau = firing_rates(spikes_t_a, spikes_i_a, spikes_res_along['t_arr'], tau, N_total=N)
        rates_1_tau_all.append(rates_1_tau)


    spikes_t_o = spikes_res_orth['spike_times']
    spikes_i_o = spikes_res_orth['neuron ind.']

    rates_2_tau_all = []
    for tau in tau_all:
        rates_2_tau = firing_rates(spikes_t_o, spikes_i_o, spikes_res_orth['t_arr'], tau, N_total=N)
        rates_2_tau_all.append(rates_2_tau)


    spikes_t_e = spikes_res_equal['spike_times']
    spikes_i_e = spikes_res_equal['neuron ind.']

    rates_3_tau_all = []
    for tau in tau_all:
        rates_3_tau = firing_rates(spikes_t_e, spikes_i_e, spikes_res_equal['t_arr'], tau, N_total=N)
        rates_3_tau_all.append(rates_3_tau)

    rates_all = [rates_1_tau_all, rates_2_tau_all, rates_3_tau_all]


    f = open(filename, 'ab')

    for k in range(0,3):
        R={}
        R['rates_' +I_dir[k]+ '; J='+str(J_syn)+'; tau_all'] = rates_all[k]
        pickle.dump(R,f)
    f.close()

t1 = 0.5
t_tr = 0.5
t2 = 2


tau_choice = 100*ms

# t = spikes_res_along['time']
t=np.linspace(0,5,5000)
np.random.seed(1)
ind1 = int(t1*1000)
ind2 = int(t2*1000)
ind_tr = int(t_tr*1000)

t1_s = 0.5
t2_s = 1.5
t1_s = 0.9
t2_s = 1.1
ind1 = int(t1_s*1000)
ind2 = int(t2_s*1000)
# ======================= RASTERS ==========================


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


f = open(filename, 'rb')
S1 = pickle.load(f)
spikes_t_a = S1['I_along; J='+str(J_syn)+'; spikes_t']
spikes_i_a = S1['I_along; J='+str(J_syn)+'; spikes_i']
vector_I_a = S1['I_along; J='+str(J_syn)+'; vector_I']
vector_m_a = S1['I_along; J='+str(J_syn)+'; vector_m']
vector_n_a = S1['I_along; J='+str(J_syn)+'; vector_n']
t_arr_a = S1['I_along; J='+str(J_syn)+'; t_arr']

S2 = pickle.load(f)
spikes_t_o = S2['I_orth; J='+str(J_syn)+'; spikes_t']
spikes_i_o = S2['I_orth; J='+str(J_syn)+'; spikes_i']
vector_I_o = S2['I_orth; J='+str(J_syn)+'; vector_I']
vector_m_o = S2['I_orth; J='+str(J_syn)+'; vector_m']
vector_n_o = S2['I_orth; J='+str(J_syn)+'; vector_n']
t_arr_o = S2['I_orth; J='+str(J_syn)+'; t_arr']

S3 = pickle.load(f)
spikes_t_e = S3['I_equal; J='+str(J_syn)+'; spikes_t']
spikes_i_e = S3['I_equal; J='+str(J_syn)+'; spikes_i']
vector_I_e = S3['I_equal; J='+str(J_syn)+'; vector_I']
vector_m_e = S3['I_equal; J='+str(J_syn)+'; vector_m']
vector_n_e = S3['I_equal; J='+str(J_syn)+'; vector_n']
t_arr_e = S3['I_equal; J='+str(J_syn)+'; t_arr']

R1 = pickle.load(f)
R2 = pickle.load(f)
R3 = pickle.load(f)

f.close()
# ======================  INPUTS u(t) ======================

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

# ======================  RASTERS  ======================
N_r = 30

r2 = raster_trials_plot[2]


cond = np.logical_and(np.logical_and(spikes_t_e[r2]>t1_s, spikes_t_e[r2]<t2_s), spikes_i_e[r2]<N_r)
# ax6.eventplot(positions =spikes_t_e[r2][cond][:,np.newaxis], lineoffsets=spikes_i_e[r2][cond],
               # linewidths=0.5, linelengths=0.75,  color = color_e)
plt.figure(4)
plt.plot(spikes_t_e[r2][cond], spikes_i_e[r2][cond], '.', color = color_e)
# plt.xlabel('time (s)')
# plt.ylabel('neuron ind.')

# plt.xticks([])

# ==========
r0 = raster_trials_plot[0]
cond = np.logical_and(np.logical_and(spikes_t_a[r0]>t1_s, spikes_t_a[r0]<t2_s), spikes_i_a[r0]<N_r)
# ax4.eventplot(positions=spikes_t_a[r0][cond][:,np.newaxis], lineoffsets=spikes_i_a[r0][cond],
              # linewidths=0.5, linelengths=0.75,  color = color_a)
plt.figure(5)
plt.plot(spikes_t_a[r0][cond], spikes_i_a[r0][cond], '.', color = color_a)
# plt.xlabel('time (s)')
# plt.ylabel('neuron ind.')
# plt.xticks([])

# ======
r1 = raster_trials_plot[1]
cond = np.logical_and(np.logical_and(spikes_t_o[r1]>t1_s, spikes_t_o[r1]<t2_s), spikes_i_o[r1]<N_r)
# ax5.eventplot(positions =spikes_t_o[r1][cond][:,np.newaxis], lineoffsets=spikes_i_o[r1][cond],
               # linewidths=0.5, linelengths=0.75,  color = color_o)
plt.figure(6)
plt.plot(spikes_t_o[r1][cond], spikes_i_o[r1][cond], '.',  color = color_o)
# plt.xlabel('time (s)')
# plt.ylabel('neuron ind.')
# plt.xticks([])


# ======================  POPULATION FIRING RATES ======================
ind_rate = np.argwhere(tau_all == tau_choice)[0][0]
rates_e = R3['rates_equal; J='+str(J_syn)+'; tau_all'][ind_rate]

plt.figure(7)
plt.plot(t_arr[ind1:ind2], np.mean(rates_e[1][ind1:ind2,:], axis = 1), color = color_e)
# plt.xlabel('time (s)')
# plt.ylabel('pop. rate (Hz)')
plt.ylim([-5,80])
# plt.xticks([0.8, 1, 1.2])

# ====

rates_a = R1['rates_along; J='+str(J_syn)+'; tau_all'][ind_rate]

plt.figure(8)
plt.plot(t_arr[ind1:ind2], np.mean(rates_a[1][ind1:ind2,:], axis=1 ), color = color_a)
# plt.xlabel('time (s)')
# plt.ylabel('pop. rate (Hz)')
plt.ylim([-5,80])
# plt.xticks([0.8, 1, 1.2])


# ax7.set_ylim([30,70])
# =====
rates_o = R2['rates_orth; J='+str(J_syn)+'; tau_all'][ind_rate]


plt.figure(9)
plt.plot(t_arr[ind1:ind2], np.mean(rates_o[1][ind1:ind2,:], axis=1), color = color_o)
# plt.xlabel('time (s)')
# plt.ylabel('pop. rate (Hz)')
plt.ylim([-5,80])
# ax8.set_ylim([30,70])
# plt.xticks([0.8, 1, 1.2])


# ====================== PROJECTIONS onto (I,m) plane ======================

#  EQUAL
proj_I_e = np.dot(rates_e[0][0], vector_I_e[0])/np.linalg.norm(vector_I_e[0])
proj_m_e = np.dot(rates_e[0][0], vector_m_e[0])/np.linalg.norm(vector_m_e[0])

        # aver_rates = (aver_rates*trial + inst_rates)/(trial+1)

plt.figure(10)
plt.plot(proj_I_e[ind_tr:], proj_m_e[ind_tr:], color = color_e)

# if (J_syn == 0.5*mV):plt.xlim([0,65])
# else: plt.xlim([0, 45])

plt.ylim([-1000, 1000])

# plt.xlabel('proj. I')
# plt.ylabel('proj. m')
# plt.axis('off')


# ===========   ALONG
proj_I_a = np.dot(rates_a[0][0], vector_I_a[0])/np.linalg.norm(vector_I_a[0])
proj_m_a = np.dot(rates_a[0][0], vector_m_a[0])/np.linalg.norm(vector_m_a[0])

plt.figure(11)
plt.plot(proj_I_a[ind_tr:], proj_m_a[ind_tr:], color = color_a)
# plt.xlabel('proj. I')
# plt.ylabel('proj. m')
# plt.ylim([-1, 20])
plt.ylim([-100,3000])
# plt.axis('off')

# ==== ORTH

proj_I_o = np.dot(rates_o[0][0], vector_I_o[0])/np.linalg.norm(vector_I_o[0])
proj_m_o = np.dot(rates_o[0][0], vector_m_o[0])/np.linalg.norm(vector_m_o[0])



plt.figure(12)
plt.plot(proj_I_o[ind_tr:], proj_m_o[ind_tr:], color = color_o)
# plt.xlabel('proj. I')
# plt.ylabel('proj. m')
plt.ylim([-1000, 1000])
# plt.axis('off')


# ================ PCA COMPONENTS ===================

X = rates_e[1][ind_tr:,:] - np.mean(rates_e[1][ind_tr:,:], axis = 0)

C = np.cov(X.transpose())  # Compute covariance matrix
C = np.dot(X.T, X) / (X.shape[0] - 1)

Lambda, eigvec = np.linalg.eig(C)
Lambda = Lambda.real
eigvec = eigvec.real

Lambda_sort = Lambda[(np.argsort(-Lambda))]
E_e = eigvec[:,np.argsort(-Lambda)]
X_prim = np.dot(X,E_e)
C_prim = np.dot(X_prim.transpose(), X_prim)

PCA_3 = []
for i in range(0,8):
    PCA_3.append(C_prim[i,i]/np.trace(C_prim))
plt.figure(13)
plt.bar(np.linspace(0,8,8), PCA_3, color = color_e)
# plt.xlabel('PC number')
# plt.ylabel('std explained')
plt.yticks([0,1])

# ======== PROJECTINGS VECTORS m, n and global axis onto first PCAs
vector_g = np.ones(len(vector_m_a[0]))
PCA_projs_e = []
g_norm = vector_g/np.linalg.norm(vector_g)
I_norm_e = vector_I_e[0]/np.linalg.norm(vector_I_e[0])
m_norm_e = vector_m_e[0]/np.linalg.norm(vector_m_e[0])

for i in range(0,3):
    PCA_g = np.dot(g_norm, E_e[:,i])

    PCA_I = np.dot(I_norm_e, E_e[:,i])

    PCA_m = np.dot(m_norm_e, E_e[:,i])

    PCA_projs_e.append([PCA_g, PCA_I, PCA_m])

PCA_projs_e = np.absolute(np.asarray(PCA_projs_e))
IM = PCA_projs_e.transpose()
plt.figure(14)
plt.imshow(IM, cmap = 'Greys')
plt.clim([0,1])
plt.colorbar()

plt.xticks([0,1,2])
plt.yticks([])
# ========= ALONG
X = rates_a[0][0][ind_tr:,:] - np.mean(rates_a[0][0][ind_tr:,:], axis = 0)


# X = rates_e_tau_short[0][0] - np.mean(rates_e_tau_short[1], axis = 0)

C = np.cov(X.transpose())  # Compute covariance matrix
C = np.dot(X.T, X) / (X.shape[0] - 1)

Lambda, eigvec = np.linalg.eig(C)
Lambda = Lambda.real
print(Lambda)
eigvec = eigvec.real

Lambda_sort = Lambda[(np.argsort(-Lambda))]
print(Lambda)

E_a = eigvec[:,np.argsort(-Lambda)]
X_prim = np.dot(X,E_a)
C_prim = np.dot(X_prim.transpose(), X_prim)

PCA_1 = []
for i in range(0,8):
    PCA_1.append(C_prim[i,i]/np.trace(C_prim))

plt.figure(15)
plt.bar(np.linspace(0,8,8), PCA_1, color = color_a)
# plt.xlabel('PC number')
# plt.ylabel('std explained')
plt.yticks([0,1])

# ======== PROJECTINGS VECTORS m, n and global axis onto first PCAs

vector_g = np.ones(len(vector_m_a[0]))
PCA_projs_a = []
g_norm = vector_g/np.linalg.norm(vector_g)
I_norm_a = vector_I_a[0]/np.linalg.norm(vector_I_a[0])
m_norm_a = vector_m_a[0]/np.linalg.norm(vector_m_a[0])

for i in range(0,3):
    PCA_g = np.dot(g_norm, E_a[:,i])

    PCA_I = np.dot(I_norm_a, E_a[:,i])

    PCA_m = np.dot(m_norm_a, E_a[:,i])

    PCA_projs_a.append([PCA_g, PCA_I, PCA_m])

PCA_projs = np.absolute(np.asarray(PCA_projs_a))
IM1 = PCA_projs.transpose()
plt.figure(16)
plt.imshow(IM1, cmap = 'Greys')
plt.clim([0,1])

plt.colorbar()
plt.xticks([0,1,2])
plt.yticks([])



# ======= ORTHOGONAL
plt.figure(17)

X = rates_o[0][0][ind_tr:, :] - np.mean(rates_o[0][0][ind_tr:, :], axis = 0)

C = np.cov(X.transpose())  # Compute covariance matrix
C = np.dot(X.T, X) / (X.shape[0] - 1)

Lambda, eigvec = np.linalg.eig(C)
Lambda = Lambda.real
eigvec = eigvec.real

Lambda_sort = Lambda[(np.argsort(-Lambda))]
E_o = eigvec[:,np.argsort(-Lambda)]
X_prim = np.dot(X,E_o)
C_prim = np.dot(X_prim.transpose(), X_prim)

PCA_2 = []
for i in range(0,8):
    PCA_2.append(C_prim[i,i]/np.trace(C_prim))
plt.bar(np.linspace(0,8,8), PCA_2, color = color_o)
# plt.xlabel('PC number')
# plt.ylabel('std explained')
plt.yticks([0,1])


# =========
vector_g = np.ones(len(vector_m_a[0]))
PCA_projs_o = []
g_norm = vector_g/np.linalg.norm(vector_g)
I_norm_o = vector_I_o[0]/np.linalg.norm(vector_I_o[0])
m_norm_o = vector_m_o[0]/np.linalg.norm(vector_m_o[0])

for i in range(0,3):
    PCA_g = np.dot(g_norm, E_o[:,i])

    PCA_I = np.dot(I_norm_o, E_o[:,i])

    PCA_m = np.dot(m_norm_o, E_o[:,i])

    PCA_projs_o.append([PCA_g, PCA_I, PCA_m])

PCA_projs_o = np.absolute(np.asarray(PCA_projs_o))
IM1 = PCA_projs_o.transpose()
plt.figure(18)
plt.imshow(IM1, cmap = 'Greys')
# plt.xticks([0,2,4])
plt.clim([0,1])
plt.xticks([0,1,2])
plt.colorbar()
plt.yticks([])
# ======


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

# ======= HISTOGRAMS for E_I vs rank_1 matrix ===============
f19 = plt.figure()
EI_matrix = S1['I_along; J=100. uV; EI_matrix']
rank1_matrix = S1['I_along; J=100. uV; rank1_matrix']
sum_matrix = EI_matrix+rank1_matrix
J_matrix = S1['I_along; J=100. uV; J_matrix']/volt

for i in range(0,len(EI_matrix)):
    for j in range(0, len(EI_matrix[0])):
        if EI_matrix[i][j]*J_matrix[i][j]<0:
            print('warning')
