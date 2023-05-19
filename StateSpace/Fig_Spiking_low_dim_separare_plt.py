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
        # inst_rates_all.append(inst_rates)
        aver_rates = (aver_rates*trial + inst_rates)/(trial+1)
        pop_rate = np.mean(aver_rates,axis=1)

        sd_firing.append(np.std(pop_rate))

    return [aver_rates, sd_firing]

H = 2.5
W = 5
# W = 7.2 the one at overleaf in the moment

plt.rcParams['figure.figsize'] = W, H
plt.rcParams['font.size'] = 24
# plt.rcParams["font.family"] = "serif"
# "font.sans-serif": ["Helvetica"] an example from matplotlib documentation
plt.rcParams['lines.linewidth'] = 2
# 8 fmar
# plt.rcParams['figure.figsize'] = 4,4
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.bottom'] = False
plt.rcParams['axes.spines.left'] = False
plt.rcParams['axes.spines.right'] = False

# ======== Firing rates ========
factor_mn = 0.01

I_ampli = 1*mV
sigma = 0.1
mu_noise = 24

g = 5
# =============== filter time constant ==========
tau_D_min = 0.55
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

N = 1000
C_sparsity = 100
# C=C_sparsity,
J_syn = 0.5*mV


tau_short = 10*ms
tau_long = 100*ms
t_arr = np.linspace(0, 5, 5000)*second

t1 = 0
t2 = 1

t1 = 0.25
t2 = 0.75

# t = spikes_res_along['time']
t = np.linspace(0, 5, 5000)
np.random.seed(1)
ind1 = int(t1*1000)
ind2 = int(t2*1000)
# ======================= RASTERS ==========================

spikes_res_equal = SPIKES_C(J_value = J_syn, factor_mn = 0.01, I_ampl = I_ampli, ind_c = 0, c_array = c_arr, mu_noise = mu_noise*mV,
                sigma_noise = sigma*mV, nmb_trials = N_trials, N_total= N, C=C_sparsity, syn_D = syn_Del, inp_D = 'orth',
                tau_s= tau_long, tau_min = tau_D_min, tau_max = tau_D_max, inhib_g = g, seed_start=3 )
spikes_t_e = spikes_res_equal['spike_times']
spikes_i_e = spikes_res_equal['neuron ind.']

# ----------
N_r = [7, 180, 392]

f1 = plt.figure()
f2 = plt.figure()
f3 = plt.figure()


for k in range(0,len(N_r)):
    plt.figure(k+1)
    cond = np.logical_and(np.logical_and(spikes_t_e[0] > t1, spikes_t_e[0] < t2), spikes_i_e[0] == N_r[k])

    # plt.plot(spikes_t_e[0][cond], spikes_i_e[0][cond], '.', markersize = 6, color = color_e)
    plt.eventplot(spikes_t_e[0][cond][:])
    plt.xticks([])
    plt.yticks([])

time_array = np.linspace(0, 5, 5000)*second
n_steps_rates = len(time_array)
N_total = 1000
# firing_rates(spikes_t_a, spikes_i_a, t_arr, tau_long, N_total=N)[0]

spike_t = spikes_t_e[0]*second
spike_i = spikes_i_e[0]
inst_rates = np.zeros((len(time_array), N_total))
dt = 1*ms
tau_s = 100*ms

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

t_cond = np.logical_and(time_array > t1*second, time_array < t2*second)

f4 = plt.figure()
f5 = plt.figure()
f6 = plt.figure()

for k in range(0, len(N_r)):
    plt.figure(k+4)
    plt.plot(time_array[t_cond], inst_rates[t_cond, N_r[k]])
    plt.xticks([])
    plt.yticks([])

proj_I = np.dot(inst_rates, np.asarray(spikes_res_equal['vector_I']) )/np.linalg.norm(spikes_res_equal['vector_I'])**2
proj_m = np.dot(inst_rates, spikes_res_equal['vector_m'])/np.linalg.norm(spikes_res_equal['vector_m'])**2


proj_I = np.dot(inst_rates, np.asarray(spikes_res_equal['vector_I']) )/N
proj_m = np.dot(inst_rates, spikes_res_equal['vector_m'])/N


t_cond = np.logical_and(time_array > 0.5*second, time_array < 2*second)

f7 = plt.figure()
plt.plot(time_array[t_cond], proj_I[t_cond])
plt.xticks([])
plt.yticks([])


f8 = plt.figure()
plt.plot(time_array[t_cond], proj_m[t_cond])
plt.xticks([])
plt.yticks([])

f9 = plt.figure()

proj_global = np.dot(inst_rates, np.ones(N) )/N
plt.plot(time_array[t_cond], proj_global[t_cond])
# plt.xticks([])
plt.yticks([0,30,40])

f10 = plt.figure()
plt.plot(proj_I[t_cond], proj_m[t_cond])
plt.axis('off')
