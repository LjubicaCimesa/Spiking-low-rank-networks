import numpy as np
from brian2 import Equations, NeuronGroup, Synapses, Network, TimedArray, SpikeMonitor, StateMonitor, \
    seed, second, mV, ms, volt, set_device, device, defaultclock, prefs
import random
import matplotlib.pyplot as plt
import pickle
plt.rcParams['font.size'] = 24
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
H = 3
W = 5
plt.rcParams['figure.figsize'] = W, H

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

# prefs.codegen.target = 'cython'
# prefs.codegen.c.compiler = 'gcc'
# I = TimedArray(values*(mV/ms), dt = dt_sim)
prefs.codegen.target = "numpy"
filename='values_RATES_mn_different_means_std=1_'
identification='m_second'


seed1 = 1000
np.random.seed(seed=seed1)
random.seed(seed1)
seed(seed1)

#=============== NUMBER OF NEURONS: TOTAL, EXCITATORY, INHIBITORY ========================
N_total = 12500
N_neuron = N_total
N_exc = int(0.8 * N_total)
N_inh = int(0.2 * N_total)
f = 0.1 # sparsity
C_e = int(f * N_exc)
C_i = int(f * N_inh)
C = C_e + C_i
# ======== network and units parameters =========
J_value = 0*mV #mV
# ================================ TIME CONSTANTS ===========================
tau_m = 20*ms #
t_ref = 0.5*ms
g = 5

#============================= NOISE PART =======================================
mu_noise = 40*mV #mV
sigma_white = 0.1*mV #1mv/sqrt(ms)?

synaptic_delay = 1.5*ms

V_th = 20*mV
V_r = 10*mV
#=========================== EXTERNAL INPUT PART ===============================

# t_run = 0.5*second
t_run = 1.2*second
dt = 1*ms
# ============ figures
# set to True to run simulations and produce spikes
redo_spikes = False
# set to True to compute rates from spikes
redo_rates = False


N_trials = 1
N_nets = 3

seed_nets = []
for ind_n in range(0,N_nets):
    seed_nets.append(np.random.randint(0, 2000, 1)[0])
# seed_nets = [599, 1372]
seed_array = []
for int_t in range(0,N_trials):
    seed_array.append(np.random.randint(0, 10000, 1)[0])

sigma_mn_array = np.linspace(3, 3, 1)*0.01
tau_s = 20*ms

# ind_tr = int(0.1/dt)
# ind_tr = 0
filename = 'LC_J=0_SNN_spikes'
filerates = 'LC_J=0_SNN_rates'


if redo_spikes:
    for sigma_mn in sigma_mn_array:

        sigma_m1n1 = 3*0.01
        sigma_m2n2 = 2.6*0.01
        sigma_n1n1 = (1.24 + 7)*0.01
        sigma_n2n2 = (1.63 + 3)*0.01
        sigma_m1n2 = 0.8*0.01
        sigma_m2n1 = -0.8*0.01
        # fk = plt.figure('f'+str(sigma_mn))

        for ind_nets in range(0, N_nets):

            seed_n = seed_nets[ind_nets]
            np.random.seed(seed=seed_n)
            random.seed(seed_n)
            seed(seed_n)
            # fk = plt.figure('f net: '+str(ind_nets))

            E_I = np.zeros((N_total, N_total))

            rank_1 = np.zeros((N_total, N_total))
            rank_2 = np.zeros((N_total, N_total))

    #================================= MATRIX CHI ==================================
            E_I = np.zeros((N_total, N_total))
            for i in range(0, N_total):
                ind_exc = random.sample(range(0, N_exc), C_e)
                ind_inh = random.sample(range(N_exc, N_inh+N_exc), C_i)
                E_I[i, ind_exc] = J_value
                E_I[i, ind_inh] = -g*J_value

            # VECTORS M,N and W
            ones = np.ones(N_total)

            x1 = np.random.normal(0, 1, N_total)
            x1 = x1 - (np.dot(x1, ones)/np.dot(ones,ones))*ones

            x2 = np.random.normal(0, 1, N_total)
            x2 = x2 - (np.dot(x2, ones)/np.dot(ones,ones))*ones - (np.dot(x2,x1)/np.dot(x1,x1))*x1

            x3 = np.random.normal(0, 1, N_total)
            x3 = x3 - (np.dot(x3, ones)/np.dot(ones,ones))*ones - (np.dot(x3,x1)/np.dot(x1,x1))*x1 - (np.dot(x3,x2)/np.dot(x2,x2))*x2

            x4 = np.random.normal(0, 1, N_total)
            x4 = x4 - (np.dot(x4, ones)/np.dot(ones,ones))*ones - (np.dot(x4,x1)/np.dot(x1,x1))*x1 \
                 - (np.dot(x4,x2)/np.dot(x2,x2))*x2 - (np.dot(x4,x3)/np.dot(x3,x3))*x3

            x5 = np.random.normal(0, 1, N_total)
            x5 = x5 - (np.dot(x5, ones)/np.dot(ones,ones))*ones - (np.dot(x5,x1)/np.dot(x1,x1))*x1 \
                 - (np.dot(x5,x2)/np.dot(x2,x2))*x2 - (np.dot(x5,x3)/np.dot(x3,x3))*x3 - (np.dot(x5,x4)/np.dot(x4,x4))*x4

            x6 = np.random.normal(0, 1, N_total)
            x6 = x6 - (np.dot(x6, ones)/np.dot(ones,ones))*ones - (np.dot(x6,x1)/np.dot(x1,x1))*x1 - (np.dot(x6,x2)/np.dot(x2,x2))*x2 \
                - (np.dot(x6,x3)/np.dot(x3,x3))*x3 - (np.dot(x6,x4)/np.dot(x4,x4))*x4 - (np.dot(x6,x5)/np.dot(x5,x5))*x5

    # ================ rank 1, 2
            m_1 = (x1 + x4) / np.sqrt(2)
            m_2 = (x2 + x3) / np.sqrt(2)

            n_1 = np.sqrt(2) * sigma_m1n1 * x1 + np.sqrt(2)*sigma_m2n1*x3 + np.sqrt(sigma_n1n1**2 - 2*sigma_m1n1**2 - 2*sigma_m2n1**2) * x5
            n_2 = np.sqrt(2) * sigma_m2n2 * x2 + np.sqrt(2)*sigma_m1n2*x4 + np.sqrt(sigma_n2n2**2 - 2*sigma_m2n2**2 - 2*sigma_m1n2**2) * x6

            rank_1 = np.outer(m_1, n_1)/N_total
            rank_2 = np.outer(m_2, n_2)/N_total
        # rank_2 is already in the E_I matrix, that is the mean part we subtract from E_I matrix
            J = E_I + rank_1 + rank_2
            J = J*volt

            device.reinit()
            device.activate()

            for trial in range(0, len(seed_array)):

                seed_t = seed_array[trial]
                np.random.seed(seed=seed_t)
                random.seed(seed_t)
                seed(seed_t)
            #==================== RANK-ONE MATRIX AND VECTORS ==========================

                C_0 = 'steelblue'
                Col_array = ['lightblue',C_0,]

                RI_ext = TimedArray(np.zeros((2,N_total))*volt/ms, dt = t_run/2)

                #=================================== NEURON MODEL ======================================
                #  RI_ext_noise(t,i)/sqrt(dt)*sqrt(second)
                eqs = Equations(
                    ''' dv/dt = (-v +(mu_noise + sigma_white * xi* second**0.5) )/tau_m +  RI_ext(t,i):   volt (unless refractory)''',
                    tau_m=tau_m)

                neurons = NeuronGroup(\
                            N_total,
                            model = eqs,
                            threshold = 'v > V_th',
                            reset='v = V_r',
                            refractory = t_ref,
                            name='neurons',
                            method='euler'
                            )

                synapses = Synapses(neurons, neurons,
                                   model='''connectivity_matrix : volt''',
                                   on_pre='''v_post += connectivity_matrix''',
                                   delay = synaptic_delay,
                                   name='synapses')

                synapses.connect()
                synapses.connectivity_matrix = (J.transpose()).flatten()

                V_r_init = V_r + np.random.normal(0, 0.1, N_total)*V_r

                neurons.v = V_r_init

                #----------------------- RASTER PLOTS -----------------------------------------------
                spikes = SpikeMonitor(neurons, name='spikes_monitor')
                syn_monitor = StateMonitor(synapses, ['connectivity_matrix'], record=range(0,5), name='syn_monitor', dt=100*ms)
                net = Network(neurons, synapses, spikes)
                net.run(t_run, report='text')

                trial_dict = {}
                trial_dict['sigma_mn'] = sigma_mn
                trial_dict['spikes_t'] = np.asarray(spikes.t)
                trial_dict['spikes_i'] = np.asarray(spikes.i)
                trial_dict['vector_m1'] = m_1
                trial_dict['vector_m2'] = m_2
                trial_dict['vector_n1'] = n_1
                trial_dict['vector_n2'] = n_2
                # ======================= deo iz Lazarevog koda, beginning =======================================

                f = open(filename, 'ab')
                pickle.dump(trial_dict, f)

                f.close()

                # ======================= deo iz Lazarevog koda, beginning =======================================
        #         dt_rates = 1*ms # ?
        #         n_steps_rates = int(t_run/dt_rates)
        #
        #         if not trial:
        #             average_rates = np.zeros((n_steps_rates, N_total))
        #
        #         inst_rates = np.zeros((n_steps_rates, N_total))
        #         time_array = np.linspace(0, t_run, n_steps_rates)
        #
        #         for k in range(n_steps_rates-1):
        #             t_k = time_array[k]
        #             t_k_plus_one = time_array[k+1]
        #             #list of neurons that spiked between t_k and t_k_plus_one
        #             spiking_neurons = spikes.i[(t_k <= spikes.t) & (spikes.t<t_k_plus_one)]
        #             for i_n in range(N_total):
        #                 if i_n in spiking_neurons: ######################§#############################
        #                     inst_rates[k+1][i_n] = inst_rates[k][i_n]*(1-dt_rates/tau_s) + dt_rates/tau_s
        #                 else: ######################################################################
        #                     inst_rates[k+1][i_n] = inst_rates[k][i_n]*(1-dt_rates/tau_s)
        #
        #         inst_rates = 1000*inst_rates
        #
        #         average_rates += inst_rates/N_trials
        #
        #         K1_curr = np.dot(inst_rates, n_1)/N_total
        #         K2_curr = np.dot(inst_rates, n_2)/N_total
        #         plt.figure(1)
        #         # plt.figure('f net: '+str(ind_nets))
        #         ind_trz = 0
        #         if N_nets==1: plt.plot(K1_curr[int(len(K1_curr)*ind_tr):], K2_curr[int(len(K2_curr)*ind_tr):], label = str(sigma_mn))
        #         else: plt.plot(K1_curr[int(ind_tr):], K2_curr[int(ind_tr):], label = str(sigma_mn), color = 'C'+str(ind_nets))
        #
        #         # , label = str(sigma_mn)
        #         plt.xlabel(r'$\kappa_1$')
        #         plt.ylabel(r'$\kappa_2$')
        #
        #         plt.figure(2)
        #         if N_nets ==1: plt.plot(time_array, K1_curr)
        #         else: plt.plot(time_array, K1_curr, color = 'C'+str(ind_nets))
        #         plt.xlabel('t (s)')
        #         plt.ylabel(r'$\kappa_1$')
        #
        #         plt.figure(3)
        #         if N_nets == 1: plt.plot(time_array, K2_curr)
        #         else: plt.plot(time_array, K2_curr, color = 'C'+str(ind_nets))
        #         plt.xlabel('t (s)')
        #         plt.ylabel(r'$\kappa_2$')
        #
        #         if not trial:
        #             K1 = np.dot(inst_rates, n_1)/N_total
        #             K2 = np.dot(inst_rates, n_2)/N_total
        #             print('trial='+str(trial))
        #         else:
        #             K1 += (np.dot(inst_rates, n_1)/N_total)
        #             K2 += (np.dot(inst_rates, n_2)/N_total)
        #
        #
        #     K1 = K1/N_trials
        #     K2 = K2/N_trials
        #     # plt.figure('f_av')
        #     # plt.plot(K1,K2, label = str(sigma_mn))
        #     # plt.xlabel(r'$\kappa_1$')
        #     # plt.ylabel(r'$\kappa_2$')
        #
        # plt.figure('f'+str(sigma_mn))
        # # plt.legend()

f1 = plt.figure()
f2 = plt.figure(figsize=[5,5])
f3 = plt.figure()
f4 = plt.figure()
f5 = plt.figure()

f = open(filename, 'rb')
S_all = []
nmb_v = N_trials*N_nets
for i in range(0, int(nmb_v)):
    S = pickle.load(f)
    S_all.append(S)
    print(S)
f.close()

ind_ex = 1

spike_t = S_all[ind_ex]['spikes_t']
spike_i = S_all[ind_ex]['spikes_i']
color_C = 'C'+str(ind_ex)
t_1 = 0.5
t_2 = t_run/second
plt.figure(1)

cond = np.logical_and(np.logical_and(spike_t>t_1, spike_t<t_2), spike_i<50)
plt.plot(spike_t[cond], spike_i[cond], '.', color = color_C)


time_array = np.linspace(0, t_run, int(t_run/dt))

R_all =[]
N_batch = N_nets*N_trials


if redo_rates:

    spikes_t_curr = []
    spikes_i_curr = []
    for j in range(0, N_batch):

        print('spike arr ind=' + str(j))

        S1 = S_all[j]

        spikes_t_curr.append(S1['spikes_t'])
        spikes_i_curr.append(S1['spikes_i'])


    R_curr = firing_rates(spikes_t_curr, spikes_i_curr, time_array, tau_s, N_neuron)
    R_all.append(R_curr)

    R={}

    R['Rates'] = R_curr

    f = open(filerates, 'ab')
    pickle.dump(R, f)
    f.close()


file1 = open(filerates, 'rb')
R_all = []
nmb_v = N_trials*N_nets

R = pickle.load(file1)
R_all.append(R)
file1.close()

plt.figure(2)
ind_tr = int(0.2/dt)
K1_diff = []
K2_diff = []

for i in range(0, N_nets):
    for j in range(0, N_trials):
        ind_r = i*N_trials+j
        print('ind_r='+str(ind_r))
        R_curr = R['Rates'][0][ind_r]
        v_m1 = S_all[ind_r]['vector_m1']
        v_m2 = S_all[ind_r]['vector_m2']
        proj1 = np.dot(R_curr, v_m1)/N_neuron
        proj2 = np.dot(R_curr, v_m2)/N_neuron
        K1 = proj1[ind_tr:]
        K2 = proj2[ind_tr:]
        plt.plot(K1, K2, color = 'C'+str(i))
        color_C = 'C'+str(i)
        K1_diff.append(np.max(K1) - np.min(K1))
        K2_diff.append(np.max(K2) - np.min(K2))

        # plt.plot(K1, K2)
# plt.xticks([-2, 0, 2])
# plt.yticks([-2, 0, 2])
plt.axis('off')

ind_1 = int(t_1/dt)
ind_2 = int(t_2/dt)

plt.figure(3)
time_cond = np.logical_and(time_array>t_1*second, time_array<t_2*second)
plt.plot(time_array[time_cond], np.mean(R['Rates'][0][0][time_cond, :],axis=1), color = color_C)
# plt.plot(time_array[time_cond], np.mean(R['Rates'][0][1][time_cond, :],axis=1))
# plt.plot(time_array[time_cond], np.mean(R_curr[time_cond, :],axis=1))

plt.ylim([0,200])


# plt.rcParams['axes.spines.left'] = False

plt.figure(4)
plt.plot(time_array[time_cond], K1[time_cond[ind_tr:]], color = color_C)
plt.plot(time_array[time_cond], np.zeros(np.count_nonzero(time_cond)), '--', color = 'k')
plt.yticks([0])

plt.figure(5)
plt.plot(time_array[time_cond], K2[time_cond[ind_tr:]], color = color_C)
plt.plot(time_array[time_cond], np.zeros(np.count_nonzero(time_cond)), '--', color = 'k')

plt.yticks([0])

# plt.gca().get_yaxis().set_visible(False)

print('K1 diff = '+str(np.max(np.asarray(K1_diff))))
print('K2 diff = '+str(np.max(np.asarray(K2_diff))))

f1.set_tight_layout(True)
f2.set_tight_layout(True)
f3.set_tight_layout(True)
f4.set_tight_layout(True)
f5.set_tight_layout(True)
