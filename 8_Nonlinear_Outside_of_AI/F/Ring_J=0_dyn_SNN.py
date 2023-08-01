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


seed1 =172
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

#============================= NOISE PART =======================================
mu_noise = 40*mV #mV
sigma_white = 0.1*mV #1mv/sqrt(ms)?

synaptic_delay = 1.5*ms

V_th = 20*mV
V_r = 10*mV
#=========================== EXTERNAL INPUT PART ===============================

t_run = 3*second
dt = 1*ms
# ============ figures
# set to True to run simulations and generate spikes
redo_spikes = False
# set to True to compute rates from spikes
redo_rates = False

N_trials = 2
N_nets = 1
seed_nets = []
for ind_n in range(0,N_nets):
    seed_nets.append(np.random.randint(0, 2000, 1)[0])
# seed_nets = [599, 1372]
seed_array = []
for int_t in range(0,N_trials):
    seed_array.append(np.random.randint(0, 10000, 1)[0])

sigma_mn_array = np.linspace(2.5, 2.5, 1)*0.01
# ======filter time constant
tau_s = 20*ms

ind_tr = int(0.1/dt)

# filename = 'Ring_SNN_spikes_1805'
# filerates = 'Ring_SNN_rates_1805'
# set to True to run simulations and generate spikes
filename = 'Ring_J=0_SNN_spikes'
# set to True to compute rates from spikes
filerates = 'Ring_J=0_SNN_rates'
if redo_spikes:
    print(redo_spikes)
    for sigma_mn in sigma_mn_array:

        sigma_m1n1 = sigma_mn
        sigma_m2n2 = sigma_mn
        # sigma_n1n1 = (1.24 + 0.86) * 0.01
        # sigma_n2n2 = (1.63 + 0.47) * 0.01
        #
        sigma_n1n1 = 2.6 * 0.01
        sigma_n2n2 = 2.6 * 0.01

        for ind_nets in range(0,N_nets):

            seed_n = seed_nets[ind_nets]
            np.random.seed(seed=seed_n)
            random.seed(seed_n)
            seed(seed_n)
            # fk = plt.figure('f net: '+str(ind_nets))

            E_I = np.zeros((N_total, N_total))

            rank_1 = np.zeros((N_total, N_total))
            rank_2 = np.zeros((N_total, N_total))

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

            g = 5
    #================================= MATRIX CHI ==================================
            E_I = np.zeros((N_total, N_total))
            for i in range(0, N_total):
                ind_exc = random.sample(range(0, N_exc), C_e)
                ind_inh = random.sample(range(N_exc, N_inh+N_exc), C_i)
                E_I[i, ind_exc] = J_value
                E_I[i, ind_inh] = -g*J_value

            m_1 = x1
            m_2 = x2

            n_1 = sigma_m1n1 * x1 + np.sqrt(sigma_n1n1**2 - sigma_m1n1**2) * x3
            n_2 = sigma_m2n2 * x2 + np.sqrt(sigma_n2n2**2 - sigma_m2n2**2) * x4

            rank_1 = np.outer(m_1, n_1)/N_total
            rank_2 = np.outer(m_2, n_2)/N_total
        # rank_2 is already in the E_I matrix, that is the mean part we subtract from E_I matrix
            J = E_I + rank_1 + rank_2
            J = J*volt

            device.reinit()
            device.activate()

            for trial in range(0, len(seed_array)):
                print('trial nmb = '+str(trial))
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

                time_array = np.linspace(0, t_run, int(t_run/dt))

                trial_dict = {}
                trial_dict['sigma_mn'] = sigma_mn
                trial_dict['spikes_t'] = np.asarray(spikes.t)
                trial_dict['spikes_i'] = np.asarray(spikes.i)
                trial_dict['time_arr'] = time_array
                trial_dict['vector_m1'] = m_1
                trial_dict['vector_m2'] = m_2
                trial_dict['vector_n1'] = n_1
                trial_dict['vector_n2'] = n_2
                # ======================= deo iz Lazarevog koda, beginning =======================================

                f = open(filename, 'ab')
                pickle.dump(trial_dict, f)

                f.close()

                # ======================= deo iz Lazarevog koda, beginning =======================================
                # dt_rates = 1*ms # ?
                # n_steps_rates = int(t_run/dt_rates)
                #
                # if not trial:
                #     average_rates = np.zeros((n_steps_rates, N_total))
                #
                # inst_rates = np.zeros((n_steps_rates, N_total))

                #
                # for k in range(n_steps_rates-1):
                #     t_k = time_array[k]
                #     t_k_plus_one = time_array[k+1]
                #     #list of neurons that spiked between t_k and t_k_plus_one
                #     spiking_neurons = spikes.i[(t_k <= spikes.t) & (spikes.t<t_k_plus_one)]
                #     for i_n in range(N_total):
                #         if i_n in spiking_neurons: ######################§#############################
                #             inst_rates[k+1][i_n] = inst_rates[k][i_n]*(1-dt_rates/tau_s) + dt_rates/tau_s
                #         else: ######################################################################
                #             inst_rates[k+1][i_n] = inst_rates[k][i_n]*(1-dt_rates/tau_s)
                #
                # inst_rates = 1000*inst_rates
                #
                # average_rates += inst_rates/N_trials
                #
                # K1_curr = np.dot(inst_rates, n_1)/N_total
                # X = K1_curr
                # K2_curr = np.dot(inst_rates, n_2)/N_total
                # Y = K2_curr
                # plt.figure(1)
                # plt.plot(K1_curr, K2_curr)
                # # plt.figure(1)
                # plt.figure('f'+str(sigma_mn))
                # # plt.figure('f net: '+str(ind_nets))
                # # ind_trz = int(4*second/tau_s)
                # # if N_nets==1: plt.plot(K1_curr[int(len(K1_curr)*perc_tr):], K2_curr[int(len(K2_curr)*perc_tr):], label = str(sigma_mn))
                # # else: plt.plot(K1_curr[int(len(K1_curr)*perc_tr):], K2_curr[int(len(K2_curr)*perc_tr):], label = str(sigma_mn), color = 'C'+str(ind_nets))
                #
                # # , label = str(sigma_mn)
                # plt.xlabel(r'$\kappa_1$')
                # plt.ylabel(r'$\kappa_2$')

                # plt.figure(2)
                # if N_nets ==1: plt.plot(time_array, K1_curr)
                # else: plt.plot(time_array, K1_curr, color = 'C'+str(ind_nets))
                # plt.xlabel('t (s)')
                # plt.ylabel(r'$\kappa_1$')
                #
                # plt.figure(3)
                # if N_nets == 1: plt.plot(time_array, K2_curr)
                # else: plt.plot(time_array, K2_curr, color = 'C'+str(ind_nets))
                # plt.xlabel('t (s)')
                # plt.ylabel(r'$\kappa_2$')
                #
                # if not trial:
                #     K1 = np.dot(inst_rates, n_1)/N_total
                #     K2 = np.dot(inst_rates, n_2)/N_total
                #     print('trial='+str(trial))
                # else:
                #     K1 += (np.dot(inst_rates, n_1)/N_total)
                #     K2 += (np.dot(inst_rates, n_2)/N_total)


            # K1 = K1/N_trials
            # K2 = K2/N_trials
            # plt.figure('f_av')
            # plt.plot(K1,K2, label = str(sigma_mn))
            # plt.xlabel(r'$\kappa_1$')
            # plt.ylabel(r'$\kappa_2$')

        # plt.figure('f'+str(sigma_mn))
        # plt.legend()
# f = plt.figure()
# plt.plot(time_array[time_array>0.2*second], np.mean(inst_rates[time_array>0.2*second, :], axis = 1))
#
# plt.ylim([0,70])
# plt.xticks([0.2,0.5,1])
# f.set_tight_layout(True)
#
# f = plt.figure()
# cond = np.logical_and(np.logical_and(spikes.t<1*second , spikes.t>0.5*second), spikes.i<50)
# plt.plot(spikes.t[cond], spikes.i[cond],'.')
# f.set_tight_layout(True)


color_list = []
for k in range(0,20):
    color_list.append('C'+str(k))
color_list.append('skyblue')
color_list.append('r')
color_list.append('g')
color_list.append('coral')
color_list.append('m')
color_list.append('y')
color_list.append('purple')
color_list.append('navy')
color_list.append('brown')
color_list.append('salmon')
color_list.append('plum')
color_list.append('maroon')
color_list.append('crimson')
color_list.append('teal')
color_list.append('orchid')


f1 = plt.figure()
f2 = plt.figure(figsize=[5,5])
f3 = plt.figure()
f4 = plt.figure()
f5 = plt.figure()

f = open(filename, 'rb')
S_all = []
nmb_v = N_nets*N_trials
ind_S = 0
while True:
    try:
        S = pickle.load(f)
        S_all.append(S)
        ind_S += 1
    except EOFError:
        break
f.close()

ind_ex = 3
S = S_all[ind_ex]

spike_t = S['spikes_t']
spike_i = S['spikes_i']

plt.figure(1)
t_1 = 0
t_2 = 1.5
cond = np.logical_and(np.logical_and(spike_t>t_1, spike_t<t_2), spike_i<50)
plt.plot(spike_t[cond], spike_i[cond], '.', color = color_list[ind_ex])
plt.yticks([0,50])

R_all =[]
N_batch = N_nets*N_trials
time_array = np.linspace(0, t_run, int(t_run/dt))


file1 = open(filerates, 'rb')
R_all = []
ind_R = 0
while True:
    try:
        R = pickle.load(file1)
        ind_R += 1
    except EOFError:
    break
file1.close()

if redo_rates:

    spikes_t_curr = []
    spikes_i_curr = []
    for j in range(ind_R, ind_S):

        print('spike arr ind=' + str(j))

        S1 = S_all[j]

        spikes_t_curr = S1['spikes_t']
        spikes_i_curr = S1['spikes_i']

        R_curr = firing_rates([spikes_t_curr], [spikes_i_curr], S1['time_arr'], tau_s, N_neuron)
        # R_all.append(R_curr)

        R={}

        R['Rates'] = R_curr

        f = open(filerates, 'ab')
        pickle.dump(R, f)
        f.close()




file1 = open(filerates, 'rb')
R_all = []
while True:
    try:
        R = pickle.load(file1)
        R_all.append(R)
    except EOFError:
    break
file1.close()


plt.figure(2)


K1_all = []
K2_all = []

K1_diff = []
K2_diff = []
for i in range(0, ind_S):
    print(i)
    R_curr = R_all[i]['Rates'][0][0]
    v_n1 = S_all[i]['vector_m1']
    v_n2 = S_all[i]['vector_m2']
    proj1 = np.dot(R_curr, v_n1)/N_neuron
    # plt.plot(proj1)
    proj2 = np.dot(R_curr, v_n2)/N_neuron
    K1 = proj1
    K2 = proj2
    K1_all.append(K1)
    K2_all.append(K2)
    # plt.plot(K1, K2, color = 'C'+str(int(i/N_trials)) )
    if i == ind_ex: plt.plot(K1, K2,  color =color_list[i])
    else: plt.plot(K1[ind_tr:], K2[ind_tr:], color =color_list[i])
    # plt.plot(np.mean(K1[int(0.9*len(K1))]), np.mean(K2[int(0.9*len(K2))]), '.', markersize = 15, color = 'k' )
    K1_diff.append(np.max(K1) - np.min(K1) )
    K2_diff.append(np.max(K2) - np.min(K2) )


plt.xticks([])
plt.yticks([])
plt.axis('off')

plt.figure(3)
print('t1 = '+str(t_1))
print('t2 = '+str(t_1))
time_cond = np.logical_and(time_array>t_1*second, time_array<t_2*second)
plt.plot(time_array[time_cond], np.mean(R_all[ind_ex]['Rates'][0][0][time_cond, :],axis=1), color = color_list[ind_ex])
plt.ylim([0,300])


# plt.rcParams['axes.spines.left'] = False


plt.figure(4)
# plt.plot(time_array[ind_tr:int(len(time_array)/2)], K1_all[ind_ex][:int(len(time_array)/2)-ind_tr],  color = color_list[3])
plt.plot(time_array[time_cond], K1_all[ind_ex][time_cond],  color = color_list[ind_ex])

# plt.ylim([-50, 100])
# plt.gca().get_yaxis().set_visible(False)

#hide x-axis

plt.figure(5)
# plt.plot(time_array[ind_tr:int(len(time_array)/2)], K2_all[ind_ex][:int(len(time_array)/2)-ind_tr],  color = color_list[3])


plt.plot(time_array[time_cond], K2_all[ind_ex][time_cond],  color = color_list[ind_ex])

# plt.ylim([-50, 100])
# plt.gca().get_yaxis().set_visible(False)
print('K1 diff = '+str(np.max(np.asarray(K1_diff))))
print('K2 diff = '+str(np.max(np.asarray(K2_diff))))
f1.set_tight_layout(True)
f2.set_tight_layout(True)
f3.set_tight_layout(True)
f4.set_tight_layout(True)
f5.set_tight_layout(True)
