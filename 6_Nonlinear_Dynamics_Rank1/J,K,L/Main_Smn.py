from Spiking_rank_Smn import *
# from Spiking_rank_Mmn import *

import numpy as np
import matplotlib.pyplot as plt

import pickle

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

plt.rcParams['font.size'] = 11
plt.rcParams["font.family"] = "serif"
# promenila font!!!!!!
plt.rcParams['lines.linewidth'] = 2

H = 3
W = 5
# W = 7.2 the one at overleaf in the moment

plt.rcParams['figure.figsize'] = W,H
plt.rcParams['font.size'] = 24

plt.rcParams['figure.figsize'] = W, H
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['lines.markersize'] = 6


redo_spikes = False

redo_rates = False
N_nets = 7
N_S_trials = 2

Sm = 2
Sn = 2*0.01
Smn_array = np.linspace(0, 4, 11)*0.01


Mn_array = np.linspace(0, 1, 11)*0.01
Mm = 1

g = 5
J_val = 0.1*mV
N_neuron = 12500
mu_noise = 40*mV
syn_delay = 1.5*ms

tau_all = np.asarray([100])*ms
t_run = 1.2*second
filename = 'spikes_file'
filename= 'spikes_file'
if redo_spikes:
    spikes_res = spike_non_lin_Smn(N_nets = N_nets, N_trials = N_S_trials, Sm = Sm, Sn=Sn,
                                       Smn_array= Smn_array, N=N_neuron, g = g, syn_D = syn_delay,
                                       J_value = J_val, mu_ext = mu_noise, sigma_ext = 0.1*mV,
                                       t_run = t_run, filename = filename)


f = open(filename, 'rb')
S_all = []
nmb_v = len(Smn_array)*N_S_trials*N_nets
for i in range(0,int(nmb_v)):
    S = pickle.load(f)
    S_all.append(S)
    print(S)
f.close()

R_all =[]
N_batch = N_nets*N_S_trials

filerates = 'rates_file'

if redo_rates:
    for i in range(0, len(Smn_array)):
        print('Smn = ' + str(Smn_array[i]))
        spikes_t_curr = []
        spikes_i_curr = []
        for j in range(0, N_batch):

            print('spike arr ind=' + str(i*N_batch + j))

            S1 = S_all[i*N_batch + j]

            spikes_t_curr.append(S1['spikes_t'])
            spikes_i_curr.append(S1['spikes_i'])

        spike_t = S1['spikes_t']
        spike_i = S1['spikes_i']
        plt.figure()
        plt.plot(spike_t[spike_i<20], spike_i[spike_i<20], '.')

        R_curr = firing_rates(spikes_t_curr, spikes_i_curr, S1['time_arr'], 100*ms, N_neuron)
        R_all.append(R_curr)

        R={}
        R['Smn'] = Smn_array[i]
        R['Rates'] = R_curr

        f = open(filerates, 'ab')
        pickle.dump(R, f)
        f.close()


filerates = 'rates_file'
f1 = open(filerates, 'rb')
R_all = []
nmb_v = len(Smn_array)*N_S_trials*N_nets
for i in range(0, len(Smn_array)):
    R = pickle.load(f1)
    R_all.append(R)
    # print(R)
f.close()



nmb_1 = N_S_trials*N_nets
K_all = []
K_p_all = []
K_n_all = []
pop_p_all = []
pop_n_all = []

for i in range(0, len(Smn_array)):
    K_Smn = []
    K_p_Smn = []
    K_n_Smn = []
    pop_p_Smn = []
    pop_n_Smn = []
    for j in range(0, N_nets):
        for k in range(0, N_S_trials):
            ind_S = i*N_nets*N_S_trials + j*N_S_trials + k

            S1 = S_all[ind_S]
            v_n = S1['vector_n']

            ind_R = j*N_S_trials+k
            print('ind R'+str(ind_R))

            R1 = R_all[i]['Rates'][0][ind_R]
            proj_n = np.dot(R1, v_n )/N_neuron

            K_curr = np.mean(proj_n[-200:])
            pop_curr= np.mean( np.mean(R1, axis = 1)[-200:])
            # print(len(K_curr))
            # print(K_curr)
            if K_curr>=0:
                K_p_Smn.append(K_curr)
                pop_p_Smn.append(pop_curr)

            else:
                K_n_Smn.append(K_curr)
                pop_n_Smn.append(pop_curr)
            K_Smn.append(K_curr)

            ind_ras = len(Smn_array)-3
            if i == ind_ras:
                if K_curr>=0:
                    spikes_t_p = S1['spikes_t']
                    spikes_i_p = S1['spikes_i']
                else:
                    spikes_t_n = S1['spikes_t']
                    spikes_i_n = S1['spikes_i']


    K_p_all.append(K_p_Smn)
    K_n_all.append(K_n_Smn)

    pop_p_all.append(pop_p_Smn)
    pop_n_all.append(pop_n_Smn)

    K_all.append(K_Smn)

#  ====== KAPPA ==========
fig1, ax = plt.subplots()
K_p_std = []
K_p_mean = []
for i in range(0, len(Smn_array)):
    K_p_mean.append(np.mean(K_p_all[i]))
    K_p_std.append ( np.std(K_p_all[i]))

plt.plot(Smn_array, K_p_mean, '.', color = 'C3')

ax.errorbar(Smn_array, K_p_mean, yerr = K_p_std, fmt = 'none', color = 'C3')


K_n_std = []
K_n_mean = []
for i in range(0, len(Smn_array)):
    K_n_mean.append(np.mean(K_n_all[i]))
    K_n_std.append ( np.std(K_n_all[i]))

plt.plot(Smn_array, K_n_mean, '.', color = 'C0')

ax.errorbar(Smn_array, K_n_mean, yerr = K_n_std, fmt = 'none', color = 'C0')


# ======POP RATE
fig2, ax = plt.subplots()

pop_p_std = []
pop_p_mean = []
for i in range(0, len(Smn_array)):
    pop_p_mean.append(np.mean(pop_p_all[i]))
    pop_p_std.append ( np.std(pop_p_all[i]))

plt.plot(Smn_array, pop_p_mean, '.', color = 'C3')
plt.yticks([-3,0,3])

ax.errorbar(Smn_array, pop_p_mean, yerr = pop_p_std, fmt = 'none', color = 'C3')


pop_n_std = []
pop_n_mean = []
for i in range(0, len(Smn_array)):
    pop_n_mean.append(np.mean(pop_n_all[i]))
    pop_n_std.append (np.std(pop_n_all[i]))

plt.plot(Smn_array, pop_n_mean, '.', color = 'k')

ax.errorbar(Smn_array, pop_n_mean, yerr = pop_n_std, fmt = 'none', color = 'C0')

plt.ylim([-10,150])


fig3 = plt.figure()
cond_p = np.logical_and(spikes_t_p>1, spikes_i_p<20)
plt.plot(spikes_t_p[cond_p], spikes_i_p[cond_p],'.', color ='C3' )

fig4 = plt.figure()

cond_n = np.logical_and(spikes_t_n>1, spikes_i_n<20)
plt.plot(spikes_t_n[cond_n], spikes_i_n[cond_n], '.',  color = 'C0')

fig1.set_tight_layout(True)
fig2.set_tight_layout(True)
fig3.set_tight_layout(True)
fig4.set_tight_layout(True)
