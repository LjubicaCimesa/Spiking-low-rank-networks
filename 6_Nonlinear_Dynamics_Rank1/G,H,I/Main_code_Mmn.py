# from Spiking_rank_Smn import *
from Spiking_rank_Mmn import *

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

#  set to True if you want to run simulations again
redo_spikes = False
# set to True if you want to compute rates from spikes
redo_rates = False

N_nets = 5
N_S_trials = 3

Sm = 2
Sn = 2*0.01
# Smn_array = np.linspace(0, 4, 11)*0.01



# Mn_array = np.asarray([3])*0.03 poslednja vrednost
Mn_array = np.linspace(0, 5, 11)*0.03

# =array([0.009, 0.018, 0.027, 0.036, 0.045, 0.054])
Mm = 1*0.01


# g = 3.5 previous
# g=3.8 i dalje ne
g = 4.8
J_val = 0.1*mV
J_val = 0.2*mV
N_neuron = 12500
mu_noise = 17.702*mV

syn_delay = 2.5*ms
tau_all = np.asarray([100])*ms
t_run = 1.2*second

filename = 'spikes_file'
filerates = 'rates_file'
tau_ref = 2*ms

if redo_spikes:
    spikes_res = spike_non_lin_Mmn(N_nets = N_nets, N_trials = N_S_trials, Sm = Sm, Sn=Sn,
                                       Mm = Mm, Mn_array = Mn_array, N=N_neuron, g = g, C = 4000,
                                       syn_D = syn_delay, J_value = J_val, mu_ext = mu_noise, sigma_ext = 0.1*mV,
                                       t_run = t_run, t_ref = tau_ref, filename = filename)


Mn_array_all = []
f = open(filename, 'rb')
S_all = []
ind_S = 0
N_tot = int( N_nets*N_S_trials*len(Mn_array))
for i in range(0,N_tot):
     S = pickle.load(f)
     print(S)
     S_all.append(S)
f.close()
#
# while True:
#     try:
#         S = pickle.load(f)
#         S_all.append(S)
#         print(S)
#         ind_S += 1
#     except EOFError:
#         break
# f.close()

R_all =[]
N_batch = N_nets*N_S_trials


# Smn_array = [0, 0.04]
# Smn_array1 = np.asarray([Smn_array[0], Smn_array[-1]])
if redo_rates:
    for i in range(0, len(Mn_array)):
        print('Mmn = ' + str(Mn_array[i]))
        spikes_t_curr = []
        spikes_i_curr = []

        for j in range(0, N_batch):

            # print('spike arr ind=' + str(i*N_batch + j))

            S1 = S_all[i*N_batch + j]

            spikes_t_curr.append(S1['spikes_t'])
            spikes_i_curr.append(S1['spikes_i'])

        print(len(spikes_t_curr))

        spike_t = S1['spikes_t']
        spike_i = S1['spikes_i']
        plt.figure()
        plt.plot(spike_t[spike_i<20], spike_i[spike_i<20], '.')

        R_curr = firing_rates(spikes_t_curr, spikes_i_curr, S1['time_arr'], 100*ms, N_neuron)
        R_all.append(R_curr)

        R={}
        R['Smn'] = Mn_array[i]
        R['Rates'] = R_curr

        f = open(filerates, 'ab')
        pickle.dump(R, f)
        f.close()


f1 = open(filerates, 'rb')
R_all = []
nmb_v = len(Mn_array)*N_S_trials*N_nets
for i in range(0, len(Mn_array)):
    R = pickle.load(f1)
    R_all.append(R)
    # print(R)
f1.close()

 #
 # for j in range(0,N_nets):
 #        for k in range(0, N_S_trials):
 #            ind = i*(N_nets+N_S_trials) + j*N_S_trials + k
 #            proj = np.dot(R[ind], vectors_n[ind])/N_neuron
 #            proj_f = np.mean(proj[-200:])
 #            print(ind)




nmb_1 = N_S_trials*N_nets

K_all = []
K_p_all = []
K_n_all = []
pop_p_all = []
pop_n_all = []
pop_all = []

for i in range(0, len(Mn_array)):
    K_Mmn = []
    K_p_Mmn = []
    K_n_Mmn = []
    pop_p_Mmn = []
    pop_n_Mmn = []
    pop_Mmn = []
    ind_R = 0
    for j in range(0, N_nets):
        for k in range(0, N_S_trials):
            ind_S = i*N_nets*N_S_trials + j*N_S_trials + k
            print(ind_S)
            S1 = S_all[ind_S]
            v_n = S1['vector_n']


            print('indR='+str(ind_R))

            # R1 = R_all[i]['Rates'][0][ind_R]
            R1 = R_all[i]['Rates'][0][ind_R]
            proj_n = np.dot(R1, v_n )/N_neuron

            K_curr = np.mean(proj_n[-200:])
            pop_curr= np.mean( np.mean(R1, axis = 1)[-100:])
            pop_Mmn.append(pop_curr)
            ind_R +=1
            # print(len(K_curr))
            # print(K_curr)
            if K_curr>=1.5:
                K_p_Mmn.append(K_curr)
                pop_p_Mmn.append(pop_curr)

            else:
                K_n_Mmn.append(K_curr)
                pop_n_Mmn.append(pop_curr)
            K_Mmn.append(K_curr)
            print(K_curr)

            ind_ras = len(Mn_array)-2
            if i == ind_ras:
                if K_curr>=1.5:
                    spikes_t_p = S1['spikes_t']
                    spikes_i_p = S1['spikes_i']
                else:
                    # if np.any(S1['spikes_i']<20):
                    spikes_t_n = S1['spikes_t']
                    spikes_i_n = S1['spikes_i']

    # if K_curr<1.5:
    #     spikes_t_n = S1['spikes_t']
    #     spikes_i_n = S1['spikes_i']
    #     plt.figure()
    #     plt.plot(spikes_t_n, spikes_i_n, '.')
    #     plt.title(str(i))
    K_p_all.append(K_p_Mmn)
    K_n_all.append(K_n_Mmn)

    pop_p_all.append(pop_p_Mmn)
    pop_n_all.append(pop_n_Mmn)

    K_all.append(K_Mmn)
    pop_all.append(pop_Mmn)

#  ====== KAPPA ==========
fig1, ax = plt.subplots()
K_p_std = []
K_p_mean = []
for i in range(0, len(Mn_array)):
    K_p_mean.append(np.mean(K_p_all[i]))
    K_p_std.append ( np.std(K_p_all[i]))

plt.plot(Mn_array, K_p_mean, '.', color = 'C3')

ax.errorbar(Mn_array, K_p_mean, yerr = K_p_std, fmt = 'none', color = 'C3')


K_n_std = []
K_n_mean = []
for i in range(0, len(Mn_array)):
    K_n_mean.append(np.mean(K_n_all[i]))
    K_n_std.append ( np.std(K_n_all[i]))

plt.plot(Mn_array, K_n_mean, '.', color = 'k')

ax.errorbar(Mn_array, K_n_mean, yerr = K_n_std, fmt = 'none', color = 'k')


# ======POP RATE
fig2, ax = plt.subplots()

pop_p_std = []
pop_p_mean = []
for i in range(0, len(Mn_array)):
    pop_p_mean.append(np.mean(pop_p_all[i]))
    pop_p_std.append ( np.std(pop_p_all[i]))

plt.plot(Mn_array, pop_p_mean, '.', color = 'C3')

ax.errorbar(Mn_array, pop_p_mean, yerr = pop_p_std, fmt = 'none', color = 'C3')


pop_n_std = []
pop_n_mean = []
for i in range(0, len(Mn_array)):
    pop_n_mean.append(np.mean(pop_n_all[i]))
    pop_n_std.append ( np.std(pop_n_all[i]))

plt.plot(Mn_array, pop_n_mean, '.', color = 'k')

ax.errorbar(Mn_array, pop_n_mean, yerr = pop_n_std, fmt = 'none', color = 'k')

# plt.ylim([-10,150])


# fig3 = plt.figure()
# cond_p = np.logical_and(spikes_t_p>=0.2, spikes_i_p<20)
# plt.plot(spikes_t_p[cond_p], spikes_i_p[cond_p],'.', color ='C3' )
# plt.xticks([0.3, 0.4])


fig3 = plt.figure()
cond_p = np.logical_and(spikes_t_p>1, spikes_i_p<20)
plt.plot(spikes_t_p[cond_p], spikes_i_p[cond_p],'.', color ='C3' )
plt.yticks([0,10,20])

fig4 = plt.figure()

# cond_n = np.logical_and(spikes_t_n>1, spikes_i_n<100)
plt.plot(spikes_t_n, spikes_i_n, '.',  color = 'k')
plt.yticks([0,5000])


# plt.xticks([0.3, 0.4])
# plt.yticks([0,10,20])

fig1.set_tight_layout(True)
fig2.set_tight_layout(True)
fig3.set_tight_layout(True)
fig4.set_tight_layout(True)
