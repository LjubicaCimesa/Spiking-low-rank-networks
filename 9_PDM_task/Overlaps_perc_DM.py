import numpy as np
from brian2 import Equations, NeuronGroup, Synapses, Network, TimedArray, SpikeMonitor, StateMonitor, \
    seed, second, mV, ms, volt, set_device, device, defaultclock
import random
import matplotlib.pyplot as plt

import pickle
plt.rcParams['font.size'] = 24
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
H = 3
W = 5
plt.rcParams['figure.figsize'] = W, H



seed1 = 0
np.random.seed(seed=seed1)
random.seed(seed1)
seed(seed1)

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


# I = TimedArray(values*(mV/ms), dt = dt_sim)


#=============== NUMBER OF NEURONS: TOTAL, EXCITATORY, INHIBITORY ========================
N_total = 12500
N_neuron = N_total
N_exc = int(0.8 * N_total)
N_inh = int(0.2 * N_total)
f = 0.1 # sparsity
C_e = int(f * N_exc)
C_i = int(f * N_inh)
C = C_e + C_i
#================================= MATRIX CHI ==================================
E_I = np.zeros((N_total, N_total))

rank_1 = np.zeros((N_total, N_total))
rank_2 = np.zeros((N_total, N_total))

J_value = 0.1*mV #mV
# ================================ TIME CONSTANTS ===========================
tau_m = 20*ms #
t_ref = 0.5*ms

#============================= NOISE PART =======================================
mu_noise = 40*mV #mV
sigma_white = 0.1*mV #1mv/sqrt(ms)?
# sigma_white = 0.2*mV  # povecala
# sigma_white = 0.3*mV
synaptic_delay = 1.5*ms

V_th = 20*mV
V_r = 10*mV
#=========================== EXTERNAL INPUT PART ===============================

tau_fix = 100*ms
tau_sim = 800*ms
# tau_delay = 100*ms
# tau_dec = 20*ms
tau_delay = 100*ms
tau_dec = 20*ms
tau_tot = tau_fix+tau_sim+tau_delay+tau_dec
t_run = tau_tot
dt = 1*ms

N_trials = 10

seed_array = np.random.randint(0, 10000, 30)[20:]

# u_mean_array =np.asarray( [-50, -30, -16, -8,  -4, -2, -1, 0, 1, 2, 4, 8, 16,30, 50])*3.2/100
# u_mean_array =np.asarray( [-50, -30, -16,  -2, -1, 0, 1, 2, 16,30, 50])*3.2/100
# u_mean_array =np.asarray( [-16, -8, -4, -2, -1, 0, 1, 2, 4, 8, 16])*3.2/100
# u_mean_array =np.asarray( [-16, -8, -4,  4, 8, 16])*3.2/100
u_mean_array = np.asarray([-30, -16,  -8,  -4,  -2,  -1,   0,   1,   2,   4,   8,  16, 30])*3.2/100
# u_mean_array = np.asarray([4,8,16])*3.2/100 jos ovo dodaj u proba
u_mean_array = np.asarray([ -16])*3.2/100
u_mean_array = np.asarray([ -16, -8, - 4, -2, 0, 2, 4, 8, 16])*3.2/100


# u_mean_array = np.asarray([16])*3.2/100 kreni od -16


# u_mean_array =np.asarray( [16])*3.2/100
# u_mean_array = np.linspace(0, 0.01, 5)
sigma_u =1
# sigma_u = 1

# VECTORS M,N and W
ones = np.ones(N_total)

x1 = np.random.normal(0, 1, N_total)
x1 = x1 - (np.dot(x1, ones)/np.dot(ones,ones))*ones

x2 = np.random.normal(0, 1, N_total)
x2 = x2 - (np.dot(x2, ones)/np.dot(ones,ones))*ones - (np.dot(x2,x1)/np.dot(x1,x1))*x1

x3 = np.random.normal(0, 1, N_total)
x3 = x3 - (np.dot(x3, ones)/np.dot(ones,ones))*ones - (np.dot(x3,x1)/np.dot(x1,x1))*x1 - (np.dot(x3,x2)/np.dot(x2,x2))*x2

g = 5

E_I = np.zeros((N_total, N_total))
for i in range(0, N_total):
    ind_exc = random.sample(range(0, N_exc), C_e)
    ind_inh = random.sample(range(N_exc, N_inh+N_exc), C_i)
    E_I[i, ind_exc] = J_value
    E_I[i, ind_inh] = -g*J_value

m_2 = np.ones(N_total)
n_2 = np.zeros(N_total)
n_2[:N_exc] = J_value*C_e/N_exc
n_2[N_exc:] = -g*J_value*C_i/N_inh

#  from Adrian's paper S2.1
# sigma_nm = 1.4 from Adrian's paper, I need to reduce it a bit
# sigma_nm_array = np.linspace(0, 1.6, 10)
# sigma_nm_array = np.linspace(1, 1.6, 1)*0.01
sigma_nm_array = np.asarray([1.6])*0.01
# sigma_nm = 1.2 + 0.75

sigma_nI = 2.6*0.1 # ovo sam promenila
# sigma_nI = 2*0.1 # ovo sam promenila
# sigma_mw = 2.1*0.01  # ovo sam promenila
sigma_mw = 2.1
# sigma_mm = 2*0.01
sigma_mm = 2*0.01
# /100 factor, vector m needs to take 1/10, and vector n 1/10 (1/10 * 1/10 = 1/100)

f1 = plt.figure()
f2 = plt.figure()
f3 = plt.figure()
f4 = plt.figure()
f5 = plt.figure()
f6 = plt.figure()
f7 = plt.figure()

spikes_t_all = []
spikes_i_all = []

# set redo_spikes to True if you need to run more trials
redo_spikes = False
# set redo rates to True if you completed all the trials, you have all the spikes and you just need to compute rates
redo_rates = False


file_spikes = 'spikes_overlap =' +str(sigma_nm_array[0])
file_rates = 'rates_overlap = '+str(sigma_nm_array[0])


tau_s = 100*ms

if redo_spikes:
    for sigma_nm in sigma_nm_array:
        print('sigma_mn ='+str(sigma_nm))
        I_i = x1
        n_1 = sigma_nI*x1 + np.sqrt(sigma_nm)*x2
        m_1 = np.sqrt(sigma_nm)*x2 + np.sqrt(sigma_mm-sigma_nm)*x3

        w = sigma_mw/np.sqrt(sigma_mm - sigma_nm) * x3
        # w = np.random.normal(0, 4, N_total) this is before training?

        device.reinit()
        device.activate()
        # Sm = 2
        # Sn = 2
        # Smn = 3

        rank_1 = np.outer(m_1, n_1)/N_total
        # rank_2 is already in the E_I matrix, that is the mean part we subtract from E_I matrix
        J = E_I + rank_1
        J = J*volt

        right_choice_all = []
        left_choice_all = []
        Kappa = []

        for u_mean in u_mean_array:
            right_choice = 0
            left_choice = 0
            print(u_mean)
            for trial in range(0, len(seed_array)):
                print('trial = ' + str(trial))
                seed1 = seed_array[trial]
                np.random.seed(seed=seed1)
                random.seed(seed1)
                seed(seed1)

                u_k_m = np.zeros(int(tau_tot/dt))
                u_k_m[int(tau_fix/dt): int((tau_fix+tau_sim)/dt)] = u_mean
                # #  u_k is drawn uniformly
                eps_k = np.zeros(int(tau_tot/dt))
                for k in range(int(tau_fix/dt), int((tau_fix + tau_sim)/dt)):
                    eps_k[k] = np.random.normal(0, sigma_u, 1)
                u_k = u_k_m + eps_k
                # u_k = u_k_m

                I_input_all = []
        #==================== RANK-ONE MATRIX AND VECTORS ==========================

                C_0 = 'steelblue'
                Col_array = ['lightblue', C_0]

                I_i_ff = np.zeros((len(u_k), N_total))
                for k in range(0, len(u_k)):
                    I_i_ff[k,:] = u_k[k]*I_i

                I_i_ff = I_i_ff*mV
                RI_ext = TimedArray(I_i_ff, dt)

                #=================================== NEURON MODEL ======================================
                #  RI_ext_noise(t,i)/sqrt(dt)*sqrt(second)
                eqs = Equations(
                    ''' dv/dt = (-v + RI_ext(t,i)+(mu_noise + sigma_white * xi* second**0.5) )/tau_m  :   volt (unless refractory)''',
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

                spikes_t_all.append(spikes.t)
                spikes_i_all.append(spikes.i)

                dt_rates = 1*ms # ?
                n_steps_rates = int(t_run/dt_rates)
                if not trial:
                    average_rates = np.zeros((n_steps_rates, N_total))

                inst_rates = np.zeros((n_steps_rates, N_total))
                time_array = np.linspace(0, t_run, n_steps_rates)


                trial_dict = {}
                trial_dict['sigma_mn'] = sigma_nm
                trial_dict['u_mean'] = u_mean
                trial_dict['u_k'] = u_k
                trial_dict['spikes_t'] = np.asarray(spikes.t)
                trial_dict['spikes_i'] = np.asarray(spikes.i)
                trial_dict['time_arr'] = time_array
                trial_dict['vector_m1'] = m_1
                trial_dict['vector_n1'] = n_1
                trial_dict['vector_I'] = I_i
                trial_dict['vector_w'] = w
                trial_dict['I_all_t'] = I_i_ff
                trial_dict['seed'] = seed1

                f1 = open(file_spikes, 'ab')
                pickle.dump(trial_dict, f1)

                f1.close()

                # ======================= deo iz Lazarevog koda, beginning =======================================

            #     for k in range(n_steps_rates-1):
            #             t_k = time_array[k]
            #             t_k_plus_one = time_array[k+1]
            #             #list of neurons that spiked between t_k and t_k_plus_one
            #             spiking_neurons = spikes.i[(t_k <= spikes.t) & (spikes.t<t_k_plus_one)]
            #             for i_n in range(N_total):
            #                 if i_n in spiking_neurons : ######################§#############################
            #                     inst_rates[k+1][i_n] = inst_rates[k][i_n]*(1-dt_rates/tau_s) + dt_rates/tau_s
            #                 else: ######################################################################
            #                     inst_rates[k+1][i_n] = inst_rates[k][i_n]*(1-dt_rates/tau_s)
            #
            #     inst_rates = 1000*inst_rates
            #
            #     Kappa.append(np.mean((np.dot(inst_rates, m_1)/N_total)[-100:]))
            #
            #     average_rates += inst_rates/N_trials
            #
            #     if not trial:
            #         proj_m_av = np.dot(inst_rates, m_1)/N_total
            #         proj_I_av = np.dot(inst_rates, I_i)/N_total
            #         readout_av = np.dot(inst_rates, w)/N_total/np.std(w)**2
            #     else:
            #         proj_m_av += np.dot(inst_rates, m_1)/N_total
            #         proj_I_av += np.dot(inst_rates, I_i)/N_total
            #         readout_av += np.dot(inst_rates, w)/N_total
            #
            #     readout_curr = np.dot(inst_rates, w)/N_total/np.std(w)**2
            #     decision = np.sum(readout_curr[-int(tau_dec/dt):])
            #     plt.figure(1)
            #     plt.plot(np.linspace(0, tau_tot, int(tau_tot/dt)), readout_curr)
            #     if decision>0: right_choice+=1
            #     else: print('left');left_choice+=1
            #     # plt.xlabel('time (s)')
            #     # plt.ylabel('readout')
            #
            #
            #
            # right_choice_all.append(right_choice/N_trials)
            # left_choice_all.append(left_choice/N_trials)
            #
            #
            # proj_m_av = proj_m_av/N_trials
            # proj_I_av = proj_I_av/N_trials
            # readout_av = readout_av/N_trials
            # # proj_m = np.dot(average_rates, m_1)/N_total
            # # proj_I = np.dot(average_rates, I_i)/N_total
            # plt.figure(2)
            # plt.plot(proj_I_av, proj_m_av)
            # plt.xlabel('proj on I')
            # plt.ylabel('proj on m')

        # Kappa is needed for different overlaps

        # plt.figure(3)
        # plt.plot(np.zeros(len(Kappa))+sigma_nm, Kappa, '.')
        # # plt.xlabel(r'$\sigma_{mn}$')
        # # plt.ylabel(r'$\kappa$')
        #
        #
        # plt.figure(4)
        # plt.plot(u_mean_array, right_choice_all)
        #
        # plt.figure(6)
        # plt.plot(spikes.t, spikes.i, '.')
        # plt.xlabel('coherence')
        # plt.ylabel('choices to right')


# fx =plt.figure()
# spikes_t_n = spikes_t_all[0]
# spikes_i_n = spikes_i_all[0]
# cond1 = spikes_i_n<50
# plt.plot(spikes_t_n[cond1], spikes_i_n[cond1],'.')
#
# fy = plt.figure()
# spikes_t_p = spikes_t_all[1]
# spikes_i_p = spikes_i_all[1]
# cond2 = spikes_i_p<50
# plt.plot(spikes_t_p[cond2], spikes_i_p[cond2],'.', color = 'C1')


file1 = open(file_spikes, 'rb')
ind_S = 0
S_all = []
u_means = []
while True:
    try:
        S = pickle.load(file1)
        S_all.append(S)
        u_means.append(S['u_mean'])
        ind_S += 1
    except EOFError:
        break
file1.close()

u_means_sorted = np.unique(sorted(u_means))

u_mean_array = sorted(u_means)

N_tr = int(len(u_means)/len(u_means_sorted))


print('N_recorded trials = '+str(N_tr))
time_array = np.linspace(0, t_run, int(t_run/dt))
ind_st = 0
R_all = []

file2 = open(file_rates,'ab')
file2.close()
count_rates = 0
# terminate = 0
file2 = open(file_rates,'rb')

k = 1
count_rates = 0
while k:
    try:
        pickle.load(file2)
        count_rates+=1
    except:
        k = 0

file2.close()

print('count_rates='+str(count_rates))
if redo_rates:

    for i in range(0, len(sigma_nm_array)):
        for j in range(count_rates, len(S_all)):
            ind_st = i*len(sigma_nm_array) + j
            # ind_end = i*len(sigma_nm_array)*len(u_mean_array)+(j+1)*N_trials
            # print(ind_st)
            print('ind_st = '+str(ind_st))
            spikes_t_curr = []
            spikes_i_curr = []

            # for ind_tr in range(0,N_tr):
            #
            #     print('ind_st='+str(ind_st))
            S1 = S_all[ind_st]

            spikes_t_curr= [S1['spikes_t']]
            spikes_i_curr= [S1['spikes_i']]
            # ind_st+=1

            R_curr = firing_rates(spikes_t_curr, spikes_i_curr, S1['time_arr'], tau_s, N_neuron)
            R_all.append(R_curr)

            R={}

            R['u_k_mean'] = S1['u_mean']
            R['Rates'] = R_curr
            # R['seed'] = S1['seed']
            file2 = open(file_rates, 'ab')
            pickle.dump(R, file2)
            file2.close()



file2 = open(file_rates, 'rb')
R_all = []
for i in range(0, len(sigma_nm_array)):
    for j in range(0,len(S_all)):


        R = pickle.load(file2)
        R_all.append(R)
file2.close()

S_all_sorted =[]
R_all_sorted =[]

u_means_sorted = np.unique(sorted(u_means))

for u_m_sorted in u_means_sorted:

    for j in range(0,len(S_all)):
       if (u_m_sorted == S_all[j]['u_mean']):
            S_all_sorted.append(S_all[j])
            R_all_sorted.append(R_all[j])
            print(S_all[j]['u_mean'])

# S_all_unsorted = S_all
# S_all = S_all_sorted


u_mean_array = sorted(u_means)
right_choice_all_1 = np.zeros((len(sigma_nm_array), len(u_means_sorted)))
ind_spikes = 0
for i in range(0, len(sigma_nm_array)):
    for j in range(0, len(S_all_sorted)):
        right_choice_1 = 0

        R_curr = R_all_sorted[j]['Rates'][0][0]
        S_curr = S_all_sorted[j]
        # print('ind_spikes'+str(ind_spikes))
        # print('sigma_mn'+ str(S_curr['sigma_mn']))
        # print('u_mean'+ str(S_curr['u_mean']))
        w = S_curr['vector_w']
        readout_curr = np.dot(R_curr, w)/N_total
        decision = np.sum(readout_curr[-int(tau_dec/dt):])
        if decision>0:
            right_choice_all_1[i][np.argwhere(S_all_sorted[j]['u_mean']==u_means_sorted)] += 1/N_tr
            print('right')
        else:
            print('left')

            ind_spikes+=1
        # r
        # right_choice_all_1[i][np.argwhere(S_all[j]['u_mean']==u_means_sorted)] += right_choice_1

# plt.figure()
# for i in range(0, len(sigma_nm_array)):
#     plt.plot(right_choice_all_1[i][:])
# plt.title('second way')

color_neg = 'steelblue'
color_pos = 'darkorange'
S_neg1 = S_all[0]
S_neg2 = S_all[0]
S_pos2 = S_all[-1]
S_pos1 = S_all[-1]

# plt.rcParams['axes.spines.bottom'] = False

plt.figure(1)
plt.plot(S_neg1['time_arr'], S_neg1['u_k'], color_neg)
plt.plot(S_pos1['time_arr'], S_pos1['u_k'], color_pos)
# plt.gca().get_xaxis().set_visible(False)

# plt.rcParams['axes.spines.bottom'] = True

plt.figure(2)
R_neg1 = firing_rates(np.asarray([S_neg1['spikes_t']]), np.asarray([S_neg1['spikes_i']]), S_neg1['time_arr'], tau_s, N_neuron)
readout_neg1 = np.dot(R_neg1[0][0], S_neg1['vector_w'])/np.linalg.norm(S_neg1['vector_w'])**2

R_neg2 = firing_rates(np.asarray([S_neg2['spikes_t']]), np.asarray([S_neg2['spikes_i']]), S_neg2['time_arr'], tau_s, N_neuron)
readout_neg2 = np.dot(R_neg2[0][0], S_neg2['vector_w'])/np.linalg.norm(S_neg2['vector_w'])**2

R_pos1 = firing_rates(np.asarray([S_pos1['spikes_t']]), np.asarray([S_pos1['spikes_i']]), S_pos1['time_arr'], tau_s, N_neuron)
readout_pos1 = np.dot(R_pos1[0][0], S_pos1['vector_w'])/np.linalg.norm(S_pos1['vector_w'])**2

R_pos2 = firing_rates(np.asarray([S_pos2['spikes_t']]), np.asarray([S_pos2['spikes_i']]), S_pos2['time_arr'], tau_s, N_neuron)
readout_pos2 = np.dot(R_pos2[0][0], S_pos2['vector_w'])/np.linalg.norm(S_pos2['vector_w'])**2

color_neg2 = 'lightskyblue'
color_pos2 = 'moccasin'

plt.plot(S_neg1['time_arr'], readout_neg1, color_neg, label=' ')
plt.plot(S_pos1['time_arr'], readout_pos1, color_pos, label=' ')
# plt.plot(S_neg2['time_arr'], readout_neg2, color_neg2, label=' ')
# plt.plot(S_pos2['time_arr'], readout_pos2, color_pos2, label=' ')
# plt.legend()
# plt.gca().get_xaxis().set_visible(False)
plt.rcParams['axes.spines.bottom'] = False


plt.figure(3)
s_t_neg = S_neg1['spikes_t']
s_i_neg = S_neg1['spikes_i']
plt.plot(s_t_neg[s_i_neg<30], s_i_neg[s_i_neg<30], '.', color= color_neg)

plt.gca().get_xaxis().set_visible(False)

plt.figure(4)
s_t_pos = S_pos1['spikes_t']
s_i_pos = S_pos1['spikes_i']
plt.plot(s_t_pos[s_i_pos<30], s_i_pos[s_i_pos<30], '.',  color=color_pos)
plt.gca().get_xaxis().set_visible(False)

plt.rcParams['axes.spines.bottom'] = True

plt.figure(5)
proj_I_neg1 = np.dot(R_neg1[0][0], S_neg1['vector_I'])/np.linalg.norm(S_neg1['vector_I'])
proj_m_neg1 = np.dot(R_neg1[0][0], S_neg1['vector_m1'])/np.linalg.norm(S_neg1['vector_m1'])

proj_I_neg2 = np.dot(R_neg2[0][0], S_neg2['vector_I'])/np.linalg.norm(S_neg2['vector_I'])
proj_m_neg2 = np.dot(R_neg2[0][0], S_neg2['vector_m1'])/np.linalg.norm(S_neg2['vector_m1'])

proj_I_pos1 = np.dot(R_pos1[0][0], S_pos1['vector_I'])/np.linalg.norm(S_pos1['vector_I'])
proj_m_pos1 = np.dot(R_pos1[0][0], S_pos1['vector_m1'])/np.linalg.norm(S_pos1['vector_m1'])

proj_I_pos2 = np.dot(R_pos2[0][0], S_pos2['vector_I'])/np.linalg.norm(S_pos2['vector_I'])
proj_m_pos2 = np.dot(R_pos2[0][0], S_pos2['vector_m1'])/np.linalg.norm(S_pos2['vector_m1'])


plt.plot(proj_m_neg1, proj_I_neg1, color_neg, label=str(S_neg1['u_mean']))
# plt.plot(proj_m_neg2, proj_I_neg2, color_neg2, label=str(S_neg2['u_mean']))
plt.plot(proj_m_pos1, proj_I_pos1, color_pos, label=str(S_pos1['u_mean']))
# plt.plot(proj_m_pos2, proj_I_pos2, color_pos2, label=str(S_pos2['u_mean']))
plt.axis('off')
plt.figure(6)
plt.plot(S_neg1['time_arr'], np.mean(R_neg1[0][0], axis = 1), color_neg)
plt.plot(S_pos1['time_arr'], np.mean(R_pos1[0][0], axis = 1), color_pos)

plt.figure(7)
plt.plot(u_means_sorted, right_choice_all_1[0],'.', markersize=7, color = 'green')

plt.figure()
plt.plot(S_all[0]['u_k'])
#
#
# plt.figure(7)
# sigma_nm_array1 = np.asarray([1.2])*0.01
# file_spikes1 = '1overlap_spikes_2606+overlap= ' +str(sigma_nm_array1)
# file_rates1 = '1overlap_rates_2606+overlap= '+ str(sigma_nm_array1)
#
# # second_overlap
#
#
# file11 = open(file_spikes1, 'rb')
# S_all1 = []
# # nmb_v = N_nets*N_trials
# ind_S = 0
# while True:
#     try:
#         S = pickle.load(file11)
#         S_all1.append(S)
#         ind_S += 1
#     except EOFError:
#         break
# file11.close()
# #
# R_all1 = []
# file21 = open(file_rates1, 'rb')
# ind_R = 0
# while True:
#     try:
#         R = pickle.load(file21)
#         R_all1.append(R)
#         ind_R += 1
#     except EOFError:
#         break
# file21.close()
#
# u_mean_array1 = []
#
#
# i_s  = 0
# right_choice_all1 = []
# for i in range(0, len(R_all1)):
#     right_choice_1 = 0
#     u_mean_array1.append(S_curr['u_mean'])
#     for ind_t in range(0, N_trials):
#         S_curr = S_all1[i_s]
#         print(S_curr['u_mean'])
#
#         i_s+=1
#         R_curr = R_all1[i]['Rates'][0][ind_t]
#         w = S_curr['vector_w']
#         readout_curr = np.dot(R_curr, w)/N_total/np.std(w)**2
#         decision = np.sum(readout_curr[-int(tau_dec/dt):])
#         if decision>0:
#             right_choice_1+=1
#             print('right')
#         else:
#             print('left')
#     right_choice_all1.append(right_choice_1/N_trials)
# plt.figure()
# u_means_array = sorted(u_means)
# ind_arr = np.argsort(u_mean_array1)
# for i in ind_arr:
#     plt.plot(u_mean_array1[i], right_choice_all1[i],'.')
#     print(i)
# # plt.plot(u_mean_array1[ 6,  7,  8,  1,  2,  9, 10,  3,  0, 11, 12,  4,  5, 13, 14], right_choice_all1[ind_arr],'.')
#
# plt.plot(u_means_sorted, right_choice_all1[ind_arr],'.')
#
#
f1.set_tight_layout(True)
f2.set_tight_layout(True)
f3.set_tight_layout(True)
f4.set_tight_layout(True)
f5.set_tight_layout(True)
f6.set_tight_layout(True)
f7.set_tight_layout(True)

