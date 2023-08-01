import numpy as np
from brian2 import Equations, NeuronGroup, Synapses, Network, TimedArray, SpikeMonitor, StateMonitor, PopulationRateMonitor, \
    seed, second, mV, ms, volt, set_device, device, defaultclock
import random
import pickle

def spike_non_lin_Smn(N_nets = 1, N_trials = 10, Sm =2, Sn = 2, Smn_array = np.linspace(0, 2.3, 11),  N = 12500,
                  g=5, syn_D = 1.5*ms, J_value = 0.1*mV, mu_ext = 20*mV, sigma_ext = 0.1*mV, t_run = 2*second, filename='file',
                      seed1 = 0):

    spike_dict = {}
    spike_dict['Smn_array'] = Smn_array

    np.random.seed(seed=seed1)
    random.seed(seed1)
    seed(seed1)

    seed_nets = np.random.randint(0, 10000, N_nets)
    print(seed_nets)

    seed_trials = np.random.randint(0, 5000, N_trials)
    print(seed_trials)
    #=============== NUMBER OF NEURONS: TOTAL, EXCITATORY, INHIBITORY ========================
    N_total = N
    N_exc = int(0.8 * N_total)
    N_inh = int(0.2 * N_total)
    f = 0.1 # sparsity
    C_e = int(f * N_exc)
    C_i = int(f * N_inh)
    C = C_e + C_i
    #================================= MATRIX CHI ==================================

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

    #==================== RANK-ONE MATRIX AND VECTORS ==========================
    ones = np.ones(N_total)


    #============================= NOISE PART =======================================
    mu_noise = mu_ext #mV
    sigma_noise = sigma_ext #1mv/sqrt(ms)?

    # t_run = 2*second
    tau_m = 20*ms #
    t_ref = 0.5*ms

    sigma_noise_new = sigma_noise/np.sqrt(2*tau_m)
    # t_ref= 0*ms
    synaptic_delay = syn_D

    V_th = 20*mV
    V_r = 10*mV
    #=========================== EXTERNAL INPUT PART ===============================
    I_i = np.zeros(N_total)*volt
    RI_ext = TimedArray([np.dot(I_i, 0), I_i, np.dot(I_i, 0)], dt=t_run/3)

    g_array = np.linspace(5,9,12)

    device.reinit()
    device.activate()

    Kappa1_all = []
    Kappa2_all = []
    spikes_t_all = []
    spikes_i_all = []
    vector_m_all = []
    vector_n_all = []

    for Smn in Smn_array:
        ind_k = 0
        print('Iteracija :')
        print('Smn = '+str(Smn))

        for s_nets in seed_nets:

            np.random.seed(seed=s_nets)
            random.seed(s_nets)
            seed(s_nets)

            x = np.random.normal(0, 1, N_total)
            x = x - (np.dot(x,ones)/np.dot(ones,ones))*ones

            y = np.random.normal(0, 1, N_total)
            y = y - (np.dot(y,ones)/np.dot(ones,ones))*ones - (np.dot(y,x)/np.dot(x,x))*x

            m_1 = Sm*x
            n_1 = (Smn/Sm)*x + np.sqrt(Sn**2 - (Smn/Sm)**2)*y
            # print(m_1)
            K1_trials = []
            K2_trials = []
            for s_trial in seed_trials:

                np.random.seed(seed=s_trial)
                random.seed(s_trial)
                seed(s_trial)
                ind_k+=1
                print(ind_k)
                print('Trial seed = '+str(s_trial))


                V_r_init = V_r + np.random.normal(0, 0.1, N_total)*V_r

                rank_1 = np.outer(m_1, n_1)/N_total
                # rank_2 is already in the E_I matrix, that is the mean part we subtract from E_I matrix
                J = E_I + rank_1
                J = J*volt
                #=================================== NEURON MODEL ======================================
                eqs = Equations(

                    ''' dv/dt = (-v +  RI_ext(t,i))/tau_m + (mu_noise + sigma_noise * xi* second**0.5) /tau_m : volt (unless refractory)''',
                    tau_m=tau_m)

                neurons= NeuronGroup(\
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

                neurons.v = V_r_init

                #----------------------- RASTER PLOTS -----------------------------------------------
                spikes = SpikeMonitor(neurons, name='spikes_monitor')
                syn_monitor = StateMonitor(synapses,['connectivity_matrix'],record=range(0,5), name='syn_monitor', dt=100*ms)

                net = Network(neurons, synapses, spikes)
                net.run(t_run, report='text')

                spikes_t_all.append(np.asarray(spikes.t))
                spikes_i_all.append(np.asarray(spikes.i))
                vector_m_all.append(m_1)
                vector_n_all.append(n_1)
                trial_dict = {}
                trial_dict['Smn'] = Smn
                trial_dict['spikes_t'] = np.asarray(spikes.t)
                trial_dict['spikes_i'] = np.asarray(spikes.i)
                trial_dict['vector_m'] = m_1
                trial_dict['vector_n'] = n_1
                # ======================= deo iz Lazarevog koda, beginning =======================================
                dt = 1*ms # ?

                n_steps_rates = int(t_run/dt)
                time_array = np.linspace(0, t_run, n_steps_rates)
                trial_dict['time_arr'] = time_array
                f = open(filename, 'ab')
                pickle.dump(trial_dict, f)

                f.close()

    #             inst_rates = np.zeros((n_steps_rates, N_total))
    #             tau_s = 100*ms
    # #
    #             for k in range(n_steps_rates-1):
    #                 t_k = time_array[k]
    #                 t_k_plus_one = time_array[k+1]
    #                 #list of neurons that spiked between t_k and t_k_plus_one
    #                 spiking_neurons = spikes.i[(t_k <= spikes.t) & (spikes.t<t_k_plus_one)]
    #                 for i in range(N_total):
    #                     if i in spiking_neurons : ######################§#############################
    #                         inst_rates[k+1][i] = inst_rates[k][i]*(1-dt/tau_s) + dt/tau_s
    #                     else: ######################################################################
    #                         inst_rates[k+1][i] = inst_rates[k][i]*(1-dt/tau_s)
    #
    #             inst_rates=1000*inst_rates
    #             Kappa_1 = np.dot(np.mean(inst_rates[int(4*t_run/5/dt):, :], axis = 0), m_1)/N_total
    #             Kappa_2 = np.dot(np.mean(inst_rates[int(4*t_run/5/dt):, :], axis = 0), m_2)/N_total
    #             print('Kappa_1'+str(Kappa_1))
    #
    #             K1_trials.append(Kappa_1)
    #             K2_trials.append(Kappa_2)
    #
    #             if (Kappa_1>0) and (Smn == Smn_array[-1]):
    #                     print('upper')
    #                     spike_dict['spike-times, upper'] = np.asarray(spikes.t)
    #                     spike_dict['spike-neurons, upper'] = np.asarray(spikes.i)
    #             if (Kappa_1<0) and (Smn == Smn_array[-1]):
    #                     print('lower')
    #                     spike_dict['spike-times, lower'] = np.asarray(spikes.t)
    #                     spike_dict['spike-neurons, lower'] = np.asarray(spikes.i)
    #
    #         Kappa1_all.append(K1_trials)
    #         print('Kappa 1 all')
    #         print(Kappa1_all)
    #         Kappa2_all.append(K2_trials)
    #         print('Kappa 2 all')
    #         print(Kappa2_all)
    # Kappa_pos = []
    # Kappa_neg = []
    # Kappa2_pos = []
    # Kappa2_neg = []
    # # Kappa1_all = np.asarray(Kappa1_all)
    # # Kappa2_all = np.asarray(Kappa2_all)
    # N_trials = N_nets
    #
    # # for i in range(0,len(Smn_array)):
    # #     Kappa_pos.append(Kappa1_all[ N_trials*i + np.where(Kappa1_all[N_trials*i:N_trials*(i+1)]>0)[0] ])
    # #     Kappa_neg.append(Kappa1_all[ N_trials*i + np.where(Kappa1_all[N_trials*i:N_trials*(i+1)]<=0)[0] ])
    # #     Kappa2_pos.append(Kappa2_all[ N_trials*i + np.where(Kappa1_all[N_trials*i:N_trials*(i+1)]>0)[0] ])
    # #     Kappa2_neg.append(Kappa2_all[  N_trials*i + np.where(Kappa1_all[N_trials*i:N_trials*(i+1)]<=0)[0] ])
    #
    # spike_dict['SNN_K1_pos'] = Kappa_pos
    # spike_dict['SNN_K1_neg'] = Kappa_neg
    # spike_dict['K1_all'] = Kappa1_all
    #
    # spike_dict['K2_all'] = Kappa2_all
    # spike_dict['SNN_K2_pos'] = Kappa2_pos
    # spike_dict['SNN_K2_neg'] = Kappa2_neg
    spike_dict['spikes_t_all'] = spikes_t_all
    spike_dict['spikes_i_all'] = spikes_i_all
    spike_dict['vectors_m'] = vector_m_all
    spike_dict['vectors_n'] = vector_n_all
    spike_dict['t_arr'] = time_array

    return spike_dict

# kappa_dict = {'Figure 1C':{'Smn':Smn_array, 'Kappa_1_SNN': K1}}
# kappa_dict2 = {'Figure 1D':{'Smn':Smn_array, 'Kappa_2_SNN': K2}}
#
#
# Smn_pos = []
# Smn_neg = []
# Kappa_pos = []
# Kappa_neg=[]
# K1= np.asarray(K1)
# for i in range(0,len(Smn_array)):
#     Kappa_pos.append(K1[ 10*i + np.where(K1[10*i:10*(i+1)]>0)[0] ])
#     Kappa_neg.append(K1[ 10*i + np.where(K1[10*i:10*(i+1)]<=0)[0] ])
# for i in range(0,len(Smn_array)):
#     plt.plot(Smn_array[i], np.mean(Kappa_pos[i]),'.', color= 'C0')
#     plt.plot(Smn_array[i], np.mean(Kappa_neg[i]),'.',color= 'C0')
#
# Kappa2 = np.asarray(K2)
# for i in range(0,len(Smn_array)):
#     plt.plot(Smn_array, np.mean(Kappa2.reshape(11,10),axis=1))
#
# f1 = plt.figure()
# plt.plot(Smn_array,np.asarray(K1).reshape(len(Smn_array),2),'.', color='C0')
# plt.xlabel(r'$m^{(1)}n^{(1)}/N$')
# plt.ylabel(r'$\kappa$')
# f1.set_tight_layout('True')
#
#
# f2 = plt.figure()
# plt.plot(Smn_array,np.asarray(K2).reshape(len(Smn_array),2),'.', color='C0')
# plt.xlabel(r'$m^{(1)}n^{(1)}/N$')
# plt.ylabel('pop. rate [Hz]')
# plt.ylim([20,40])
# f2.set_tight_layout('True')
#
# kappa_dict = {'Figure 1C new':{'Smn':Smn_array, 'Kappa1_pos': Kappa_pos, 'Kappa1_neg':Kappa_neg,
#                                'Kappa2':Kappa2}}
# filename = 'Results_figure1'
# import pickle
# file_F1 = open(filename, 'ab')
# pickle.dump(kappa_dict, file_F1)
# file_F1.close()

# # ========================= RASTER PLOTS ======================
# f3 = plt.figure()
# # plt.figsize = 3,4
# f4 = plt.figure()
# # plt.figsize = 3,4
#
# Smn = 2.07
# print('Iteracija :')
# print(Smn)
# np.random.seed(50)
# m_1, n_1 = np.random.multivariate_normal([0,0],[[Sm**2, Smn],[Smn, Sn**2]],N_total).transpose()
# m_1 = m_1 - (np.dot(m_1,ones)/np.dot(ones,ones))*ones
# n_1 = n_1 - (np.dot(n_1,ones)/np.dot(ones,ones))*ones
#
# seed1_array = [0,20]
#
#
# for seed1 in seed1_array:
#
#     np.random.seed(seed=seed1)
#     random.seed(seed1)
#     seed(seed1)
#
#     V_r_init = V_r + np.random.normal(0, 0.1, N_total)*V_r
#
#     E_I = np.zeros((N_total, N_total))
#     for i in range(0, N_total):
#         ind_exc = random.sample(range(0, N_exc), C_e)
#         ind_inh = random.sample(range(N_exc, N_inh+N_exc), C_i)
#         E_I[i, ind_exc] = J_value
#         E_I[i, ind_inh] = -g*J_value
#
#     m_2 = np.ones(N_total)
#     n_2 = np.zeros(N_total)
#     n_2[:N_exc] = J_value*C_e/N_exc
#     n_2[N_exc:] = -g*J_value*C_i/N_inh
#
#     rank_1 = np.outer(m_1, n_1)/N_total/100
#     # rank_2 is already in the E_I matrix, that is the mean part we subtract from E_I matrix
#     J = E_I + rank_1
#     J = J*volt
#     #=================================== NEURON MODEL ======================================
#     eqs = Equations(
#         ''' dv/dt = (-v + RI_j + RI_ext(t,i) + (mu_noise + sigma_noise * xi* second**0.5) )/tau_m : volt (unless refractory)
#         RI_j : volt
#         ''',
#         tau_m=tau_m)
#
#     neurons= NeuronGroup(\
#                 N_total,
#                 model = eqs,
#                 threshold = 'v > V_th',
#                 reset='v = V_r',
#                 refractory = t_ref,
#                 name='neurons',
#                 method='euler'
#                 )
#
#     synapses = Synapses(neurons, neurons,
#                        model='''connectivity_matrix : volt''',
#                        on_pre='''v_post += connectivity_matrix''',
#                        delay = synaptic_delay,
#                        name='synapses')
#
#     synapses.connect()
#     synapses.connectivity_matrix = (J.transpose()).flatten()
#
#     neurons.v = V_r_init
#
#     #----------------------- RASTER PLOTS -----------------------------------------------
#     spikes = SpikeMonitor(neurons, name='spikes_monitor')
#     RI_j_mon = StateMonitor(neurons, ['RI_j'],  record=range(0,2), name='RI_J_mon', dt=0.5*ms)
#     syn_monitor = StateMonitor(synapses,['connectivity_matrix'],record=range(0,5), name='syn_monitor', dt=100*ms)
#
#     net = Network(neurons, synapses, spikes)
#     net.run(t_run, report='text')
#
#
#     # ======================= deo iz Lazarevog koda, beginning =======================================
#     t1 = time.time()
#     dt = 1*ms # ?
#     n_steps_rates = int(t_run/dt)
#
#     inst_rates = np.zeros((n_steps_rates, N_total))
#     time_array = np.linspace(0, t_run, n_steps_rates)
#     tau_s = 100*ms
#
#     for k in range(n_steps_rates-1):
#             t_k = time_array[k]
#             t_k_plus_one = time_array[k+1]
#             #list of neurons that spiked between t_k and t_k_plus_one
#             spiking_neurons = spikes.i[(t_k <= spikes.t) & (spikes.t<t_k_plus_one)]
#             for i in range(N_total):
#                 if i in spiking_neurons : ######################§#############################
#                     inst_rates[k+1][i] = inst_rates[k][i]*(1-dt/tau_s) + dt/tau_s
#                 else: ######################################################################
#                     inst_rates[k+1][i] = inst_rates[k][i]*(1-dt/tau_s)
#
#     inst_rates=1000*inst_rates
#     Kappa_1 = np.mean(np.dot(inst_rates[:-int(4*t_run/5/dt)],n_1)/N_total)
#     Kappa_2 = np.mean(np.dot(inst_rates[:-int(4*t_run/5/dt)],m_2)/N_total)
#     # print(Kappa_1)
#     print(Kappa_1)
#     # print(Kappa_2
#     print(Kappa_2)
#     device.delete()
#     if seed1 == seed1_array[0]:
#         plt.figure(3)
#         # plt.figsize = [3,4]
#         plt.plot(spikes.t[np.logical_and(spikes.i<51, spikes.t<1*second)], spikes.i[np.logical_and(spikes.i<51, spikes.t<1*second)], '.', color = 'C0')
#         plt.xlabel('time[s]')
#         plt.ylabel('neuron num.')
#         f3.set_tight_layout('True')
#
#         SNN_dict1 = {'Figure 1E':{'spikes time':spikes.t[np.logical_and(spikes.i<51, spikes.t<1*second)],
#                                   'spikes ind': spikes.i[np.logical_and(spikes.i<51, spikes.t<1*second)]}}
#         file_F1 = open(filename, 'ab')
#         pickle.dump(SNN_dict1, file_F1)
#         file_F1.close()
#
#     else:
#         plt.figure(4)
#         # plt.figsize = [3,4]
#         plt.plot(spikes.t[np.logical_and(spikes.i<51, spikes.t>1*second)], spikes.i[np.logical_and(spikes.i<51, spikes.t>1*second)], '.', color = 'C0')
#         plt.xlabel('time[s]')
#         plt.ylabel('neuron num.')
#         f4.set_tight_layout('True')
#
#         SNN_dict2 = {'Figure 1F':{'spikes time':spikes.t[np.logical_and(spikes.i<51, spikes.t<1*second)],
#                                   'spikes ind': spikes.i[np.logical_and(spikes.i<51, spikes.t<1*second)]}}
#         file_F1 = open(filename, 'ab')
#         pickle.dump(SNN_dict2, file_F1)
#         file_F1.close()
#
#     # print('population firing rates:')
#     # print(np.mean(inst_rates[-1,:]))
#
#


# K2_phase = [[26.986058669118165,
#   28.901501269986149,
#   18.278562855288843,
#   15.646947852880167,
#   15.345118582387949,
#   13.912094009514599,
#   11.498130431771935,
#   10.215130719448998,
#   10.936963979648786,
#   10.257470753278467,
#   9.5503097832573705,
#   9.8260283716230319,
#   29.210510568550063,
#   26.284534181974887,
#   18.809259568430221,
#   17.010269386798349,
#   14.587819802132142,
#   15.170260478221042,
#   12.098456724236266,
#   10.586059486659551,
#   10.986026255705342,
#   10.721589896505607,
#   10.038649317675985,
#   8.72726058475439],
#  [33.256961425124004,
#   26.145563603329844,
#   18.636254199358401,
#   15.539628975489736,
#   17.004817664280836,
#   13.292879862188338,
#   12.313414807044955,
#   10.159134368820753,
#   11.653656456208946,
#   9.6541848225871565,
#   8.9934853458962198,
#   10.361866423701898,
#   28.400177792425112,
#   22.783653630982535,
#   18.289333830949193,
#   17.784349496851767,
#   13.210586855749957,
#   15.284859048157768,
#   11.996594377513912,
#   10.295220177663673,
#   10.358682658593565,
#   9.3831742586120939,
#   9.3121459736002468,
#   8.7864984202497745],
#  [32.323330242145431,
#   22.255401961890811,
#   19.014578483615196,
#   18.624444851126736,
#   12.13044354304793,
#   13.05873029499641,
#   11.200267009705712,
#   10.521921522243264,
#   9.5823934293289099,
#   11.114141319441629,
#   8.9663698874541407,
#   8.3061176926952953,
#   41.232622793432483,
#   29.189252512305821,
#   18.386411483956138,
#   18.122436180227748,
#   14.776259210291805,
#   12.622764834205581,
#   12.176458426839242,
#   11.237520668562363,
#   11.156444535608079,
#   9.9018314203886639,
#   10.099927847505501,
#   9.6539378778921954],
#  [31.089915544973724,
#   26.506831355714521,
#   20.59205661654601,
#   14.094799886736036,
#   16.017569495432777,
#   14.451263858514796,
#   12.559987616455111,
#   10.644727742865028,
#   10.767581234357722,
#   12.717774195068467,
#   10.11867077856262,
#   8.1084565611174462,
#   34.801515881098403,
#   23.271941059881446,
#   22.07081250804492,
#   15.446785309428979,
#   16.492179851741504,
#   14.324322654341659,
#   11.856611517968476,
#   10.816192145058594,
#   12.73174381351852,
#   9.8060104205889793,
#   10.073563161634073,
#   9.4374703992891131],
#  [31.910032437820735,
#   25.350323992960075,
#   22.24365829373447,
#   14.414665084843495,
#   12.81524797396164,
#   12.445757578387639,
#   11.49888119854602,
#   9.7973402576402808,
#   12.224810348608017,
#   10.459540058259091,
#   10.309453242297973,
#   9.9956152829326683,
#   45.525082604153596,
#   19.237919838518206,
#   24.287851327433611,
#   18.835939921981055,
#   17.665401911404651,
#   15.945102970301363,
#   13.279806843372349,
#   10.727861602668467,
#   10.526834142467356,
#   11.647154712957759,
#   9.9860679622285513,
#   9.4913104604646463],
#  [31.019559002964375,
#   24.478526367150284,
#   21.376744027177459,
#   14.223530497934574,
#   15.001797444424021,
#   12.118912453601572,
#   11.912490948229994,
#   11.658549631790207,
#   10.071137048622072,
#   10.034885054544565,
#   10.739944099180388,
#   10.272999968139095,
#   51.740846423171249,
#   30.119738822566916,
#   22.492707583272448,
#   15.927282754845056,
#   14.182362138917979,
#   13.737038171602762,
#   13.053768065008311,
#   10.057633844763977,
#   9.3944710337280881,
#   12.439696777904357,
#   8.7747267758783405,
#   9.6468606226302516],
#  [31.567510069843109,
#   30.540618312282007,
#   21.439148032730113,
#   16.651208412490071,
#   14.220202751818567,
#   12.638366417196023,
#   11.673199921444436,
#   13.145866244158196,
#   9.6146819318541379,
#   10.763570062908876,
#   9.7497939069611501,
#   8.7503502689183819,
#   49.026679606714055,
#   34.626541011303658,
#   29.472005600051077,
#   24.699375637344136,
#   14.844280760310694,
#   17.216145526726894,
#   13.586871819288435,
#   10.893033808114462,
#   12.731855668596008,
#   10.939520088750974,
#   9.9038307126632859,
#   10.026455133313034],
#  [34.094982747495997,
#   26.034921422916014,
#   21.036581512979286,
#   22.043992660373569,
#   16.311209694163118,
#   13.891734156614561,
#   14.842274499385713,
#   9.2488814193511786,
#   12.096964624245052,
#   10.673123994124841,
#   9.5606440302558031,
#   7.8173626164542522,
#   35.379886740182677,
#   30.144888752074984,
#   22.307747596798741,
#   24.647789496415438,
#   12.796479983898321,
#   13.535948945993058,
#   12.270726027567775,
#   12.273137525659871,
#   11.106691087508294,
#   12.484110766173057,
#   11.427928080825945,
#   9.8373965377383481],
#  [28.581395527085213,
#   21.97881077492897,
#   21.886240988722793,
#   13.643841458073522,
#   15.925595191039527,
#   11.325448111121473,
#   12.401553388815996,
#   12.283768420312967,
#   10.19623296235515,
#   9.922444950365751,
#   10.276045551633738,
#   8.6829155988978179,
#   34.638282268397724,
#   35.297345737153947,
#   30.304285110759615,
#   23.552515066211917,
#   17.417621264158512,
#   19.793340122898201,
#   11.89688422908611,
#   13.137710077269725,
#   11.249813611017412,
#   8.9674654580425059,
#   8.9296382662158607,
#   9.3838486323089612],
#  [28.635830705077204,
#   28.035934537993363,
#   22.054365200962884,
#   14.496059954192015,
#   14.397396034568644,
#   15.817593938329857,
#   10.640940988356745,
#   11.786363671916483,
#   12.167680814104123,
#   10.18866768281808,
#   9.329360751122044,
#   9.4619834386652428,
#   62.625999601677037,
#   39.935364374745667,
#   29.932709984046497,
#   25.365934984150698,
#   20.646855113967373,
#   14.504313453001606,
#   13.902992483965631,
#   11.276397869258583,
#   13.943864473562639,
#   11.441483290489746,
#   10.411765917832902,
#   9.3541957059654308],
#  [30.160145470163624,
#   23.958272148896217,
#   23.978062590490381,
#   16.143172678682831,
#   14.889591887931152,
#   14.044558182866126,
#   11.462687162221863,
#   11.025764073045934,
#   11.003742384433945,
#   10.804336665188611,
#   9.0571690726068432,
#   8.9396056924424148,
#   64.630611353486699,
#   44.954942195512132,
#   33.711453144432042,
#   23.720630068259556,
#   20.158990519087649,
#   15.06245532051064,
#   13.347968303903418,
#   11.349417324951286,
#   11.687557910440269,
#   11.586905790048311,
#   10.608708913848538,
#   10.431414344727775]]
# K2_final = np.zeros((len(Smn_array), len(g_array)))
# for i in range(0,len(Smn_array)):
#     for j in range(0,len(g_array)):
#         K2_final[i][j] = np.mean([K2_phase[i][j],K2_phase[i][int(j+12)]])
#         print(i)
#         print(j)
#
# f = plt.figure()
# plt.imshow(K2_final, cmap=plt.cm.Reds, aspect='auto', origin = 'lower', extent=[g_array[0], g_array[-1],   Smn_array[0], Smn_array[-1]])
# plt.colorbar(cmap=plt.cm.Reds)
# plt.xlabel(r'$g$')
# plt.ylabel(r'$m^{(1)}n^{(1)}/N/100$')
# f.set_tight_layout('True')

