import numpy as np
from brian2 import Equations, NeuronGroup, Synapses, Network, TimedArray, SpikeMonitor, StateMonitor, PopulationRateMonitor, \
    seed, second, mV, ms, volt, set_device, device, defaultclock
import random
import pickle

def spike_non_lin_Mmn(N_nets = 1, N_trials = 10, Sm =2, Sn = 2, Mm = 2, Mn_array = np.linspace(0, 2.3, 11),  N = 12500,
                      C = 1250, g = 5, syn_D = 1.5*ms, J_value = 0.1*mV, mu_ext = 20*mV,
                      sigma_ext = 0.1*mV, t_run = 2*second, t_ref = 0.5*ms, filename='file'):
    spike_dict = {}
    spike_dict['Mn_array'] = Mn_array
    spike_dict['Mm'] = Mm
    seed1 = 0
    np.random.seed(seed=seed1)
    random.seed(seed1)
    seed(seed1)

    seed_nets = np.random.randint(0, 10000, N_nets)
    seed_trials = np.random.randint(0, 5000, N_trials)
    # seed_nets = np.asarray([2732])
    # seed_trials = np.asarray([2607, 4373])
    print('Seed_nets')
    print(seed_nets)
    print('seed trials')
    print(seed_trials)

    #=============== NUMBER OF NEURONS: TOTAL, EXCITATORY, INHIBITORY ========================
    N_total = N
    N_exc = int(0.8 * N_total)
    N_inh = int(0.2 * N_total)
    f = C/N_total # sparsity
    C_e = int(f * N_exc)
    C_i = int(f * N_inh)
    print('C_e = '+str(C_e))
    print('C_i = '+str(C_i))

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
    print('Mm = ')
    print(Mm)

    for Mn in Mn_array:
        print('Iteracija :')
        print('Mn = '+str(Mn))
        for s_nets in seed_nets:

            np.random.seed(seed=s_nets)
            random.seed(s_nets)
            seed(s_nets)

            x = np.random.normal(0, 1, N_total)
            x = x - (np.dot(x,ones)/np.dot(ones,ones))*ones

            y = np.random.normal(0, 1, N_total)
            y = y - (np.dot(y,ones)/np.dot(ones,ones))*ones - (np.dot(y,x)/np.dot(x,x))*x

            m_1 = Mm + Sm*x
            n_1 = Mn + Sn*y
            print('m_1:')
            print(m_1)
            K1_trials = []
            K2_trials = []

            for s_trial in seed_trials:

                np.random.seed(seed=s_trial)
                random.seed(s_trial)
                seed(s_trial)

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
                trial_dict['Mmn'] = Mm*Mn
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

                # ======================= deo iz Lazarevog koda, beginning =======================================
                dt = 1*ms # ?
                n_steps_rates = int(t_run/dt)
                time_array = np.linspace(0, t_run, n_steps_rates)
            #     inst_rates = np.zeros((n_steps_rates, N_total))
            #
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
            #     # Kappa_1 = np.dot(np.mean(inst_rates[int(4*t_run/5/dt):, :], axis = 0), n_1)/N_total
            #     Kappa_1 = np.dot(np.mean(inst_rates[-200:, :], axis = 0), n_1)/N_total
            #     Kappa_2 = np.dot(np.mean(inst_rates[int(4*t_run/5/dt):, :], axis = 0), m_2)/N_total
            #     print('Kappa_1'+str(Kappa_1))
            #
            #     K1_trials.append(Kappa_1)
            #     K2_trials.append(Kappa_2)
            #
            #     # if (Kappa_1>0) and (Mn == Mn_array[-3]):
            #     #         print('upper')
            #     #         spike_dict['spike-times, upper'] = np.asarray(spikes.t)
            #     #         spike_dict['spike-neurons, upper'] = np.asarray(spikes.i)
            #     # if (Kappa_1<0) and (Mn == Mn_array[-3]):
            #     #         print('lower')
            #     #         spike_dict['spike-times, lower'] = np.asarray(spikes.t)
            #     #         spike_dict['spike-neurons, lower'] = np.asarray(spikes.i)
            #
            # Kappa1_all.append(K1_trials)
            # print('Kappa 1 all')
            # print(Kappa1_all)
            # Kappa2_all.append(K2_trials)
            # print('Kappa 2 all')
            # print(Kappa2_all)
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
    # spike_dict['inst rates'] = inst_rates
    # spike_dict['K1_all'] = Kappa1_all
    # spike_dict['K2_all'] = Kappa2_all

    # spike_dict['SNN_K2_pos'] = Kappa2_pos
    # spike_dict['SNN_K2_neg'] = Kappa2_neg
    spike_dict['spikes_t_all'] = spikes_t_all
    spike_dict['spikes_i_all'] = spikes_i_all
    spike_dict['vectors_m'] = vector_m_all
    spike_dict['vectors_n'] = vector_n_all
    spike_dict['t_arr'] = time_array
    spike_dict['Mm'] = Mm
    spike_dict['Mn_array'] = Mn_array

    return spike_dict

