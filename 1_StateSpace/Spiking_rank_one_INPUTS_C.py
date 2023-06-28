import numpy as np
from brian2 import Equations, NeuronGroup, Synapses, Network, TimedArray, SpikeMonitor, StateMonitor, PopulationRateMonitor, \
    seed, second, mV, ms, volt, set_device, device, defaultclock
import random
import matplotlib.pyplot as plt
import pickle
import os
import scipy
import time
# set_device('cpp_standalone', directory='EB_standalone')
# defaultclock.dt = 0.05*ms
filename='values_RATES_mn_different_means_std=1_'
identification='m_second'

plt.rcParams['font.size']=14
plt.rcParams['figure.figsize']= 4, 2
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['lines.markersize'] = 4
# defaultclock.dt = 0.1*ms

def SPIKES_C(J_value = 0.6*mV, factor_mn = 0.01, I_ampl = 1, ind_c = 1, c_array = np.linspace(0,0.2,11), mu_noise = 24*mV,
             sigma_noise = 0.1*mV, nmb_trials = 10, N_total = 1000, C = 100,  syn_D = 'const', inp_D = 'const',
             tau_s = 100*ms, tau_min = 0.55, tau_max = 1.55, inhib_g = 5, t_I_ramp = 0, seed_start = 0):

    print(J_value)
    spikes_dict = {}
    # spikes_dict['c_array'] = c_array
    spikes_dict['J'] = J_value
    # seed for vectors m and n and matrices E_I
    # seed0 = 1
    # np.random.seed(seed=seed0)
    # random.seed(seed0)
    # seed(seed0)
    np.random.seed(seed=seed_start)
    random.seed(seed_start)
    seed(seed_start)
#=============== NUMBER OF NEURONS: TOTAL, EXCITATORY, INHIBITORY ========================
    N_exc = int(0.8 * N_total)
    N_inh = int(0.2 * N_total)
    f = C/N_total
    # f = 0.1 # sparsity
    C_e = int(f * N_exc)
    C_i = int(f * N_inh)
    # C = C_e + C_i
#================================= MATRIX CHI ==================================


#============================= NOISE PART =======================================
    mu_noise = mu_noise
    sigma_noise = sigma_noise

    t_run = 5*second
    t_beg = 1*second
    tau_m = 20*ms #
    t_ref = 0.5*ms

    # sigma_noise_new = sigma_noise/np.sqrt(2*tau_m)
    # t_ref= 0*ms

    V_th = 20*mV
    V_r = 10*mV
#=========================== EXTERNAL INPUT PART ===============================
    g = inhib_g
    device.reinit()
    device.activate()
# ========================== STRUCTURE VECTORS =========================
    Sm = 2
    Sn = 2
    Smn = 0
#==================== RANK-ONE MATRIX AND VECTORS ==========================
    ones = np.ones(N_total)
    # m_1, n_1 = np.random.multivariate_normal([0,0],[[Sm**2, Smn],[Smn, Sn**2]],N_total).transpose()
    # m_1 = m_1 - (np.dot(m_1,ones)/np.dot(ones,ones))*ones
    # n_1 = n_1 - (np.dot(n_1,ones)/np.dot(ones,ones))*ones
    # I_dir_orth = np.random.normal(0,1,N_total)
    # I_dir_orth = I_dir_orth - (np.dot(I_dir_orth, ones)/np.dot(ones, ones))*ones - \
    #     (np.dot(I_dir_orth, m_1)/np.dot(m_1, m_1))*m_1 - (np.dot(I_dir_orth, n_1)/np.dot(n_1, n_1))*n_1
    # I_dir_orth = 2*I_dir_orth



#============== RANK 2 ========================
    # E-I matrix
    # E_I = np.zeros((N_total, N_total))
    # for i in range(0, N_total):
    #     ind_exc = random.sample(range(0, N_exc), C_e)
    #     ind_inh = random.sample(range(N_exc, N_inh+N_exc), C_i)
    #     E_I[i, ind_exc] = J_value
    #     E_I[i, ind_inh] = -g*J_value
    # m_2 = np.ones(N_total)
    # n_2 = np.zeros(N_total)
    # n_2[:N_exc] = J_value*C_e/N_exc
    # n_2[N_exc:] = -g*J_value*C_i/N_inh

    print('factor mn=')
    print(factor_mn)

#     rank_1 = factor_mn * np.outer(m_1, n_1)/N_total
#     # rank_2 is already in the E_I matrix, that is the mean part we subtract from E_I matrix
#
#     J = E_I + rank_1
#     J = J*volt
#     print(J)
# # ======================================================================
#     dt = 1*ms
#
# # ================ Input I in time ================
#     if inp_D == 'const':
#         print('Input is along (1,1..)')
#         I_vector = np.zeros(N_total) + 1
#         spikes_dict['vector_I'] = I_vector
#
#
#     elif inp_D == 'along':
#         print('Input is along n')
#         I_vector = n_1
#         spikes_dict['vector_I'] = n_1
#
#     elif inp_D == 'orth':
#         print('Input is orthogonal to n')
#         I_vector = I_dir_orth
#         spikes_dict['vector_I'] = I_dir_orth
#     else:
#         print('Input is not chosen properly')
#
#     t_ramp = 0*second # 0.5 POSTAVLJENO ZA RAMPU
#     I_input = np.zeros((int(t_run/dt), N_total))
#     for tk in range(0, int((t_beg-t_ramp)/dt)):
#         I_input[tk,:] = 0*I_vector
#     for tk in range(int((t_beg-t_ramp)/dt), int((t_beg+t_ramp)/dt)):
#         I_input[tk,:] = I_vector*( dt*tk/second - t_ramp/second )
#     for tk in range(int((t_beg+t_ramp)/dt), int(t_run/dt)):
#         I_input[tk,:] = I_vector
#     I_input = I_input*I_ampl
#
    dt = 1*ms # ?

    n_steps_rates = int(t_run/dt)

# different seeds for each trial
# different seeds for membrane potentials


    b = 0
    seed_array = []
    while b < N_total:
        s_sample = np.random.randint(0, 20000, 1)
        if not (s_sample in seed_array):
            seed_array.append(s_sample)
            b+=1

    seed_array = np.concatenate(seed_array)

    inst_rates_all = []
    projection_n_all = []
    projection_m_all = []
    projection_I_all = []

    Inputs_all = []
    spikes_t_all = []
    spikes_i_all = []

    for c in c_array:
        print(c)

        average_rates = np.zeros((n_steps_rates, N_total))

        for iter in range(0, nmb_trials):

            seed1 = seed_array[iter]
            print('Iteracija :'+str(iter))
            np.random.seed(seed=seed1)
            random.seed(seed1)
            seed(seed1)

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


            m_1, n_1 = np.random.multivariate_normal([0,0],[[Sm**2, Smn],[Smn, Sn**2]],N_total).transpose()
            m_1 = m_1 - (np.dot(m_1,ones)/np.dot(ones,ones))*ones
            n_1 = n_1 - (np.dot(n_1,ones)/np.dot(ones,ones))*ones
            I_dir_orth = np.random.normal(0,1,N_total)
            I_dir_orth = I_dir_orth - (np.dot(I_dir_orth, ones)/np.dot(ones, ones))*ones - \
                (np.dot(I_dir_orth, m_1)/np.dot(m_1, m_1))*m_1 - (np.dot(I_dir_orth, n_1)/np.dot(n_1, n_1))*n_1
            I_dir_orth = 2*I_dir_orth


            rank_1 = factor_mn * np.outer(m_1, n_1)/N_total
    # rank_2 is already in the E_I matrix, that is the mean part we subtract from E_I matrix

            J = E_I + rank_1
            J = J*volt
            print(J)
        # ======================================================================
            dt = 1*ms

        # ================ Input I in time ================
            if inp_D == 'const':
                print('Input is along (1,1..)')
                I_vector = np.zeros(N_total) + 1
                spikes_dict['vector_I'] = I_vector

            elif inp_D == 'along':
                print('Input is along n')
                I_vector = n_1
                spikes_dict['vector_I'] = n_1

            elif inp_D == 'orth':
                print('Input is orthogonal to n')
                I_vector = I_dir_orth
                spikes_dict['vector_I'] = I_dir_orth
            else:
                print('Input is not chosen properly')

            t_ramp = 0*second # 0.5 POSTAVLJENO ZA RAMPU
            I_input = np.zeros((int(t_run/dt), N_total))
            for tk in range(0, int((t_beg-t_ramp)/dt)):
                I_input[tk,:] = 0*I_vector
            for tk in range(int((t_beg-t_ramp)/dt), int((t_beg+t_ramp)/dt)):
                I_input[tk,:] = I_vector*( dt*tk/second - t_ramp/second )
            for tk in range(int((t_beg+t_ramp)/dt), int(t_run/dt)):
                I_input[tk,:] = I_vector
            I_input = I_input*I_ampl

            Inputs_all.append(c*I_input[-1,:])

            dt = 1*ms # ?

            n_steps_rates = int(t_run/dt)

# different seeds for each trial
# different seeds for membrane potentials
            # initial potential

            # V_r_init = np.zeros(N_total)
            # r = 0
            # while (r < N_total):
            #     V_int = V_r + np.random.normal(0, 1, 1)*V_r
            #     if V_int < V_th:
            #         V_r_init[r] = V_int
            #         r += 1
            # V_r_init = V_r_init*volt

            V_r_init = V_r + np.random.normal(0, 0.1, N_total)*V_r

            RI_ext = TimedArray(c*I_input, dt=dt)
            #=================================== NEURON MODEL ======================================
            eqs = Equations(

                ''' dv/dt = (-v +  RI_ext(t,i) + mu_noise + sigma_noise * xi* second**0.5) /tau_m : volt (unless refractory)''',
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

            neurons.v = V_r_init

            synapses = Synapses(neurons, neurons,
                               model='''connectivity_matrix : volt
                               ''',
                               on_pre='''v_post += connectivity_matrix''',

                               # delay = 100*ms,
                               name='synapses')

            synapses.connect()
            synapses.connectivity_matrix = (J.transpose()).flatten()

    #======================= setting up delays
            synaptic_delay = np.zeros((1, N_total*N_total))
            if syn_D == 'const':
                synaptic_delay = synaptic_delay + tau_min
                print('const delays')
            else:
                print('rand delays')
                for k in range(0,N_total):
                    # synaptic_delay[k,:] = np.linspace(0.55, 1.55, N_total)
                    synaptic_delay[:,k*N_total:(k+1)*N_total] = tau_min + k*(tau_max-tau_min)/(N_total-1)
            print(synaptic_delay)
            synapses.delay = synaptic_delay*ms

    #----------------------- RASTER PLOTS -----------------------------------------------
            spikes = SpikeMonitor(neurons, name='spikes_monitor')
            # syn_mon = StateMonitor(synapses,['connectivity_matrix','delays'], record = range(0,N_total), name='syn_monitor', dt=100*ms)
            syn_mon = StateMonitor(synapses,['connectivity_matrix'], record = range(0,N_total), name='syn_monitor', dt=100*ms)

            V_mon = StateMonitor(neurons,'v', record=range(0,5), name='V_monitor', dt=1*ms)
            #  syn_monitor and V monitor removed, too much memory
            net = Network(neurons, synapses, spikes)
            net.run(t_run, report='text')


    # ======================= deo iz Lazarevog koda, beginning =======================================
    #
            inst_rates = np.zeros((n_steps_rates, N_total))
            time_array = np.linspace(0, t_run, n_steps_rates)
    #         # tau_s = 100*ms now given as input parameters
    #
    #         for k in range(n_steps_rates-1):
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
    #         inst_rates = 1000*inst_rates
    #         average_rates += inst_rates/len(seed_array)

            # projection_m  np.dot(average_rates, m_1)/N_total

            # projection_I = np.dot(average_rates, I_vector)/N_total
            if c ==c_array[ind_c]:
                spikes_t_all.append(np.asarray(spikes.t))
                spikes_i_all.append(np.asarray(spikes.i))

            device.delete()

        projection_n = np.dot(average_rates, n_1)/N_total
        projection_m_all.append(np.dot(average_rates, m_1)/N_total)
        projection_I_all.append(np.dot(average_rates, I_vector)/N_total)

        Kappa = np.dot(np.mean(average_rates[4000:,:], axis = 0), m_1)/N_total
        # projection_I = np.dot(inst_rates, n_1)/N_total

        inst_rates_all.append(average_rates)
        # projection_m_all.append(projection_m)
        projection_n_all.append(projection_n)
        # projection_I_all.append(projection_I)



    # spikes_dict['time'] = np.linspace(0, t_run, int(t_run/dt))
    spikes_dict['Kappa'] = Kappa
    # spikes_dict['synapses'] = syn_mon
    # spikes_dict['time'] = V_mon.t
    spikes_dict['time'] = time_array
    # spikes_dict['V'] = V_mon.v

    # spikes_dict['spike_times'] = np.asarray(spikes.t)
    # spikes_dict['neuron ind.'] = np.asarray(spikes.i)

    spikes_dict['spike_times'] = np.asarray(spikes_t_all)
    spikes_dict['neuron ind.'] = np.asarray(spikes_i_all)


    # spikes_dict['delays'] = syn_mon.delay
    spikes_dict['vector_m'] = m_1
    spikes_dict['vector_n'] = n_1
    if np.count_nonzero(spikes_dict['vector_I']==spikes_dict['vector_n'])==1000:
        print('Equal')
    else:
        print('notttt equal')
    # spikes_dict['proj_m'] = np.mean(np.asarray(projection_m_all), axis = 0)
    spikes_dict['proj_m'] = projection_m_all
    spikes_dict['proj_n'] = projection_n_all
    spikes_dict['proj_I'] = projection_I_all

    # spikes_dict['proj_I'] = np.mean(np.asarray(projection_I_all), axis = 0)

    spikes_dict['firing_rates'] = inst_rates_all
    spikes_dict['I input'] = Inputs_all

    return spikes_dict
