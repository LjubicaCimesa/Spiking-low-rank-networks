import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
plt.rcParams['font.size']=14
plt.rcParams['figure.figsize']= 6,5
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['lines.markersize'] = 10

import seaborn as sns

gaussian_norm = (1/np.sqrt(np.pi))
gauss_points, gauss_weights = np.polynomial.hermite.hermgauss(300)
gauss_points = gauss_points*np.sqrt(2)


def phi_tanh(x, offset=0):
    temp_x = np.tanh(x-offset)
    return temp_x

def phi_shif_tanh(x, offset = 0):
    temp_x = 1 + np.tanh(x-offset)
    return temp_x

def phi_derivative(x, offset = 0):
    der = 1 - np.tanh(x - offset)**2
    return der

def Phi(mu, Delta0, offset=0):
    # integrand = np.tanh(mu+np.sqrt(delta0)*gauss_points)
    integrand = phi_tanh(mu + np.sqrt(Delta0)*gauss_points, offset)
    return gaussian_norm * np.dot(integrand,gauss_weights)

def Phi_shif(mu, Delta0, offset = 0):
    # integrand = np.tanh(mu+np.sqrt(delta0)*gauss_points)
    integrand = phi_shif_tanh(mu + np.sqrt(Delta0)*gauss_points, offset)
    return gaussian_norm * np.dot(integrand,gauss_weights)

def Prime(mu, Delta0, offset = 0):
    integrand = phi_derivative(mu + np.sqrt(Delta0)*gauss_points, offset)
    return gaussian_norm * np.dot(integrand, gauss_weights)


def Integrate(x_inp, t, J, I, offset=0, tau = 0.1):
    dx_inpdt = (-x_inp + np.dot(J, phi_tanh(x_inp, offset)) + I)/tau
    return dx_inpdt

def Integrate_shif(x_inp, t, J, I , offset = 0, tau = 0.1):
    dx_inpdt = (-x_inp + np.dot(J, phi_shif_tanh(x_inp, offset)) + I)/tau
    return dx_inpdt

def rate_non_lin_Mmn(offset = 1.7, Mn_array = np.linspace(0, 2, 20), Mm_array = np.asarray([2]),
                     Sm=2, Sn=2, Mn_example = 2, T=100, dt =0.1,  N=1000, N_trials = 1):

    rate_dict = {}
    seed_trials = np.random.randint(0, 1000, N_trials)
    seed_trials = [0,1]
    np.random.seed(0)
    # print(seed_trials)
    N_total = N

    # ======================== SIMULATIONS, COMPUTING KAPPA FROM X values of INPUTS ====================================
    x0_array = [np.random.normal(-10, 10, N_total), np.random.normal(0, 1, N_total), np.random.normal(10, 1, N_total),
                np.random.normal(2.5, 10, N_total)]

    global_v = np.ones(N_total)
    Proj_glob = []

    Kappa_all_sim = []
    Kappa_all_shif_sim = []
    ind_end = int(0.2*T/dt)

    # x0 = x0_array[1]
    for Mm in Mm_array:
        for Mn in Mn_array:
            Kappa_sim = []
            Kappa_shif_sim = []
            Proj_g = []

            for seed_t in seed_trials:
                np.random.seed(seed_t)

                x = np.random.normal(0, 1, N_total)
                ones = np.ones(N_total)
                x = x - (np.dot(x,ones)/np.dot(ones,ones))*ones

                y = np.random.normal(0, 1, N_total)
                y = y - (np.dot(y,ones)/np.dot(ones,ones))*ones - (np.dot(y,x)/np.dot(x,x))*x

                m = Mm + Sm * x
                n = Mn + Sn * y
                J = np.outer(m, n)/N

                MmMn_array = np.outer(Mm_array, Mn_array).flatten()
                for x0 in x0_array:

                    t = np.linspace(0, T, int(T/dt))

                    solv_x = scipy.integrate.odeint(Integrate, x0, t, args=(J, np.zeros(N_total), offset))
                    K_sim = np.dot(n, phi_tanh(np.mean(solv_x[-ind_end:,:], axis=0), offset=offset).transpose())/N
        # =======================
                    solv_x_shif = scipy.integrate.odeint(Integrate_shif, x0, t, args=(J, np.zeros(N_total), offset))
                    K_shif_sim = np.dot(n, phi_shif_tanh(np.mean(solv_x_shif[-ind_end:,:], axis=0), offset=offset).transpose())/N

                    Kappa_sim.append(K_sim)
                    Kappa_shif_sim.append(K_shif_sim)

                    Proj_g_curr = np.dot(global_v, phi_shif_tanh(np.mean(solv_x_shif[-ind_end:,:], axis=0), offset = offset))/N
                    Proj_g.append(Proj_g_curr)


                # plt.plot(t, solv_x[:,0])
            # MmMn = np.asarray(MmMn)
            # ind_sort = np.argsort(MmMn)

            Kappa_sim = np.asarray(Kappa_sim)
            Kappa_shif_sim = np.asarray(Kappa_shif_sim)
            Proj_glob.append(np.asarray(Proj_g))
            # Kappa_all_sim.append(np.asarray(Kappa_sim[ind_sort]))
            # Kappa_all_shif_sim.append(np.asarray(Kappa_shif_sim[ind_sort]))
            Kappa_all_sim.append(np.asarray(Kappa_sim))
            Kappa_all_shif_sim.append(np.asarray(Kappa_shif_sim))

    rate_dict['Kappa tanh sim'] = np.asarray(Kappa_all_sim).transpose()
    rate_dict['Kappa shif tanh sim'] = np.asarray(Kappa_all_shif_sim).transpose()
    # rate_dict['Mmn array'] = MmMn[ind_sort]
    rate_dict['Mmn array'] = Mm*Mn_array
    rate_dict['Proj global'] = np.asarray(Proj_glob).transpose()
    Kappa_all = []
    Kappa_all_shif = []

    K0_array = [-2, 2, -1, 0.5]
    for K0 in K0_array:
        MmMn = []

        Kappa = []
        Kappa_shif = []

        for Mm in Mm_array:
            m = Mm + Sm * x
            for Mn in Mn_array:
                n = Mn + Sn * y
                MmMn.append(Mm*Mn)
                J = np.outer(m, n)/N

                t = np.linspace(0, T, int(T/dt))

                K = np.zeros(len(t))
                K_shif = np.zeros(len(t))
                K[0] = K0
                K_shif[0] = K0

                for k in range(0, len(t)-1):

                    mu = K[k]*Mm
                    Delta0 = Sm**2*K[k]**2
                    K[k+1] = K[k] + dt*(-K[k] + Mn*Phi(mu, Delta0, offset = offset))

                for k in range(0, len(t)-1):

                    mu_shif = K_shif[k]*Mm
                    Delta0_shif = Sm**2*K_shif[k]**2
                    K_shif[k+1] = K_shif[k] + dt*(-K_shif[k] + Mn*Phi_shif(mu_shif, Delta0_shif, offset = offset)
                                                   + K_shif[k]* Sm**2*Prime(mu_shif, Delta0_shif, offset = offset))

                Kappa.append(np.mean(K[-ind_end:])*np.dot(m,m)/np.dot(m,n))
                Kappa_shif.append(np.mean(K_shif[-ind_end:])*np.dot(m,m)/np.dot(m,n))

        MmMn = np.asarray(MmMn)
        ind_sort = np.argsort(MmMn)

        Kappa = np.asarray(Kappa)
        Kappa_shif = np.asarray(Kappa_shif)

        Kappa_all.append(np.asarray(Kappa[ind_sort]))
        Kappa_all_shif.append(np.asarray(Kappa_shif[ind_sort]))


    rate_dict['Kappa tanh'] = np.asarray(Kappa_all).transpose()
    rate_dict['Kappa shif tanh'] = np.asarray(Kappa_all_shif).transpose()
    rate_dict['Mmn array'] = MmMn[ind_sort]

#  intersection for overlap = 2, function tanh
    Mm = Mm_array[0]
    Mn = Mn_example
    K = np.linspace(-5, 10, int(1e6)+1)

    f1 = K
    f2 = []
    m = Mm + Sm * x
    n = Mn + Sn * y

    for k in K:

        mu = k*Mm
        Delta0 = k**2*Sm**2
        f2.append(Mn*Phi(mu, Delta0, offset = offset)*np.dot(m,m)/np.dot(m,n))

    rate_dict['intersect, overl=2, F1'] = f1*np.dot(m,m)/np.dot(m,n)
    rate_dict['intersect, overl=2, F2'] = f2

    #  intersection for overlap = 2, shifted function tanh

    f1 = K
    f2 = []

    for k in K:
        mu = k*Mm
        Delta0 = k**2*Sm**2
        f2.append(Mn*Phi_shif(mu, Delta0, offset = offset))

    rate_dict['intersect, overl=2, F1_shif'] = f1
    rate_dict['intersect, overl=2, F2_shif'] = f2

    # activity x

    Mm = Mm_array[0]
    Mn = Mn_example
    m = Mm + Sm * x
    n = Mn + Sn * y

    J = np.outer(m, n)/N


    t = np.linspace(0, T, int(T/dt))
    K_shif_all = []
    K_all = []
    x_shif_all = []
    # print('MmMn_example='+str(Mm*Mn_example))

    for x0 in x0_array:

        solv_x = scipy.integrate.odeint(Integrate, x0, t, args=(J, np.zeros(N_total), offset))
        K = np.dot(n, phi_tanh(np.mean(solv_x[-ind_end:,:], axis=0), offset=offset).transpose())/N

        # K = np.dot(m, phi_tanh(np.mean(solv_x[-100:,:],axis=0)).transpose())/N

        solv_x_shif = scipy.integrate.odeint(Integrate_shif, x0, t, args=(J, np.zeros(N_total), offset))
        K_shif = np.dot(n, phi_shif_tanh(np.mean(solv_x_shif[-ind_end:,:], axis=0), offset=offset).transpose())/N
        # K_shif = np.dot(m, phi_shif_tanh(np.mean(solv_x_shif[-100:,:], axis=0), offset=offset).transpose())/N
        # print(K_shif)

        if K < 0.5:
            rate_dict['x lower'] = solv_x
            # print('K low = '+str(K))

        if K >= 3:
            rate_dict['x upper'] = solv_x
            # print('K upp = '+str(K))

        if K_shif < 0.5:
            rate_dict['x lower shif'] = solv_x_shif
            # print('K low shif = '+str(K_shif))

        if K_shif >= 3:
            rate_dict['x upper shif'] = solv_x_shif
            # print('K upp shif = '+str(K_shif))


    rate_dict['vector n, Mmn='+str(Mn*Mm)] = n
    rate_dict['time'] = t
    return rate_dict
