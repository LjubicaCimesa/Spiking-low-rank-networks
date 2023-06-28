import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
plt.rcParams['font.size']=14
plt.rcParams['figure.figsize']= 3,2
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['lines.markersize'] = 4

gaussian_norm = (1/np.sqrt(np.pi))
gauss_points, gauss_weights = np.polynomial.hermite.hermgauss(350)
gauss_points = gauss_points*np.sqrt(2)

def phi_tanh(x, offset = 0):
    temp_x = np.tanh(x-offset)
    return temp_x

def phi_shif_tanh(x, offset = 0):
    temp_x = 1 + np.tanh(x-offset)
    return temp_x

def phi_derivative(x, offset = 0):
    der = 1-np.tanh(x-offset)**2
    return der

def phi_third_derivative(x, offset = 0):
    der = 2*(1-phi_tanh(x,offset)**2)*(3*phi_tanh(x,offset)**2-1)
    return der

def Prime(mu, Delta0, offset = 0):
    integrand = phi_third_derivative(mu + np.sqrt(Delta0)*gauss_points, offset)
    return gaussian_norm * np.dot(integrand, gauss_weights)

def Trime(mu, Delta0, offset = 0):
    integrand = phi_third_derivative(mu + np.sqrt(Delta0)*gauss_points, offset)
    return gaussian_norm * np.dot(integrand, gauss_weights)

def Integrate(x_inp, t, J, I, offset = 0, tau = 0.1):
    dx_inpdt = (-x_inp + np.dot(J, phi_tanh(x_inp, offset)) + I)/tau
    return dx_inpdt

def Integrate_shif(x_inp, t, J, I, offset = 0, tau = 0.1):
    dx_inpdt = (-x_inp + np.dot(J, phi_shif_tanh(x_inp, offset)) + I)/tau
    return dx_inpdt


def rate_non_lin_Smn(offset = 0, Smn_array = np.linspace(0,4,30), Sm = 2, Sn = 2,
                          Smn_example = 2, T=100, dt = 0.1, N=1000, N_trials = 1):

    rate_dict = {}
    np.random.seed(1)
    seed_trials = np.random.randint(0, 1000, N_trials)

    N_total = N
    M_m = 0
    M_n = 0

    # Sm = 2
    # Sn = 2

    # ======================== SIMULATIONS, COMPUTING KAPPA FROM X values of INPUTS ====================================
    x0_array = [np.random.normal(-10, 10, N_total), np.random.normal(10, 5, N_total),
               np.random.normal(2.5, 10, N_total), np.random.normal(1, 10, N_total)]
    x0_array = [np.random.normal(-10, 10, N_total), np.random.normal(10, 5, N_total)]

    # x0_array = [np.random.normal(1, 10, N_total)]

    global_v = np.ones(N_total)
    Proj_glob = []

    Kappa_all_sim = []
    Kappa_all_shif_sim = []
    ind_end = int(0.2*T/dt)

    for i in range(0, len(Smn_array)):
        Smn = Smn_array[i]
        Kappa_sim = []
        Kappa_shif_sim = []
        Proj_g = []

        for seed_t in seed_trials:
            x0 = x0_array[1]
            np.random.seed(seed_t)

            x = np.random.normal(0, 1, N_total)
            ones = np.ones(N_total)
            x = x - (np.dot(x,ones)/np.dot(ones,ones))*ones

            y = np.random.normal(0, 1, N_total)
            y = y - (np.dot(y,ones)/np.dot(ones,ones))*ones - (np.dot(y,x)/np.dot(x,x))*x

            # m, n = np.random.multivariate_normal([0,0],[[Sm**2, Smn],[Smn, Sn**2]],N_total).transpose()
            # m = m - (np.dot(m,ones)/np.dot(ones,ones))*ones
            # n = n - (np.dot(n,ones)/np.dot(ones,ones))*ones
            m = Sm*x
            n = (Smn/Sm)*x + np.sqrt(Sn**2 - (Smn/Sm)**2)*y
            J = np.outer(m,n)/N

            for x0 in x0_array:
                # dt = 0.001
                t = np.linspace(0, T, int(T/dt))
                solv_x = np.zeros((int(T/dt), N))
                solv_x[0,:] = x0
                for tk in range(0,int(T/dt)-1):
                    solv_x[tk+1,:] = solv_x[tk,:] + dt*(-solv_x[tk,:] + np.dot(J, phi_tanh( solv_x[tk,:], offset = offset) ))

                # solv_x = scipy.integrate.odeint(Integrate, x0, t, args=(J, np.zeros(N_total), offset))
                K_sim = np.dot(m, phi_tanh(np.mean(solv_x[-ind_end:,:],axis=0), offset).transpose())/N

                # solv_x_shif = scipy.integrate.odeint(Integrate_shif, x0, t, args=(J, np.zeros(N_total), offset))

                solv_x_shif = np.zeros((int(T/dt), N))
                solv_x_shif[0,:] = x0
                for tk in range(0,int(T/dt)-1):
                    solv_x_shif[tk+1,:] = solv_x_shif[tk,:] + dt*(-solv_x_shif[tk,:] + np.dot(J, phi_shif_tanh(solv_x_shif[tk,:], offset = offset) ))

                K_shif_sim = np.dot(n, phi_shif_tanh(np.mean(solv_x_shif[-ind_end:,:],axis=0), offset).transpose())/N


                Kappa_sim.append(K_sim)
                Kappa_shif_sim.append(K_shif_sim)


                Proj_g_curr = np.dot(global_v, phi_shif_tanh(np.mean(solv_x_shif[-ind_end:,:], axis=0), offset=offset))/N
                # print(Proj_g_curr)
                Proj_g.append(Proj_g_curr)

                # print(K)
                # print(i)

        Kappa_all_sim.append(Kappa_sim)
        Kappa_all_shif_sim.append(Kappa_shif_sim)
        Proj_glob.append(np.asarray(Proj_g))


    rate_dict['x_shif'] = solv_x_shif
    rate_dict['vector_n'] = n
    rate_dict['vector_m'] = m

    rate_dict['Kappa tanh sim'] = np.asarray(Kappa_all_sim).transpose()
    rate_dict['Kappa shif tanh sim'] = np.asarray(Kappa_all_shif_sim).transpose()
    rate_dict['Smn array'] = Smn_array

    rate_dict['Proj global'] = np.asarray(Proj_glob).transpose()

    Smn = Smn_example
    K = np.linspace(-15, 15, int(1e3))

    f1 = K
    f2 = []
    P = []

    # for k in K:
    #     Delta0 = k**2*Sm**2
    #     f2.append(k*Smn*Prime(0, Delta0, offset=offset))
    #     P.append(Prime(0, Delta0, offset=offset))
    # rate_dict['intersect, overl=2, F1'] = f1
    # rate_dict['intersect, overl=2, F2'] = f2
    #
    #     #  intersection for overlap = 2, shifted function tanh
    #
    # f1 = K
    # f2 = []
    #
    # for k in K:
    #     Delta0 = k**2*Sm**2
    #     f2.append(k*Smn*Prime(0, Delta0, offset))
    #
    # rate_dict['intersect, overl=2, F1_shif'] = f1
    # rate_dict['intersect, overl=2, F2_shif'] = f2


    m = Sm*x
    Smn = Smn_example
    # print('Smn _ example '+str(Smn_example))
    n = np.sqrt(Sn**2 - (Smn/Sm)**2) * y + (Smn/Sm) * x

    J = np.outer(m, n)/N

    # T = 5 # Total time of integration, expressed in time constants of single units
    # dt = 0.01
    # t = np.linspace(0, T, int(T/dt))

    # K0_array = [-2,2, 0.05]
    x0_array = [np.random.normal(-10, 10, N_total), np.random.normal(10, 5, N_total),
                np.random.normal(2.5, 10, N_total), np.random.normal(1, 10, N_total),
                np.random.normal(-5, 10, N_total)]
    # print('Smn_example = '+ str(Smn_example))
    for x0 in x0_array:

        solv_x = scipy.integrate.odeint(Integrate, x0, t, args=(J, np.zeros(N_total), offset))
        solv_x_shif = scipy.integrate.odeint(Integrate_shif, x0, t, args=(J, np.zeros(N_total), offset))

        K = np.dot(n, phi_tanh(np.mean(solv_x[-ind_end:,:],axis=0), offset=offset).transpose())/N
        K_shif = np.dot(n, phi_shif_tanh(np.mean(solv_x_shif[-100:,:],axis=0), offset).transpose())/N

        # print('Kappa shift example')
        # print(K_shif)
        if K < -0.5:
            rate_dict['x lower'] = solv_x
            # print('K low = '+str(K))
        if K >= 0.5:
            rate_dict['x upper'] = solv_x
            # print('K upp = '+str(K))

        if K_shif < -0.5:
            rate_dict['x lower shif'] = solv_x_shif
            # print('K low shif = '+str(K_shif))
        if K_shif >= 0.5:
            rate_dict['x upper shif'] = solv_x_shif
            # print('K upp shif = '+str(K_shif))

    rate_dict['time'] = t


    return rate_dict


