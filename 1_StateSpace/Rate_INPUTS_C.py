import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
plt.rcParams['font.size']=14
plt.rcParams['figure.figsize']= 3,2
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['lines.markersize'] = 4
plt.rcParams['backend'] = 'QT4Agg'
# for problem with main thread
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
    return gaussian_norm * np.dot(integrand, gauss_weights)

def Phi_shif(mu, Delta0, offset = 0):
    # integrand = np.tanh(mu+np.sqrt(delta0)*gauss_points)
    integrand = phi_shif_tanh(mu + np.sqrt(Delta0)*gauss_points, offset)
    return gaussian_norm * np.dot(integrand, gauss_weights)

def Integrate(x_inp, t, J, I, offset=0, tau = 0.1):
    dx_inpdt = (-x_inp + np.dot(J, phi_tanh(x_inp, offset)) + I)/tau
    return dx_inpdt

def Integrate_shif(x_inp, t, J, I , offset = 0, tau = 0.1):
    dx_inpdt = (-x_inp + np.dot(J, phi_shif_tanh(x_inp, offset)) + I)/tau
    return dx_inpdt

# def phi_tanh(x):
#     temp_x = np.tanh(x)
#     return temp_x
#
# def phi_derivative(x):
#     der = 1-np.tanh(x)**2
#     return der
#
def Prime(mu, Delta0):

    Delta0z = np.sqrt(Delta0)*gauss_points
    integrand = phi_derivative(mu+Delta0z)

    return gaussian_norm * np.dot(integrand, gauss_weights)


def rate_C(c_array = np.linspace(0,2,20), inp_D = 'along', N_neurons =1000, seed_val= 1):
    np.random.seed(seed_val)
    N_total = N_neurons
    N = N_neurons
    ones = np.ones(N_total)
    m = np.random.normal(0, 1, N_total)
    n = np.random.normal(0, 1, N_total)
    m = m - (np.dot(m,ones)/np.dot(ones,ones))*ones
    n = n - (np.dot(n,ones)/np.dot(ones,ones))*ones - (np.dot(n,m)/np.dot(m,m))*m
    J = np.outer(m,n)/N

    I_orth = np.random.normal(0,1,N_total)
    I_orth = I_orth - (np.dot(I_orth,ones)/np.dot(ones,ones))*ones - \
             (np.dot(I_orth,m)/np.dot(m,m))*m - (np.dot(I_orth,n)/np.dot(n,n))*n

    if inp_D == 'along':
        I_dir = n
        print('Input is along n')
    elif inp_D == 'orth':
        I_dir = I_orth
        print('Input is orthogonal to n')
    elif inp_D == 'const':
        print('Input is along (1,1..)')
        I_dir = np.zeros(N)+1
    else:
        print('Input is not chosen properly')


    # x0_array = [np.random.normal(2.0,0.1,N_total), np.random.normal(-2.0,0.1,N_total)]
    # x0_array = [ np.random.normal(2.0,0.1,N_total)]
    x0 = np.random.normal(0,0.5,N_total)

    rate_dict = {}
    rate_dict['c_array'] = c_array
    rate_dict['vector m'] = m
    rate_dict['vector n'] = n
    # rate_dict['vector I'] = I

    x_all = []
    x_all_shift = []
    Kappa = []
    Kappa_shif = []

    for c in c_array:

        t_run = 5
        dt = 0.001
        tau_m = 0.1

        t_beg = 1
        t1 = np.linspace(0, t_beg, int(t_beg/dt))

        solv_x_beg = scipy.integrate.odeint(Integrate, x0, t1, args=(J, np.zeros(N_total), tau_m))
        solv_x_shift_beg = scipy.integrate.odeint(Integrate_shif, x0, t1, args=(J, np.zeros(N_total), tau_m))


        t2=np.linspace(t_beg, t_run, int((t_run-t_beg)/dt))

        x1 = solv_x_beg[-1:].reshape(N_total)
        x1_shif = solv_x_shift_beg[-1:].reshape(N_total)

        solv_x_step = scipy.integrate.odeint(Integrate, x1, t2, args=(J, c*I_dir, tau_m))
        solv_x_shift_step = scipy.integrate.odeint(Integrate_shif, x1_shif, t2, args=(J, c*I_dir, tau_m))

        # K = np.dot(n, phi_tanh(np.mean(solv_x_step[-100:,:],axis=0)).transpose())/N
        # K_shif = np.dot(n, phi_shif_tanh(np.mean(solv_x_shift_step[-100:,:],axis=0)).transpose())/N

        K = np.dot(m, np.mean(solv_x_step[-100:,:],axis=0).transpose())/N
        K_shif = np.dot(m, np.mean(solv_x_shift_step[-100:,:]).transpose())/N

        # this is a mistake, you need to compute x with the shifted function
        # K = np.dot(m, np.mean(solv_x[-100:,:],axis=0).transpose())/N
        Kappa.append(K)
        Kappa_shif.append(K_shif)

        x_all.append(np.concatenate((solv_x_beg, solv_x_step)))
        x_all_shift.append(np.concatenate((solv_x_shift_beg, solv_x_shift_step)))
        time_all = np.concatenate((t1, t2))
        rate_dict['t'] = time_all

    rate_dict['x_all_c'] = x_all
    rate_dict['x_all_shif_c'] = x_all_shift
    rate_dict['vector I'] = I_dir
    rate_dict['vector m'] = m
    rate_dict['vector n'] = n
    rate_dict['Kappa_shif'] = Kappa_shif
    rate_dict['Kappa'] = Kappa

    return rate_dict


