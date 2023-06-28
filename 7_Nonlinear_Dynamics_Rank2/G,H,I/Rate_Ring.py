import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
from matplotlib import colors
plt.rcParams['font.size'] = 24
import random
H = 3
W = 5
# W = 7.2 the one at overleaf in the moment

plt.rcParams['figure.figsize'] = W, H
# plt.rcParams['legend.fontsize'] = 8
# plt.rcParams['lines.markersize'] = 4
# plt.rcParams['backend'] = 'QT4Agg'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False


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



seed1 = 1
np.random.seed(seed=seed1)




dt = 0.001
t_run = 10
tau_m = 0.1
t_trz = 1

N_total = 1000
N = 1000
#  number of trials for each network and number of different network realisations

N_trials = 3
N_nets = 15

seed_nets = []
for ind_n in range(0,N_nets):
    seed_nets.append(np.random.randint(0, 2000, 1)[0])
# seed_nets = [599, 1372]
seed_trials = []
x0_all = []
for int_t in range(0,N_trials):
    seed1 = np.random.randint(0, 10000, 1)[0]
    np.random.seed(np.random.randint(0, 10000, 1)[0])
    x0 = np.random.normal(0,10, N_total).transpose()
    x0_all.append(x0)
    seed_trials.append(seed1)


sigma_mn_arr = np.linspace(1.6, 3.5, 1)
sigma_mn_arr = np.linspace(2, 3.5, 1)
sigma_m2n2_const = 2
K1_all = []
K2_all = []
x0_mean_arr = [-2, -1, 0, 1, 2]
N_r = 100

ind_ex = 12

f1 = plt.figure(figsize=[5,5])
f2 = plt.figure()
f3 = plt.figure()
f4 = plt.figure()
f5 = plt.figure()


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

color_list[3] = 'C4'
color_list[4] = 'C3'
K1_diff = []
K2_diff = []
solv_x_all = []

for sigma_mn in sigma_mn_arr:
    # print(sigma_mn)
    sigma_m1n1 = sigma_mn
    # sigma_m2n2 = sigma_mn
    sigma_m2n2 = sigma_m1n1

    sigma_n1n1 = 1.24 + 6
    sigma_n2n2 = 1.63 + 2

    sigma_m1n2 = 0
    sigma_m2n1 = 0

    fk = plt.figure('f'+str(sigma_mn))

    for ind_nets in range(0,N_nets):

        seed_n = seed_nets[ind_nets]
        np.random.seed(seed=seed_n)

        fk = plt.figure('f'+str(sigma_mn))

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

        m_1 = x1
        m_2 = x2

        n_1 = sigma_m1n1 * x1 + np.sqrt(sigma_n1n1**2 - sigma_m1n1**2) * x3
        n_2 = sigma_m2n2 * x2 + np.sqrt(sigma_n2n2**2 - sigma_m2n2**2) * x4


        rank_1 = np.outer(m_1, n_1)/N_total
        rank_2 = np.outer(m_2, n_2)/N_total
        # rank_2 is already in the E_I matrix, that is the mean part we subtract from E_I matrix
        J = rank_1 + rank_2

        for ind_t in range(0, len(seed_trials)):
            # print(ind_t)
            x0 = x0_all[ind_t]
            # np.random.seed(seed=seed_t)
            time_array = np.linspace(0, t_run, int(t_run/dt))
            solv_x = scipy.integrate.odeint(Integrate, x0, time_array, args=(J, np.zeros(N_total), tau_m))
            solv_x_all.append(solv_x)

            K1_curr = np.dot(m_1, phi_tanh(solv_x).transpose())/N
            K2_curr = np.dot(m_2, phi_tanh(solv_x).transpose())/N
            K1_all.append(K1_curr)
            K2_all.append(K2_curr)
            #  plotting
            plt.figure(1)
            ind_cl = ind_nets
            if N_nets==1:
                plt.plot(K1_curr[int(len(K1_curr)*t_trz/t_run):], K2_curr[int(len(K2_curr)*t_trz/t_run):], label = str(sigma_mn), color= color_list[ind_cl])
                plt.plot(np.mean(K1_curr[int(0.9*len(K1_curr))]), np.mean(K2_curr[int(0.9*len(K2_curr))]), '.', markersize = 10, color ='k' )

            if (ind_nets*N_trials + ind_t == ind_ex):
                # break
                # print('ind_ex'+str(ind_nets*N_trials + ind_t ))
                # print('ind_nets'+str(ind_nets))
                # print('ind_t '+str(ind_t))
                plt.plot(K1_curr, K2_curr ,label = str(sigma_mn), color = color_list[ind_cl])
                plt.plot(np.mean(K1_curr[int(0.9*len(K1_curr))]), np.mean(K2_curr[int(0.9*len(K2_curr))]), '.', markersize = 10 , color = color_list[ind_cl])
                # print(color_list[ind_cl])

            else:
                plt.plot(K1_curr[int(len(K1_curr)*t_trz/t_run):], K2_curr[int(len(K2_curr)*t_trz/t_run):], label = str(sigma_mn), color = color_list[ind_cl])
                plt.plot(np.mean(K1_curr[int(0.9*len(K1_curr))]), np.mean(K2_curr[int(0.9*len(K2_curr))]), '.', markersize = 10 , color = color_list[ind_cl])

            K1_diff.append(np.max(K1_curr[int(len(K1_curr)*t_trz/t_run):]) - np.min(K1_curr[int(len(K1_curr)*t_trz/t_run):]) )
            K2_diff.append(np.max(K2_curr[int(len(K2_curr)*t_trz/t_run):]) - np.min(K2_curr[int(len(K2_curr)*t_trz/t_run):]) )

            color_C = 'C' + str(ind_nets)
            # print('show')
            # print(solv_x)
plt.axis('off')


ind_tr = 0
color_ex = color_list[ind_ex//N_trials]
cl_ex_rgba = colors.to_rgba(color_ex)
#
plt.figure(2)
cmap_array = []
# for i in range()
for i in range(0,1000):
    # print('i'+str(i))
    cl_ex_l = cl_ex_rgba[3]*i*0.001
    cmap_array.append([cl_ex_rgba[0], cl_ex_rgba[1], cl_ex_rgba[2] , cl_ex_l])
cmap_ex = colors.ListedColormap(np.asarray(cmap_array))
# cmap = plt.get_cmap("tab10")
# 1f77b4
bounds=np.linspace(0,2,100)
norm = colors.BoundaryNorm(bounds, cmap_ex.N)
rates = phi_shif_tanh(solv_x_all[ind_ex][ind_tr:,:N_r].transpose())
bounds = np.linspace(np.min(rates), np.max(rates), 10)
norm = colors.BoundaryNorm(bounds, cmap_ex.N)
plt.imshow(rates, origin='upper',
   cmap = cmap_ex,   extent=[time_array[ind_tr], 5, N_r,0])
#
# , vmin=0, vmax=5
# norm=norm,
# , vmin=0, vmax=5
plt.show()
plt.axis('tight')
plt.xticks([0,5])
# plt.colorbar()
#
# # , vmin=0, vmax=2
plt.figure(3)
# plt.plot(3)
plt.plot(time_array[ind_tr:], np.mean(phi_shif_tanh(solv_x)[ind_tr:], axis = 1), color = color_ex)
plt.ylim([0,2])

# plt.rcParams['axes.spines.left'] = False

N_curr = 3


ind_ex = 12
plt.figure(4)
K1_curr = K1_all[ind_ex]
# , color = color_list[ind_ex//N_trials]
plt.plot(time_array[ind_tr:], K1_curr[ind_tr:], color = color_ex)
# int(len(time_array)/2)
# int(len(K1_curr)/2)


# plt.yticks([-0.5,0])
# plt.gca().get_yaxis().set_visible(False)
# plt.yticks([0,-0.2])

plt.figure(5)
K2_curr = K2_all[ind_ex]
plt.plot(time_array[ind_tr:], K2_curr[ind_tr:], color = color_ex)
plt.yticks([0,-0.5])
# int(len(time_array)/2)
# int(len(K2_curr)/2

# plt.yticks([0,-0.2])
# plt.yticks([-0.5,0])
# plt.gca().get_yaxis().set_visible(False)

# print('K1 diff = '+str(np.max(np.asarray(K1_diff))))
# print('K2 diff = '+str(np.max(np.asarray(K2_diff))))

f1.set_tight_layout(True)
f2.set_tight_layout(True)
f3.set_tight_layout(True)
f4.set_tight_layout(True)
f5.set_tight_layout(True)
