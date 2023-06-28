from RateNet_Kappa_Smn import *
from RateNet_MmMn import *

plt.rcParams['font.size'] = 24
plt.rcParams['lines.markersize'] = 8
# plt.rcParams["font.family"] = "serif"  # promenila font!!!!!!
plt.rcParams['lines.linewidth'] = 2


H = 3
W = 5
# W = 7.2 the one at overleaf in the moment

plt.rcParams['figure.figsize'] = W, H
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

f1 = plt.figure()
f2 = plt.figure()
f3 = plt.figure()
f4 = plt.figure()
f5 = plt.figure()
f6 = plt.figure()
f7 = plt.figure()
f8 = plt.figure()
f9 = plt.figure()
f10 = plt.figure()

off = 2.7
off = 2.9

T = 20
dt = 0.01

# Mn_array = np.linspace(-1,4,20)
# Mm_array = np.asarray([2])
Mn_array = np.linspace(0.1, 6, 15)
Mm_array = np.asarray([2])
Mn_example = 5

Sm = 2
Sn = 6
Smn_array = np.concatenate((np.linspace(0.1, 5, 10), np.linspace(5, Sm*Sn, 10)))
Smn_example = Smn_array[-2]
N_neuron = 1000

gl_vector = np.ones(N_neuron)

N_trials_Mmn = 4
N_trials_Smn = 8
c_I = 1
rate_res_Mmn = rate_non_lin_Mmn(offset=off, Mn_array=Mn_array, Mm_array=Mm_array, Sm=Sm, Sn=Sn, Mn_example=Mn_example,
                                T=T, dt = dt, N= N_neuron, N_trials=N_trials_Mmn)
rate_res_Smn = rate_non_lin_Smn(offset=off, Smn_array=Smn_array, Sm=Sm, Sn=Sn, Smn_example=Smn_example,
                                T=T, dt=dt, N=N_neuron, N_trials = N_trials_Smn)

np.random.seed(0)
# random_ind = np.random.randint(0, rate_res_Mmn['x upper'].shape[1], 50)


# =====FIGURE=====

color_u = 'C0'
color_l = 'lightblue'
color_i = 'gray'

N_r = 100

plt.figure(1)
plt.imshow(phi_shif_tanh(rate_res_Mmn['x lower shif'][100:,:N_r].transpose(), offset = off), origin='upper',
           cmap = 'Reds', extent=[0,2,N_r,0], vmin=0, vmax=2)
# plt.colorbar()
plt.axis('tight')

plt.figure(2)
plt.imshow(phi_shif_tanh(rate_res_Mmn['x upper shif'][100:,:N_r].transpose(), offset = off), origin='upper',
           cmap = 'Reds', extent=[0,2,N_r,0], vmin=0, vmax=2 )
# plt.colorbar()
plt.axis('tight')

# plt.xlabel('time (s)')
# plt.ylabel(r'$\phi(y)$')
# plt.yticks([0,1, 2])
# plt.legend()

# ============ when <m>,<n> = 0 =======


plt.figure(3)
plt.imshow(phi_shif_tanh(rate_res_Smn['x lower shif'][100:,:N_r].transpose(), offset = off), origin='upper',
           cmap = 'Reds', extent=[0,2,N_r,0], vmin=0, vmax=1)
# plt.colorbar()
plt.axis('tight')
plt.figure(4)
plt.imshow(phi_shif_tanh(rate_res_Smn['x upper shif'][100:,:N_r].transpose(), offset = off), origin='upper',
           cmap = 'Reds', extent=[0,2,N_r,0], vmin=0, vmax=2 )
# plt.colorbar()
plt.axis('tight')


#  ========= Mmn
# sim.
plt.figure(5)
X_mn = rate_res_Mmn['Kappa shif tanh sim']
Kappa_pos = []
Kappa_neg = []
Mmn_array = rate_res_Mmn['Mmn array']
Gl_Mmn_pos = []
Gl_Mmn_neg = []
Proj_gl_mn = np.asarray(rate_res_Mmn['Proj global'])
# Proj_gl_mn = Proj_gl_mn.reshape(len(Mn_array), int(len(Proj_gl_mn)/len(Mn_array)))

for i in range(0,len(Mmn_array)):
    p = 0
    n = 0
    p_gl = 0
    n_gl = 0
    try:
        p = np.sum(X_mn[X_mn[:,i]>=0.5, i] ) / np.count_nonzero(X_mn[:,i]>=1)
        p_gl = np.sum(Proj_gl_mn[X_mn[:,i]>=0.5, i])/ np.count_nonzero(X_mn[:,i]>=1)
        # print(p_gl)
    except:
        print('No pos for Mmn = ')+str(Mmn_array[i])
    try:
        n = np.sum( X_mn[X_mn[:,i]<0.5, i] ) / np.count_nonzero(X_mn[:,i]<1)
        n_gl = np.sum( Proj_gl_mn[X_mn[:,i]<0.5, i] ) / np.count_nonzero(X_mn[:,i]<1)
        # print(n_gl)
    except:
        print('No neg for Mmn = ')+str(Mmn_array[i])
    Kappa_pos.append(p)
    Kappa_neg.append(n)
    Gl_Mmn_pos.append(p_gl)
    Gl_Mmn_neg.append(n_gl)



plt.plot(rate_res_Mmn['Mmn array'], Kappa_pos, '.',  color = 'C3')
plt.plot(rate_res_Mmn['Mmn array'], Kappa_neg, '.', color = 'k')

plt.figure(9)
plt.plot(rate_res_Mmn['Mmn array'], Gl_Mmn_pos, '.', color = 'C3')
plt.plot(rate_res_Mmn['Mmn array'], Gl_Mmn_neg, '.', color = 'k')

# ======= MONTE CARLO
plt.figure(5)
projs_all = []
# N_MC = 20
projs_pos = []
projs_neg = []

N_MC = 50
R_numbers = np.random.randint(0,10000, N_MC)
ones = np.ones(N_neuron)
K_array = np.linspace(-2,15,1000)
T = 10
dt = 0.01
K_mn_all = []

ind_end = int(0.1*T/dt)
# ind_end = int(0.5*T/dt)
t = np.linspace(0, T, int(T/dt))

K_1 = []
K_fix = []
gl_Mmn_fix = []

for Mn in Mn_array:
    K_mn = []
    # print('Mn = ')

    K1 = []
    gl_Mmn_1 = []

    gl_MN = []

    for K in K_array:
        proj_mean = 0
        proj_gl_mean = 0
        for i in range(0, N_MC):
            # print('i=')
            # print(i)
            np.random.seed(R_numbers[i])
            x = np.random.normal(0, 1, N_neuron)
            x = x - (np.dot(x,ones)/np.dot(ones,ones))*ones

            y = np.random.normal(0, 1, N_neuron)
            y = y - (np.dot(y,ones)/np.dot(ones,ones))*ones - (np.dot(y,x)/np.dot(x,x))*x

            Mm = Mm_array[0]
            m = Mm + Sm * x
            n = Mn + Sn * y

            ind_end = int(0.2*T/dt)
            # proj_r =  np.dot(phi_shif_tanh(K*m, off).transpose(), n)/N_neuron
            proj_r =  np.dot(phi_shif_tanh(K*m, off).transpose(), n)/N_neuron
            proj_gl = np.dot(phi_shif_tanh(K*m, offset=off), gl_vector)/N_neuron

            proj_mean += proj_r
            proj_gl_mean += proj_gl

        proj_mean = proj_mean/N_MC
        proj_gl_mean = proj_gl_mean/N_MC
        # print(proj_mean)
        K1.append(proj_mean)
        gl_Mmn_1.append(proj_gl_mean)

    dK = -K_array + K1
    for j in range(0, len(dK)-1):
        if dK[j]*dK[j+1]<0:
            if dK[j]-dK[j+1] > 0:
                K_mn.append(K_array[j])
                gl_MN.append(gl_Mmn_1[j])
                # print(j)
    K_fix.append(np.asarray(K_mn))
    gl_Mmn_fix.append(np.asarray(gl_MN))
    if Mn == Mn_array[-5]:
        dK_Mmn = dK
        F2_Mmn = K_array


    #
    # plt.figure()
    # plt.plot(K_array, -K_array+K1)
K_up = []
K_down = []
Mmn_up = []
Mmn_down = []
gl_mn_down = []
gl_mn_up = []
for i in range(0, len(Mn_array)):

    if len(K_fix[i])>1:
        if K_fix[i][0]<K_fix[i][1]:
            K_down.append(K_fix[i][0]);
            K_up.append(K_fix[i][1])
            Mmn_down.append(Mm*Mn_array[i]);
            Mmn_up.append(Mm*Mn_array[i])
            gl_mn_down.append(gl_Mmn_fix[i][0]);
            gl_mn_up.append(gl_Mmn_fix[i][1])


    else:
        K_down.append(K_fix[i][0])
        Mmn_down.append(Mm*Mn_array[i])
        gl_mn_down.append(gl_Mmn_fix[i][0])


plt.plot(Mmn_down, K_down, color= 'k')
plt.plot(Mmn_up, K_up, color= 'C3')
# plt.ylabel('Kappa')
# plt.xlabel('overlap (mn)')
plt.figure(9)

plt.plot(Mmn_down, gl_mn_down, color= 'k')
plt.plot(Mmn_up, gl_mn_up, color= 'C3')


# plt.plot(rate_res_Mmn['Mmn array'], rate_res_Mmn['Kappa shif tanh'][:,0], color = color_u)
# plt.plot(rate_res_Mmn['Mmn array'], rate_res_Mmn['Kappa shif tanh'][:,1:], color = color_u)

# ax10.legend()
# plt.xlabel(r'$O(m^{(1)}, n^{(1)})$')
# plt.ylabel(' v')
# plt.legend()


# ================ Smn ==================
plt.figure(6)

# =============== MONTE CARLO
projs_all = []
# N_MC = 20
projs_pos = []
projs_neg = []

N_MC = 50
R_numbers = np.random.randint(0, 10000, N_MC)
ones = np.ones(N_neuron)
# K0_array = [-10, -1, 10, 0.1]

ind_end = int(0.1*T/dt)
# ind_end = int(0.5*T/dt)
K_array = np.linspace(-10, 10, 100)
K_1 = []
K_fix = []
Smn_all = []
Gl_Smn_MC = []
for Smn in Smn_array:
    # print(Smn)
    K_mn = []
    K1 = []
    Smn_fix = []
    g_Smn_MC = []
    g_Smn_fix = []
    for K in K_array:
        proj_mean = 0
        proj_gl_mean = 0
        for i in range(0, N_MC):
            np.random.seed(R_numbers[i])
            x = np.random.normal(0, 1, N_neuron)
            x = x - (np.dot(x,ones)/np.dot(ones,ones))*ones

            y = np.random.normal(0, 1, N_neuron)
            y = y - (np.dot(y,ones)/np.dot(ones,ones))*ones - (np.dot(y,x)/np.dot(x,x))*x

            m = Sm*x
            n = (Smn/Sm)*x + np.sqrt(Sn**2 - (Smn/Sm)**2)*y

            ind_end = int(0.2*T/dt)
            proj_r =  np.dot(phi_shif_tanh(K*m, off).transpose(), n)/N_neuron
            # proj = proj_r*np.dot(m,m)/np.dot(m,n)
            proj=proj_r
            proj_gl = np.dot(phi_shif_tanh(K*m, offset=off), gl_vector)/N_neuron
            # print(proj_gl)

            # print(proj)
            proj_mean += proj
            proj_gl_mean+= proj_gl

        proj_mean = proj_mean/N_MC
        proj_gl_mean = proj_gl_mean/N_MC
        K1.append(proj_mean)
        g_Smn_MC.append(proj_gl_mean)

    dK = -K_array + K1
    for j in range(0, len(dK)-1):
        if dK[j]*dK[j+1]<0:
            # print(dK[j]-dK[j+1])
            if dK[j]-dK[j+1] > 0.01:
                K_mn.append(K_array[j])
                # print(j)
                Smn_fix.append(Smn)
                g_Smn_fix.append(g_Smn_MC[j])
                # print(g_Smn_MC[j])


    K_fix.append(K_mn)
    Smn_all.append(Smn_fix)
    Gl_Smn_MC.append(g_Smn_fix)

    if (Smn == Smn_array[-5]):
        dK_Smn = dK
        F2_Smn = K_array


# ===== SIM
K_up = []
K_down = []
K_zero = []
Smn_up = []
Smn_down = []
Smn_zero = []
Gl_Smn_up = []
Gl_Smn_down = []
Gl_Smn_zero = []

for i in range(0, len(Smn_array)):
    for k in range(0,len(K_fix[i])):
        K_c = K_fix[i][k]
        Gl = Gl_Smn_MC[i][k]
        if K_c > 0.5:
            K_up.append(K_c);
            Smn_up.append(Smn_array[i])
            Gl_Smn_up.append(Gl)
        if K_c < -0.5:
            K_down.append(K_c);
            Smn_down.append(Smn_array[i])
            Gl_Smn_down.append(Gl)
        if np.logical_and(K_c<0.5, K_c>-0.5):
            K_zero.append(K_c);
            Smn_zero.append(Smn_array[i])
            Gl_Smn_zero.append(Gl)

plt.figure(6)
plt.plot(Smn_down, K_down, color= 'C0')
plt.plot(Smn_up, K_up, color= 'C3')
plt.plot(Smn_zero, K_zero, color= 'k')

plt.figure(10)
plt.plot(Smn_down, Gl_Smn_down, color= 'C0')
plt.plot(Smn_up, Gl_Smn_up, color= 'C3')
plt.plot(Smn_zero, Gl_Smn_zero, color= 'k')


#======= SIM.
X = rate_res_Smn['Kappa shif tanh sim']
Kappa_pos = []
Kappa_neg = []
Kappa_zero = []


Gl_Smn_up = []
Gl_Smn_down = []
Gl_Smn_zero = []
Proj_gl_mn = np.asarray(rate_res_Smn['Proj global'])

for i in range(0, len(Smn_array)):
    p = 0
    n = 0
    p_gl = 0
    n_gl = 0
    try:
        p = np.sum( X[X[:,i]>=1, i] ) / np.count_nonzero(X[:,i]>=1)
        p_gl = np.sum(Proj_gl_mn[X[:,i]>=1, i])/np.count_nonzero(X[:,i]>=1)

    except:
        print('No pos for Smn = '+str(Smn_array[i]))
    try:
        n = np.sum( X[X[:,i]<-1, i] ) / np.count_nonzero(X[:,i]<-1)
        n_gl = np.sum(Proj_gl_mn[X[:,i]<-1, i])/np.count_nonzero(X[:,i]<-1)

    except:
        print('No neg for Smn = '+str(Smn_array[i]))
    try:
        cond = np.logical_and(X[:,i]>-1, X[:,i]<1)
        print(X[cond,i])
        z = np.sum(X[cond,i]) / np.count_nonzero(cond)
        n_zero = np.sum(Proj_gl_mn[cond, i])/np.count_nonzero(cond)
    except:
        print('No zero for Smn = ' +str(Smn_array[i]))

    Kappa_pos.append(p)
    Kappa_neg.append(n)
    Kappa_zero.append(z)
    Gl_Smn_up.append(p_gl)
    Gl_Smn_down.append(n_gl)
    Gl_Smn_zero.append(n_zero)

plt.figure(6)
plt.plot(rate_res_Smn['Smn array'], Kappa_pos,'.',  color = 'C3')
plt.plot(rate_res_Smn['Smn array'], Kappa_neg, '.',  color = 'C0')
plt.plot(rate_res_Smn['Smn array'], Kappa_zero, '.', color = 'k')


plt.figure(10)
plt.plot(rate_res_Smn['Smn array'], Gl_Smn_down, '.', color = 'C0')
plt.plot(rate_res_Smn['Smn array'], Gl_Smn_up, '.',  color = 'C3')
plt.plot(rate_res_Smn['Smn array'], Gl_Smn_zero, '.',  color = 'k')

# plt.plot(rate_res_Smn['Smn array'], rate_res_Smn['Kappa shif tanh sim'][:,0], '.', color = 'black', markersize = 6, label = 'sim')
# plt.plot(rate_res_Smn['Smn array'], rate_res_Smn['Kappa shif tanh sim'][:,1:], '.', color = 'black', markersize = 6)

# plt.xlabel(r'$O(m^{(1)}, n^{(1)})$')
# plt.ylabel('v ')
# plt.legend()

plt.figure(7)
# plt.plot(F2_Mmn, dK_Mmn, color = 'k')
# plt.plot(F2_Mmn, np.zeros(len(F2_Mmn)), '--', color = 'y')
plt.plot(F2_Mmn, dK_Mmn+F2_Mmn, color = 'gray')
plt.plot(F2_Mmn, np.zeros(len(F2_Mmn))+F2_Mmn, '--', color = 'y')
for j in range(0, len(F2_Mmn)-1):
    if dK_Mmn[j]*dK_Mmn[j+1]<0:
        if dK_Mmn[j]-dK_Mmn[j+1] > 0.01:
            plt.plot(F2_Mmn[j], dK_Mmn[j]+F2_Mmn[j], '.', color = 'r', markersize= 10)
plt.xticks([min(F2_Mmn), 10])
ind_lim = np.min(np.argwhere(F2_Mmn>10))
plt.xlim(min(F2_Mmn), F2_Mmn[ind_lim])

# +F2_Mmn added because we look at the RHS = Kappa rather than dK = 0

plt.ylim([-1,10])
plt.figure(8)
plt.plot(F2_Smn, dK_Smn+F2_Smn, color='gray', markersize = 6)
plt.plot(F2_Smn, np.zeros(len(F2_Smn))+F2_Smn, '--', color = 'y')
for j in range(0, len(F2_Smn)-1):
    if dK_Smn[j]*dK_Smn[j+1]<0:
        if dK_Smn[j]-dK_Smn[j+1] > 0:
            plt.plot(F2_Smn[j+1], dK_Smn[j+1]+F2_Smn[j+1], '.', color = 'r', markersize= 10)

ind_lim1 = np.max(np.where(F2_Smn<-5))
ind_lim2 = np.min(np.where(F2_Smn>5))
plt.xlim(F2_Smn[ind_lim1], F2_Smn[ind_lim2])
plt.ylim([-5,5])

plt.figure(10)

# ==========================================================================================================

from RateNet_Kappa_Smn import *
from RateNet_MmMn import *

plt.rcParams['font.size'] = 24
plt.rcParams['lines.markersize'] = 8
# plt.rcParams["font.family"] = "serif"  # promenila font!!!!!!
plt.rcParams['lines.linewidth'] = 2

H = 3
W = 5
# W = 7.2 the one at overleaf in the moment

plt.rcParams['figure.figsize'] = W, H
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False


f1.set_tight_layout(True)
f2.set_tight_layout(True)
f3.set_tight_layout(True)
f4.set_tight_layout(True)
f5.set_tight_layout(True)

f6.set_tight_layout(True)
f7.set_tight_layout(True)
f8.set_tight_layout(True)
f9.set_tight_layout(True)
f10.set_tight_layout(True)

# f16.set_tight_layout(True)

