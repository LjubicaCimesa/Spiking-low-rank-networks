import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
plt.rcParams['font.size']=14
plt.rcParams['figure.figsize']= 3,2
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['lines.markersize'] = 4

gaussian_norm = (1/np.sqrt(np.pi))
gauss_points, gauss_weights = np.polynomial.hermite.hermgauss(300)
gauss_points = gauss_points*np.sqrt(2)

def phi_shif_tanh(x):
    temp_x = np.tanh(x)
    return temp_x

def phi_derivative(x):
    der = 1-np.tanh(x)**2
    return der

def Prime(mu, Delta0):

    Delta0z = np.sqrt(Delta0)*gauss_points
    integrand = phi_derivative(mu+Delta0z)

    return gaussian_norm * np.dot(integrand, gauss_weights)

def Integrate(x_inp, t, J, I):
    # dx_inpdt = -x_inp + np.dot(J, phi_ReLU(x, lower_bound, upper_bound)) + I
    dx_inpdt = -x_inp + np.dot(J, phi_shif_tanh(x_inp)) + I

    return dx_inpdt

np.random.seed(1)

N = 1000
N_total = N


ones = np.ones(N_total)
m = np.random.normal(0,1,N_total)
n = np.random.normal(0,1,N_total)
m = m - (np.dot(m,ones)/np.dot(ones,ones))*ones
n = n - (np.dot(n,ones)/np.dot(ones,ones))*ones - (np.dot(n,m)/np.dot(m,m))*m
I = np.random.normal(0,1,N_total)
I = I - (np.dot(I,ones)/np.dot(ones,ones))*ones - (np.dot(I,m)/np.dot(m,m))*m - (np.dot(I,n)/np.dot(n,n))*n

direction_all = [I, n]

# c_array = np.linspace(0,2,10)
c_array = [0, 0.7]
x0_array = [np.random.normal(2.0,0.1,N_total), np.random.normal(-2.0,0.1,N_total)]
x0_array = [ np.random.normal(2.0,0.1,N_total)]

PCA_0 = []
PCA_1 = []
PCA_activity_all = []
PCA_rate_all = []
x0 = np.random.normal(2.0,0.1,N_total)
PCA_all = {}

for c in c_array:
    for ind in range(0,2):

        I_dir = direction_all[ind]
        T = 10
        dt = 0.1
        t = np.linspace(0, T, int(T/dt))
        J = np.outer(m,n)/N
        solv_x = scipy.integrate.odeint(Integrate, x0, t, args=(J, c*I_dir))
        # K = np.dot(n, phi_shif_tanh(np.mean(solv_x[-100:,:],axis=0)).transpose())/N
        K = np.dot(m, np.mean(solv_x[-100:,:],axis=0).transpose())/N

        activity=solv_x

    # ======================= PCA for inputs X =======================
        X = activity-np.mean(activity, axis = 0)

        C = np.cov(X.transpose())  # Compute covariance matrix
        C = np.dot(X.T, X) / (X.shape[0] - 1)

        Lambda, eigvec = np.linalg.eig(C)
        Lambda = Lambda.real
        eigvec = eigvec.real

        Lambda_sort = Lambda[(np.argsort(-Lambda))]
        E = eigvec[:,np.argsort(-Lambda)]
        X_prim = np.dot(X,E)
        C_prim = np.dot(X_prim.transpose(), X_prim)

        PCAs_x = []
        for i in range(0,8):
            PCAs_x.append(C_prim[i,i]/np.trace(C_prim))
        PCA_activity_all.append(PCAs_x)
        # plt.figure()
        # plt.bar(x=range(0,8),height=PCAs_x)
        # plt.xticks([0,1,2,3,4,5,6,7,8])

    #=============== PCA for rates ===================
        rate = phi_shif_tanh(activity)

        Y = rate - np.mean(rate, axis=0)

        D = np.cov(Y.transpose())
        D = np.dot(Y.T,Y)/(Y.shape[0]-1)

        Lambda, eigvec = np.linalg.eig(D)
        Lambda = Lambda.real
        eigvec = eigvec.real

        Lambda_sort = Lambda[(np.argsort(-Lambda))]
        E = eigvec[:,np.argsort(-Lambda)]
        Y_prim = np.dot(Y,E)
        D_prim = np.dot(Y_prim.transpose(), Y_prim)

        PCAs_y = []
        for i in range(0,8):
            PCAs_y.append(D_prim[i,i]/np.trace(D_prim))
        PCA_rate_all.append(PCAs_y)


        if not ind:
            print('yes')
            PCA_all['orth_activ_PCA, c='+str(c)] = PCAs_x
            PCA_all['orth_rate_PCA, c='+str(c)] = PCAs_y
        else:
            PCA_all['along_activ_PCA, c='+str(c)]=PCAs_x
            PCA_all['along_rate_PCA, c='+str(c)] = PCAs_y


filename = 'Results_low_dim'
import pickle
file_F1 = open(filename, 'wb')
pickle.dump(PCA_all, file_F1)
file_F1.close()
