import warnings
warnings.filterwarnings('ignore')
import numpy as np
from numpy import dot
from numpy.linalg import inv
import matplotlib.pyplot as plt
from scipy.stats import norm as normal_dist
from sklearn.preprocessing import PolynomialFeatures
from sklearn.mixture import GaussianMixture
from utils import *

ROOT = 'python_data/2Dletters/'

# Locally weighted regression (LWR)
def LWR_traj(letter):
    n_states = 20
    poly_deg = 2 #Degree of the polynomial
    n_out = 2 #number of motion variables
    n_data = 200 #length of trajectory
    n_samples = 9 #number of demonstrations
    t_in = np.linspace(0,1,n_data) #input data for LWR

    #  Load Data
    data = np.load(ROOT + letter + '.npy')[1:n_samples+1]
    # construct the output Y by concatenating all demonstrations
    data = data.transpose([0,2,1])
    Y = np.concatenate(data,axis=0)

    #  Set the basis functions
    t_sep = np.linspace(-0.3,1.3,n_states+1)
    mus = np.zeros(n_states)
    for i in range(n_states):
        mus[i] = 0.5*(t_sep[i]+t_sep[i+1])

    sigmas = np.array([2e-3]*n_states)

    #  Compute the activation weigths
    H = np.zeros((n_states, n_data))
    for i in range(n_states):
        H[i] = normal_dist(loc = mus[i], scale = np.sqrt(sigmas[i])).pdf(t_in)
    H /= np.sum(H,axis=0) #normalizing the weights

    Hn = np.tile(H,(1,n_samples)) #repeat Hn for n samples

    #  Compute LWR
    # construct the polynomial input of degree=poly_deg
    poly = PolynomialFeatures(degree=poly_deg)
    Xr = poly.fit_transform(t_in[:,None])
    X = np.tile(Xr,(n_samples,1))

    As = []
    for i in range(n_states):
        W = np.diag(Hn[i])
        A = dot(inv(dot(X.T,dot(W,X))+np.eye(poly_deg+1)*1e-5),dot(X.T, dot(W,Y)))
        As.append(A)

    lwr_traj = []
    for i in range(n_states):
        y = np.multiply(H[i][:,None], dot(Xr,As[i]))
        lwr_traj.append(y)

    lwr_traj = np.array(lwr_traj)
    lwr_traj = np.sum(lwr_traj, axis=0)

    # Plot trajectory
    plt.figure()
    for i in range(n_samples):
        plt.plot(data[i,:,0], data[i,:,1], alpha = 0.3)
    plt.plot(lwr_traj[:,0], lwr_traj[:,1], linewidth=2)
    plt.show()

    return lwr_traj


#  Gaussian Mixture Regression (GMR)
def GMR_traj(letter, n_states=20):
    n_out = 2  # number of motion variables
    n_data = 200 #length of trajectory
    n_samples = 9 #number of demonstrations
    t_in = np.linspace(0,1,n_data) #input data for LWR

    #  Load Data
    data = np.load(ROOT + letter + '.npy')[1:n_samples+1]
    # construct the output Y by concatenating all demonstrations
    data = data.transpose([0,2,1])

    #  Concatenate time to the data
    data_time = np.zeros((data.shape[0], data.shape[1], data.shape[2] + 1))
    for i in range(n_samples):
        data_time[i] = np.hstack([t_in[:, None], data[i]])
    # concatenate the whole samples
    data_time = np.concatenate(data_time, axis=0)

    # Estimate GMM using the data
    gmm = GaussianMixture(n_components=n_states, n_init=4)
    gmm.fit(data_time)

    # GMR based on the GMM
    gmr = GMR(gmm, n_in=1, n_out=2)

    #  Predict the data based on the time input
    gmr_traj = []
    covs = []
    for t in t_in:
        y, cov = gmr.predict(t)
        gmr_traj.append(y)
        covs.append(cov)
    gmr_traj = np.array(gmr_traj)

    # Plot trajectory
    plt.figure()
    for i in range(n_samples):
        plt.plot(data[i,:,0], data[i,:,1], alpha = 0.3)
    plt.plot(gmr_traj[:,0], gmr_traj[:,1], linewidth=2)
    plt.show()

    return gmr_traj


#  Dynamical movement primitives (DMP) with Gaussian Mixture Regression (GMR)
def DMP_GMR_traj(letter, n_states=20):
    n_in = 1  # Number of variables for the radial basis function [s] (decay term)
    n_out = 2 # Number of motion variables [xi,x2]
    Kp = 50 #Stiffness Gain
    Kv = np.sqrt(2*Kp) #Damping gain with ideal underdamped damping ratio
    alpha = 1. #Decay factor
    dt = 0.01 #Length of each trajectory
    n_data = 200 #length of trajectory
    n_samples = 9 #number of demonstrations
    # L = np.hstack([np.eye(n_out)*Kp, np.eye(n_out)*Kv]) #feedback terms
    # t_in = np.arange(0,n_data*dt,dt) #time

    K = np.array([1., Kv/Kp, 1./Kp])
    K = np.kron(K, np.eye(n_out)) #transformation matrix to compute r(1).currTar = x + dx*kV/kP + ddx/kP

    # Load Data
    data = np.load(ROOT + letter + '.npy')[1:n_samples+1]
    # construct the output Y by concatenating all demonstrations
    data = data.transpose([0,2,1])
    Y = np.concatenate(data,axis=0)

    pos_trajs = data.copy()
    vel_trajs = np.gradient(pos_trajs, axis=1)/dt
    acc_trajs = np.gradient(vel_trajs, axis=1)/dt
    trajs = np.concatenate([pos_trajs, vel_trajs, acc_trajs],axis=2)

    x_targets = []
    for i in range(n_samples):
        x_target = np.dot(K, trajs[i].T).T
        x_targets.append(x_target)
    x_targets = np.array(x_targets)

    # Estimate GMM from the concatenated data [s_in, x_targets]
    s_in = np.zeros(n_data) #decay terms
    s_in[0] = 1.
    for i in range(1,n_data):
        s_in[i] = s_in[i-1] - alpha*s_in[i-1]*dt
    X = np.tile(s_in, (1,n_samples)).T
    Y = np.concatenate(x_targets,axis=0)
    data_joint = np.concatenate([X,Y],axis=1)

    gmm = GaussianMixture(n_components = n_states,n_init = 4)
    gmm.fit(data_joint)

    # Gaussian mixture regression
    gmr = GMR(gmm, n_in = 1, n_out = 2)

    # Predict the data based on the time input
    y_preds = []
    covs = []
    for s in s_in:
        y,cov = gmr.predict(s)
        y_preds.append(y)
        covs.append(cov)
    y_preds = np.array(y_preds)

    # Data Reconstruction using DMP
    y = data[0,0,:]
    dy = np.zeros((1,n_out))

    dmp_gmr_traj = []
    for t in range(n_data):
        y_target = y_preds[t]
        ddy = Kp*(y_target-y) - Kv*dy
        dy = dy + ddy*dt
        y = y + dy*dt
        dmp_gmr_traj.append(y)
    dmp_gmr_traj = np.concatenate(dmp_gmr_traj)

    # Plot trajectory
    plt.figure()
    for i in range(n_samples):
        plt.plot(data[i,:,0], data[i,:,1], alpha = 0.3)
    plt.plot(dmp_gmr_traj[:,0], dmp_gmr_traj[:,1], linewidth=2)
    plt.show()

    return dmp_gmr_traj