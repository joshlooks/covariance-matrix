import numpy as np
import scipy.stats as st
import scipy.integrate as ig
import scipy.interpolate as ip
from matplotlib import pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
import seaborn as sb
import pandas as pd
from scipy.optimize import least_squares
import copy

def ito_sis_4(betat, gamma, mu, delta, y0, T, dt):
    # Number of time steps￼
    num_steps = int(T / dt)
    time = np.linspace(0, T, num_steps)
    # Initialize arrays to store results
    y_out = np.zeros((num_steps,len(y0)))
    y_out[0,:] = y0

    # Euler-Maruyama method to solve the SDE
    for t in range(1, num_steps):
        S = y_out[t-1,:4]
        I = y_out[t-1,4:]
        N = S + I
        delta_prev = np.array([delta[1],delta[2],delta[3],0])
        s_temp = np.array([0,S[0],S[1],S[2]])
        i_temp = np.array([0,I[0],I[1],I[2]])
        beta = betat(time[t])
        dS = - np.matmul(I/N,beta)*S - mu*S + (mu*N)[::-1] + delta*s_temp - delta_prev*S  + gamma*I
        dI =  np.matmul(I/N,beta)*S - gamma*I - mu*I + delta*i_temp - delta_prev*I

        # Compute the diffusion
        dW = np.random.normal(0, np.sqrt(dt), 29)
        # 29 Wiener processes: S1*4I, S2*4I, S3*4I, S4*4I, 4I recovery, 1 birth, 1S death, 1I death, 3S aging, 3I aging
        S1Infs = [np.sqrt(beta[0,0]*S[0]*I[0]/N[0]),np.sqrt(beta[0,1]*S[0]*I[1]/N[1]),np.sqrt(beta[0,2]*S[0]*I[2]/N[2]),np.sqrt(beta[0,3]*S[0]*I[3]/N[3])]
        S2Infs = [np.sqrt(beta[1,0]*S[1]*I[0]/N[0]),np.sqrt(beta[1,1]*S[1]*I[1]/N[1]),np.sqrt(beta[1,2]*S[1]*I[2]/N[2]),np.sqrt(beta[1,3]*S[1]*I[3]/N[3])]
        S3Infs = [np.sqrt(beta[2,0]*S[2]*I[0]/N[0]),np.sqrt(beta[2,1]*S[2]*I[1]/N[1]),np.sqrt(beta[2,2]*S[2]*I[2]/N[2]),np.sqrt(beta[2,3]*S[2]*I[3]/N[3])]
        S4Infs = [np.sqrt(beta[3,0]*S[3]*I[0]/N[0]),np.sqrt(beta[3,1]*S[3]*I[1]/N[1]),np.sqrt(beta[3,2]*S[3]*I[2]/N[2]),np.sqrt(beta[3,3]*S[3]*I[3]/N[3])]
        Irec = np.sqrt(gamma*I)
        birth = np.sqrt((mu*N)[::-1][0])
        Sdeath = np.sqrt(mu*S)[-1]
        Ideath = np.sqrt(mu*I)[-1]
        Saging = np.sqrt(delta_prev*S)
        Iaging = np.sqrt(delta_prev*I)
        G = np.array([
            [-S1Infs[0],-S1Infs[1],-S1Infs[2],-S1Infs[3], 0,0,0,0, 0,0,0,0, 0,0,0,0, Irec[0],0,0,0, birth, 0, 0, -Saging[0],0,0, 0,0,0],
            [0,0,0,0, -S2Infs[0],-S2Infs[1],-S2Infs[2],-S2Infs[3], 0,0,0,0, 0,0,0,0, 0,Irec[1],0,0, 0, 0, 0, Saging[0],-Saging[1],0, 0,0,0],
            [0,0,0,0, 0,0,0,0, -S3Infs[0],-S3Infs[1],-S3Infs[2],-S3Infs[3], 0,0,0,0, 0,0,Irec[2],0, 0, 0, 0, 0,Saging[1],-Saging[2], 0,0,0],
            [0,0,0,0, 0,0,0,0, 0,0,0,0, -S4Infs[0],-S4Infs[1],-S4Infs[2],-S4Infs[3], 0,0,0,Irec[3], 0, -Sdeath, 0, 0,0,Saging[2], 0,0,0],

            [S1Infs[0],S1Infs[1],S1Infs[2],S1Infs[3], 0,0,0,0, 0,0,0,0, 0,0,0,0, -Irec[0],0,0,0, 0, 0, 0, 0,0,0, -Iaging[0],0,0],
            [0,0,0,0, S2Infs[0],S2Infs[1],S2Infs[2],S2Infs[3], 0,0,0,0, 0,0,0,0, 0,-Irec[1],0,0, 0, 0, 0, 0,0,0, Iaging[0],-Iaging[1],0,],
            [0,0,0,0, 0,0,0,0, S3Infs[0],S3Infs[1],S3Infs[2],S3Infs[3], 0,0,0,0, 0,0,-Irec[2],0, 0, 0, 0, 0,0,0, 0,Iaging[1],-Iaging[2]],
            [0,0,0,0, 0,0,0,0, 0,0,0,0, S4Infs[0],S4Infs[1],S4Infs[2],S4Infs[3], 0,0,0,-Irec[3], 0, 0, -Ideath, 0,0,0, 0,0,Iaging[2],]
        ])
        dS[0] += np.dot(G[0],dW)
        dS[1] += np.dot(G[1],dW)
        dS[2] += np.dot(G[2],dW)
        dS[3] += np.dot(G[3],dW)
        dI[0] += np.dot(G[4],dW)
        dI[1] += np.dot(G[5],dW)
        dI[2] += np.dot(G[6],dW)
        dI[3] += np.dot(G[7],dW)
        # Update values
        y_out[t,:4] = np.maximum(0, S + dS)
        y_out[t,4:] = np.maximum(0, I + dI)
    return time, y_out

def ito_sir_4(betat, gamma, mu, delta, y0, T, dt):
    # Number of time steps
    num_steps = int(T / dt)
    time = np.linspace(0, T, num_steps)
    dt = time[1] - time[0]
    # Initialize arrays to store results
    y_out = np.zeros((num_steps,len(y0)))
    y_out[0,:] = y0

    # Euler-Maruyama method to solve the SDE
    for t in range(1, len(time)):
        S = y_out[t-1,:4]
        I = y_out[t-1,4:8]
        R = y_out[t-1,8:]
        N = S + I + R
        beta = betat(time[t-1])
        delta_prev = np.array([delta[1],delta[2],delta[3],0])
        s_temp = np.array([0,S[0],S[1],S[2]])
        i_temp = np.array([0,I[0],I[1],I[2]])
        r_temp = np.array([0,R[0],R[1],R[2]])
        dS = dt*(-np.matmul(I/N,beta)*S - mu*S + (mu*N)[::-1] + delta*s_temp - delta_prev*S)
        dI = dt*(np.matmul(I/N,beta)*S - gamma*I - mu*I + delta*i_temp - delta_prev*I)
        dR = dt*(gamma*I - mu*R + delta*r_temp - delta_prev*R)

        # Compute the diffusion
        dW = np.random.normal(0, np.sqrt(dt), 33)
        # 33 Wiener processes: S1*4I, S2*4I, S3*4I, S4*4I, 4I recovery, 1 birth, 1S death, 1I death, 1R death, 3S aging, 3I aging, 3R aging
        S1Infs = [np.sqrt(beta[0,0]*S[0]*I[0]/N[0]),np.sqrt(beta[0,1]*S[0]*I[1]/N[1]),np.sqrt(beta[0,2]*S[0]*I[2]/N[2]),np.sqrt(beta[0,3]*S[0]*I[3]/N[3])]
        S2Infs = [np.sqrt(beta[1,0]*S[1]*I[0]/N[0]),np.sqrt(beta[1,1]*S[1]*I[1]/N[1]),np.sqrt(beta[1,2]*S[1]*I[2]/N[2]),np.sqrt(beta[1,3]*S[1]*I[3]/N[3])]
        S3Infs = [np.sqrt(beta[2,0]*S[2]*I[0]/N[0]),np.sqrt(beta[2,1]*S[2]*I[1]/N[1]),np.sqrt(beta[2,2]*S[2]*I[2]/N[2]),np.sqrt(beta[2,3]*S[2]*I[3]/N[3])]
        S4Infs = [np.sqrt(beta[3,0]*S[3]*I[0]/N[0]),np.sqrt(beta[3,1]*S[3]*I[1]/N[1]),np.sqrt(beta[3,2]*S[3]*I[2]/N[2]),np.sqrt(beta[3,3]*S[3]*I[3]/N[3])]
        Irec = np.sqrt(gamma*I)
        birth = np.sqrt((mu*N)[::-1][0])
        Sdeath = np.sqrt(mu*S)[-1]
        Ideath = np.sqrt(mu*I)[-1]
        Rdeath = np.sqrt(mu*R)[-1]
        Saging = np.sqrt(delta_prev*S)
        Iaging = np.sqrt(delta_prev*I)
        Raging = np.sqrt(delta_prev*R)
        G = np.array([
            [-S1Infs[0],-S1Infs[1],-S1Infs[2],-S1Infs[3], 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, birth, 0, 0, 0,  -Saging[0],0,0, 0,0,0, 0,0,0],
            [0,0,0,0, -S2Infs[0],-S2Infs[1],-S2Infs[2],-S2Infs[3], 0,0,0,0, 0,0,0,0, 0,0,0,0, 0, 0, 0, 0,      Saging[0],-Saging[1],0, 0,0,0, 0,0,0],
            [0,0,0,0, 0,0,0,0, -S3Infs[0],-S3Infs[1],-S3Infs[2],-S3Infs[3], 0,0,0,0, 0,0,0,0, 0, 0, 0, 0,      0,Saging[1],-Saging[2], 0,0,0, 0,0,0],
            [0,0,0,0, 0,0,0,0, 0,0,0,0, -S4Infs[0],-S4Infs[1],-S4Infs[2],-S4Infs[3], 0,0,0,0, 0, -Sdeath, 0, 0, 0,0,Saging[2], 0,0,0, 0,0,0],

            [S1Infs[0],S1Infs[1],S1Infs[2],S1Infs[3], 0,0,0,0, 0,0,0,0, 0,0,0,0, -Irec[0],0,0,0, 0,0,0,0, 0,0,0, -Iaging[0],0,0, 0,0,0],
            [0,0,0,0, S2Infs[0],S2Infs[1],S2Infs[2],S2Infs[3], 0,0,0,0, 0,0,0,0, 0,-Irec[1],0,0, 0,0,0,0, 0,0,0, Iaging[0],-Iaging[1],0, 0,0,0],
            [0,0,0,0, 0,0,0,0, S3Infs[0],S3Infs[1],S3Infs[2],S3Infs[3], 0,0,0,0, 0,0,-Irec[2],0, 0,0,0,0, 0,0,0, 0,Iaging[1],-Iaging[2], 0,0,0],
            [0,0,0,0, 0,0,0,0, 0,0,0,0, S4Infs[0],S4Infs[1],S4Infs[2],S4Infs[3], 0,0,0,-Irec[3], 0,0,-Ideath,0, 0,0,0, 0,0,Iaging[2], 0,0,0],

            [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, Irec[0],0,0,0, 0,0,0,0, 0,0,0, 0,0,0, -Raging[0],0,0],
            [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,Irec[1],0,0, 0,0,0,0, 0,0,0, 0,0,0, Raging[0],-Raging[1],0],
            [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,Irec[2],0, 0,0,0,0, 0,0,0, 0,0,0, 0,Raging[1],-Raging[2]],
            [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,Irec[3], 0,0,0,-Rdeath, 0,0,0, 0,0,0, 0,0,Raging[2]]
        ])
        dS[0] += np.dot(G[0],dW)
        dS[1] += np.dot(G[1],dW)
        dS[2] += np.dot(G[2],dW)
        dS[3] += np.dot(G[3],dW)
        dI[0] += np.dot(G[4],dW)
        dI[1] += np.dot(G[5],dW)
        dI[2] += np.dot(G[6],dW)
        dI[3] += np.dot(G[7],dW)
        dR[0] += np.dot(G[8],dW)
        dR[1] += np.dot(G[9],dW)
        dR[2] += np.dot(G[10],dW)
        dR[3] += np.dot(G[11],dW)
        # Update values
        y_out[t,:4] = np.maximum(0, S + dS)
        y_out[t,4:8] = np.maximum(0, I + dI)
        y_out[t,8:] = np.maximum(0, R + dR)
    return time, y_out

def ito_seir_4(betat, gamma, mu, delta, sigma, y0, T, dt):
    # Number of time steps
    num_steps = int(T / dt)
    time = np.linspace(0, T, num_steps)
    # Initialize arrays to store results
    y_out = np.zeros((num_steps,len(y0)))
    y_out[0,:] = y0

    # Euler-Maruyama method to solve the SDE
    for t in range(1, num_steps):
        S = y_out[t-1,:4]
        E = y_out[t-1,4:8]
        I = y_out[t-1,8:12]
        R = y_out[t-1,12:]
        N = S + I + R
        beta = betat(t)
        delta_prev = np.array([delta[1],delta[2],delta[3],0])
        s_temp = np.array([0,S[0],S[1],S[2]])
        e_temp = np.array([0,E[0],E[1],E[2]])
        i_temp = np.array([0,I[0],I[1],I[2]])
        r_temp = np.array([0,R[0],R[1],R[2]])
        dS = -np.matmul(I/N,beta)*S - mu*S + (mu*N)[::-1] + delta*s_temp - delta_prev*S 
        dE = np.matmul(I/N,beta)*S - sigma*E - mu*E + delta*e_temp - delta_prev*E
        dI =  sigma*E - gamma*I - mu*I + delta*i_temp - delta_prev*I
        dR = gamma*I - mu*R + delta*r_temp - delta_prev*R

        # Compute the diffusion
        dW = np.random.normal(0, np.sqrt(dt), 41)
        # 41 Wiener processes: S1*4I, S2*4I, S3*4I, S4*4I, 4E onset, 4I recovery, 1 birth, 1S death, 1E death, 1I death, 1R death, 3S aging, 3E aging, 3I aging, 3R aging
        S1Infs = [np.sqrt(beta[0,0]*S[0]*I[0]/N[0]),np.sqrt(beta[0,1]*S[0]*I[1]/N[1]),np.sqrt(beta[0,2]*S[0]*I[2]/N[2]),np.sqrt(beta[0,3]*S[0]*I[3]/N[3])]
        S2Infs = [np.sqrt(beta[1,0]*S[1]*I[0]/N[0]),np.sqrt(beta[1,1]*S[1]*I[1]/N[1]),np.sqrt(beta[1,2]*S[1]*I[2]/N[2]),np.sqrt(beta[1,3]*S[1]*I[3]/N[3])]
        S3Infs = [np.sqrt(beta[2,0]*S[2]*I[0]/N[0]),np.sqrt(beta[2,1]*S[2]*I[1]/N[1]),np.sqrt(beta[2,2]*S[2]*I[2]/N[2]),np.sqrt(beta[2,3]*S[2]*I[3]/N[3])]
        S4Infs = [np.sqrt(beta[3,0]*S[3]*I[0]/N[0]),np.sqrt(beta[3,1]*S[3]*I[1]/N[1]),np.sqrt(beta[3,2]*S[3]*I[2]/N[2]),np.sqrt(beta[3,3]*S[3]*I[3]/N[3])]
        Irec = np.sqrt(gamma*I)
        birth = np.sqrt((mu*N)[::-1][0])
        Eonset = np.sqrt(sigma*E)
        Sdeath = np.sqrt(mu*S)[-1]
        Edeath = np.sqrt(mu*E)[-1]
        Ideath = np.sqrt(mu*I)[-1]
        Rdeath = np.sqrt(mu*R)[-1]
        Saging = np.sqrt(delta_prev*S)
        Eaging = np.sqrt(delta_prev*E)
        Iaging = np.sqrt(delta_prev*I)
        Raging = np.sqrt(delta_prev*r_temp)
        G = np.array([
            [-S1Infs[0],-S1Infs[1],-S1Infs[2],-S1Infs[3], 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, birth,0,0,0,0,  -Saging[0],0,0, 0,0,0, 0,0,0, 0,0,0],
            [0,0,0,0, -S2Infs[0],-S2Infs[1],-S2Infs[2],-S2Infs[3], 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0,0,      Saging[0],-Saging[1],0, 0,0,0, 0,0,0, 0,0,0],
            [0,0,0,0, 0,0,0,0, -S3Infs[0],-S3Infs[1],-S3Infs[2],-S3Infs[3], 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0,0,      0,Saging[1],-Saging[2], 0,0,0, 0,0,0, 0,0,0],
            [0,0,0,0, 0,0,0,0, 0,0,0,0, -S4Infs[0],-S4Infs[1],-S4Infs[2],-S4Infs[3], 0,0,0,0, 0,0,0,0, 0,-Sdeath,0,0,0, 0,0,Saging[2], 0,0,0, 0,0,0, 0,0,0],

            [S1Infs[0],S1Infs[1],S1Infs[2],S1Infs[3], 0,0,0,0, 0,0,0,0, 0,0,0,0, -Eonset[0],0,0,0, 0,0,0,0, 0,0,0,0,0, 0,0,0, -Eaging[0],0,0, 0,0,0, 0,0,0],
            [0,0,0,0, S2Infs[0],S2Infs[1],S2Infs[2],S2Infs[3], 0,0,0,0, 0,0,0,0, 0,-Eonset[1],0,0, 0,0,0,0, 0,0,0,0,0, 0,0,0, Eaging[0],-Eaging[1],0, 0,0,0, 0,0,0],
            [0,0,0,0, 0,0,0,0, S3Infs[0],S3Infs[1],S3Infs[2],S3Infs[3], 0,0,0,0, 0,0,-Eonset[2],0, 0,0,0,0, 0,0,0,0,0, 0,0,0, 0,Eaging[1],-Eaging[2], 0,0,0, 0,0,0],
            [0,0,0,0, 0,0,0,0, 0,0,0,0, S4Infs[0],S4Infs[1],S4Infs[2],S4Infs[3], 0,0,0,-Eonset[3], 0,0,0,0, 0,0,-Edeath,0,0, 0,0,0, 0,0,Eaging[2], 0,0,0, 0,0,0],

            [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, Eonset[0],0,0,0, -Irec[0],0,0,0, 0,0,0,0,0, 0,0,0, 0,0,0, -Iaging[0],0,0, 0,0,0],
            [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,Eonset[1],0,0, 0,-Irec[1],0,0, 0,0,0,0,0, 0,0,0, 0,0,0, Iaging[0],-Iaging[1],0, 0,0,0],
            [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,Eonset[2],0, 0,0,-Irec[2],0, 0,0,0,0,0, 0,0,0, 0,0,0, 0,Iaging[1],-Iaging[2], 0,0,0],
            [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,Eonset[3], 0,0,0,-Irec[3], 0,0,0,-Ideath,0, 0,0,0, 0,0,0, 0,0,Iaging[2], 0,0,0],

            [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, Irec[0],0,0,0, 0,0,0,0,0, 0,0,0, 0,0,0, 0,0,0, -Raging[0],0,0],
            [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,Irec[1],0,0, 0,0,0,0,0, 0,0,0, 0,0,0, 0,0,0, Raging[0],-Raging[1],0],
            [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,Irec[2],0, 0,0,0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,Raging[1],-Raging[2]],
            [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,Irec[3], 0,0,0,0,-Rdeath, 0,0,0, 0,0,0, 0,0,0, 0,0,Raging[2]]
        ])
        dS[0] += np.dot(G[0],dW)
        dS[1] += np.dot(G[1],dW)
        dS[2] += np.dot(G[2],dW)
        dS[3] += np.dot(G[3],dW)
        dE[0] += np.dot(G[4],dW)
        dE[1] += np.dot(G[5],dW)
        dE[2] += np.dot(G[6],dW)
        dE[3] += np.dot(G[7],dW)
        dI[0] += np.dot(G[8],dW)
        dI[1] += np.dot(G[9],dW)
        dI[2] += np.dot(G[10],dW)
        dI[3] += np.dot(G[11],dW)
        dR[0] += np.dot(G[12],dW)
        dR[1] += np.dot(G[13],dW)
        dR[2] += np.dot(G[14],dW)
        dR[3] += np.dot(G[15],dW)
        # Update values
        y_out[t,:4] = np.maximum(0, S + dS)
        y_out[t,4:8] = np.maximum(0, E + dE)
        y_out[t,8:12] = np.maximum(0, I + dI)
        y_out[t,12:] = np.maximum(0, R + dR)
    return time, y_out

def symmetric_matrix(theta):
    # Determine the size of the symmetric matrix
    n = int((np.sqrt(1 + 8 * len(theta)) - 1) / 2)  # Solve n * (n + 1) / 2 = len(theta)
    if len(theta) != n * (n + 1) // 2:
        raise ValueError("Size of theta is invalid for a symmetric matrix")
    # Create an empty matrix
    matrix = np.zeros((n, n))
    # Get the indices for the upper triangular part (including the diagonal)
    triu_indices = np.triu_indices(n)
    # Assign values to the upper triangular part
    matrix[triu_indices] = theta
    # Mirror the upper triangular part to the lower triangular part
    matrix = matrix + matrix.T - np.diag(np.diag(matrix))
    return np.matrix(matrix)

def extract_theta(symmetric_matrix):
    # Ensure input is a square matrix
    if symmetric_matrix.shape[0] != symmetric_matrix.shape[1]:
        raise ValueError("Input must be a square matrix")
    # Extract the upper triangular part (including the diagonal)
    triu_indices = np.triu_indices(symmetric_matrix.shape[0])
    theta = symmetric_matrix[triu_indices]
    return np.array(theta).flatten()

def mu_matrix(n,inds,mu):
    temp = np.zeros((n,n))
    temp[inds] = mu
    return temp

def lambda_I_gen(I,n,beta):
    lambda_I = np.zeros((len(I),len(I)))
    for i in range(len(I)):
        for j in range(len(I)):
            if i == j:
                lambda_I[i,j] = -beta[i,j]*(n[i]-I[i])/(n[i]**2) - np.sum(np.delete(beta[i],i)*np.delete(I,i)/np.delete(n,i))
            else:
                lambda_I[i,j] = beta[i,j]*I[j]/(n[j]**2)
    return lambda_I

def lambda_S_gen(S,I,n,beta):
    lambda_S = np.zeros((len(S),len(S)))
    for i in range(len(S)):
        for j in range(len(S)):
            lambda_S[i,j] = beta[i,j]*S[i]*(n[j]-I[j])/(n[j]**2)
    return lambda_S

def sis_4_int(X, t, betat, gamma, mu, delta):
    S = X[:4]
    I = X[4:8]
    Theta = X[8:]
    N = np.sum(S + I)
    n = (S + I) / N
    S = S / N
    I = I/ N
    beta = betat(t)
    delta_mat = np.array([[-delta[0],0,0,0],[delta[0],-delta[1],0,0],[0,delta[1],-delta[2],0],[0,0,delta[2],0]])
    lambda_I = lambda_I_gen(I,n,beta)
    lambda_S = lambda_S_gen(S,I,n,beta)
    J_ss = lambda_I + delta_mat + mu_matrix(4,(0,3),mu[-1]) - mu_matrix(4,(3,3),mu[-1])
    J_si = -lambda_S + gamma*np.eye(4) + mu_matrix(4,(0,3),mu[-1])
    J_is = -lambda_I
    J_ii = lambda_S + delta_mat - gamma*np.eye(4) - mu_matrix(4,(3,3),mu[-1])
    A = np.block([[J_ss,J_si],[J_is,J_ii]])
    delta_prev = np.array([delta[1],delta[2],delta[3],0])
    # Compute the diffusion
    # 29 Wiener processes: S1*4I, S2*4I, S3*4I, S4*4I, 4I recovery, 1 birth, 1S death, 1I death, 3S aging, 3I aging
    S1Infs = [np.sqrt(beta[0,0]*S[0]*I[0]/n[0]),np.sqrt(beta[0,1]*S[0]*I[1]/n[1]),np.sqrt(beta[0,2]*S[0]*I[2]/n[2]),np.sqrt(beta[0,3]*S[0]*I[3]/n[3])]
    S2Infs = [np.sqrt(beta[1,0]*S[1]*I[0]/n[0]),np.sqrt(beta[1,1]*S[1]*I[1]/n[1]),np.sqrt(beta[1,2]*S[1]*I[2]/n[2]),np.sqrt(beta[1,3]*S[1]*I[3]/n[3])]
    S3Infs = [np.sqrt(beta[2,0]*S[2]*I[0]/n[0]),np.sqrt(beta[2,1]*S[2]*I[1]/n[1]),np.sqrt(beta[2,2]*S[2]*I[2]/n[2]),np.sqrt(beta[2,3]*S[2]*I[3]/n[3])]
    S4Infs = [np.sqrt(beta[3,0]*S[3]*I[0]/n[0]),np.sqrt(beta[3,1]*S[3]*I[1]/n[1]),np.sqrt(beta[3,2]*S[3]*I[2]/n[2]),np.sqrt(beta[3,3]*S[3]*I[3]/n[3])]
    Irec = np.sqrt(gamma*I)
    birth = np.sqrt((mu*n)[::-1][0])
    Sdeath = np.sqrt(mu*S)[-1]
    Ideath = np.sqrt(mu*I)[-1]
    Saging = np.sqrt(delta_prev*S)
    Iaging = np.sqrt(delta_prev*I)
    G = np.array([
        [-S1Infs[0],-S1Infs[1],-S1Infs[2],-S1Infs[3], 0,0,0,0, 0,0,0,0, 0,0,0,0, Irec[0],0,0,0, birth, 0, 0, -Saging[0],0,0, 0,0,0],
        [0,0,0,0, -S2Infs[0],-S2Infs[1],-S2Infs[2],-S2Infs[3], 0,0,0,0, 0,0,0,0, 0,Irec[1],0,0, 0, 0, 0, Saging[0],-Saging[1],0, 0,0,0],
        [0,0,0,0, 0,0,0,0, -S3Infs[0],-S3Infs[1],-S3Infs[2],-S3Infs[3], 0,0,0,0, 0,0,Irec[2],0, 0, 0, 0, 0,Saging[1],-Saging[2], 0,0,0],
        [0,0,0,0, 0,0,0,0, 0,0,0,0, -S4Infs[0],-S4Infs[1],-S4Infs[2],-S4Infs[3], 0,0,0,Irec[3], 0, -Sdeath, 0, 0,0,Saging[2], 0,0,0],

        [S1Infs[0],S1Infs[1],S1Infs[2],S1Infs[3], 0,0,0,0, 0,0,0,0, 0,0,0,0, -Irec[0],0,0,0, 0, 0, 0, 0,0,0, -Iaging[0],0,0],
        [0,0,0,0, S2Infs[0],S2Infs[1],S2Infs[2],S2Infs[3], 0,0,0,0, 0,0,0,0, 0,-Irec[1],0,0, 0, 0, 0, 0,0,0, Iaging[0],-Iaging[1],0,],
        [0,0,0,0, 0,0,0,0, S3Infs[0],S3Infs[1],S3Infs[2],S3Infs[3], 0,0,0,0, 0,0,-Irec[2],0, 0, 0, 0, 0,0,0, 0,Iaging[1],-Iaging[2]],
        [0,0,0,0, 0,0,0,0, 0,0,0,0, S4Infs[0],S4Infs[1],S4Infs[2],S4Infs[3], 0,0,0,-Irec[3], 0, 0, -Ideath, 0,0,0, 0,0,Iaging[2],]
    ])
    # Arranged [B_ss,B_si][B_is,B_ii]
    B = G @ G.transpose()
    Θ = symmetric_matrix(Theta)
    K = A*Θ + Θ*A.transpose()+B

    S = X[:4]
    I = X[4:8]
    Theta = X[8:]
    N = S + I
    beta = betat(t)
    s_temp = np.array([0,S[0],S[1],S[2]])
    i_temp = np.array([0,I[0],I[1],I[2]])
    dS = - np.matmul(I/N,beta)*S - mu*S + (mu*N)[::-1] + delta*s_temp - delta_prev*S  + gamma*I
    dI =  np.matmul(I/N,beta)*S - gamma*I - mu*I + delta*i_temp - delta_prev*I

    return np.concatenate([dS, dI, extract_theta(K)])

def lambda_X_gen(beta, S, I, n):
    lambda_X = np.zeros((len(S),len(S)))
    for i in range(len(S)):
        for j in range(len(S)):
            lambda_X[i,j] = beta[i,j]*S[i]*I[j]/(n[j]**2)
    return lambda_X

def sir_4_int(X, t, betat, gamma, mu, delta):
    S = X[:4]
    I = X[4:8]
    R = X[8:12]
    Theta = X[12:]
    N = S + I + R
    beta = betat(t)

    delta_prev = np.array([delta[1],delta[2],delta[3],0])
    s_temp = np.array([0,S[0],S[1],S[2]])
    i_temp = np.array([0,I[0],I[1],I[2]])
    r_temp = np.array([0,R[0],R[1],R[2]])
    dS = -np.matmul(I/N,beta)*S - mu*S + (mu*N)[::-1] + delta*s_temp - delta_prev*S 
    dI = np.matmul(I/N,beta)*S - gamma*I - mu*I + delta*i_temp - delta_prev*I
    dR = gamma*I - mu*R + delta*r_temp - delta_prev*R
    N = np.sum(N)
    n = (S + I + R) / N
    S = S / N
    I = I / N  
    R = R / N 

    delta_mat = np.array([[-delta[0],0,0,0],[delta[0],-delta[1],0,0],[0,delta[1],-delta[2],0],[0,0,delta[2],0]])
    lambda_I = lambda_I_gen(I, n, beta)
    lambda_S = lambda_S_gen(S, I, n, beta)
    lambda_X = lambda_X_gen(beta, S, I, n)

    J_ss = lambda_I + delta_mat + mu_matrix(4,(0,3),mu[-1]) - mu_matrix(4,(3,3),mu[-1])
    J_si = -lambda_S + mu_matrix(4,(0,3),mu[-1])
    J_is = -lambda_I
    J_ii = lambda_S + delta_mat - gamma*np.eye(4) - mu_matrix(4,(3,3),mu[-1])
    J_sr = lambda_X + mu_matrix(4,(0,3),mu[-1])
    J_rs = np.zeros((4,4))
    J_ir = -lambda_X
    J_ri = gamma*np.eye(4)
    J_rr = delta_mat - mu_matrix(4,(3,3),mu[-1])
    A = np.block([[J_ss,J_si,J_sr],[J_is,J_ii,J_ir],[J_rs,J_ri,J_rr]])

    # 33 Wiener processes: S1*4I, S2*4I, S3*4I, S4*4I, 4I recovery, 1 birth, 1S death, 1I death, 1R death, 3S aging, 3I aging, 3R aging
    S1Infs = [np.sqrt(beta[0,0]*S[0]*I[0]/n[0]),np.sqrt(beta[0,1]*S[0]*I[1]/n[1]),np.sqrt(beta[0,2]*S[0]*I[2]/n[2]),np.sqrt(beta[0,3]*S[0]*I[3]/n[3])]
    S2Infs = [np.sqrt(beta[1,0]*S[1]*I[0]/n[0]),np.sqrt(beta[1,1]*S[1]*I[1]/n[1]),np.sqrt(beta[1,2]*S[1]*I[2]/n[2]),np.sqrt(beta[1,3]*S[1]*I[3]/n[3])]
    S3Infs = [np.sqrt(beta[2,0]*S[2]*I[0]/n[0]),np.sqrt(beta[2,1]*S[2]*I[1]/n[1]),np.sqrt(beta[2,2]*S[2]*I[2]/n[2]),np.sqrt(beta[2,3]*S[2]*I[3]/n[3])]
    S4Infs = [np.sqrt(beta[3,0]*S[3]*I[0]/n[0]),np.sqrt(beta[3,1]*S[3]*I[1]/n[1]),np.sqrt(beta[3,2]*S[3]*I[2]/n[2]),np.sqrt(beta[3,3]*S[3]*I[3]/n[3])]
    Irec = np.sqrt(gamma*I)
    birth = np.sqrt((mu*n)[::-1][0])
    Sdeath = np.sqrt(mu*S)[-1]
    Ideath = np.sqrt(mu*I)[-1]
    Rdeath = np.sqrt(mu*R)[-1]
    Saging = np.sqrt(delta_prev*S)
    Iaging = np.sqrt(delta_prev*I)
    Raging = np.sqrt(delta_prev*R)
    G = np.array([
        [-S1Infs[0],-S1Infs[1],-S1Infs[2],-S1Infs[3], 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, birth, 0, 0, 0,  -Saging[0],0,0, 0,0,0, 0,0,0],
        [0,0,0,0, -S2Infs[0],-S2Infs[1],-S2Infs[2],-S2Infs[3], 0,0,0,0, 0,0,0,0, 0,0,0,0, 0, 0, 0, 0,      Saging[0],-Saging[1],0, 0,0,0, 0,0,0],
        [0,0,0,0, 0,0,0,0, -S3Infs[0],-S3Infs[1],-S3Infs[2],-S3Infs[3], 0,0,0,0, 0,0,0,0, 0, 0, 0, 0,      0,Saging[1],-Saging[2], 0,0,0, 0,0,0],
        [0,0,0,0, 0,0,0,0, 0,0,0,0, -S4Infs[0],-S4Infs[1],-S4Infs[2],-S4Infs[3], 0,0,0,0, 0, -Sdeath, 0, 0, 0,0,Saging[2], 0,0,0, 0,0,0],

        [S1Infs[0],S1Infs[1],S1Infs[2],S1Infs[3], 0,0,0,0, 0,0,0,0, 0,0,0,0, -Irec[0],0,0,0, 0,0,0,0, 0,0,0, -Iaging[0],0,0, 0,0,0],
        [0,0,0,0, S2Infs[0],S2Infs[1],S2Infs[2],S2Infs[3], 0,0,0,0, 0,0,0,0, 0,-Irec[1],0,0, 0,0,0,0, 0,0,0, Iaging[0],-Iaging[1],0, 0,0,0],
        [0,0,0,0, 0,0,0,0, S3Infs[0],S3Infs[1],S3Infs[2],S3Infs[3], 0,0,0,0, 0,0,-Irec[2],0, 0,0,0,0, 0,0,0, 0,Iaging[1],-Iaging[2], 0,0,0],
        [0,0,0,0, 0,0,0,0, 0,0,0,0, S4Infs[0],S4Infs[1],S4Infs[2],S4Infs[3], 0,0,0,-Irec[3], 0,0,-Ideath,0, 0,0,0, 0,0,Iaging[2], 0,0,0],

        [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, Irec[0],0,0,0, 0,0,0,0, 0,0,0, 0,0,0, -Raging[0],0,0],
        [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,Irec[1],0,0, 0,0,0,0, 0,0,0, 0,0,0, Raging[0],-Raging[1],0],
        [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,Irec[2],0, 0,0,0,0, 0,0,0, 0,0,0, 0,Raging[1],-Raging[2]],
        [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,Irec[3], 0,0,0,-Rdeath, 0,0,0, 0,0,0, 0,0,Raging[2]]
    ])

    B = G @ G.transpose()
    Θ = symmetric_matrix(Theta)
    K = A*Θ +Θ*A.transpose()+B
    return np.concatenate([dS,dI,dR,extract_theta(K)])

def seir_4_int(X, t, betat, gamma, mu, delta, sigma):
    S = X[:4]
    E = X[4:8]
    I = X[8:12]
    R = X[12:16]
    Theta = X[16:]
    N = S + E + I + R
    beta = betat(t)

    delta_prev = np.array([delta[1],delta[2],delta[3],0])
    s_temp = np.array([0,S[0],S[1],S[2]])
    e_temp = np.array([0,E[0],E[1],E[2]])
    i_temp = np.array([0,I[0],I[1],I[2]])
    r_temp = np.array([0,R[0],R[1],R[2]])
    dS = -np.matmul(I/N,beta)*S - mu*S + (mu*N)[::-1] + delta*s_temp - delta_prev*S 
    dE = np.matmul(I/N,beta)*S - sigma*E - mu*E + delta*e_temp - delta_prev*E
    dI =  sigma*E - gamma*I - mu*I + delta*i_temp - delta_prev*I
    dR = gamma*I - mu*R + delta*r_temp - delta_prev*R

    N = np.sum(N)
    n = (S + E + I + R) / N
    S = S / N
    E = E / N
    I = I / N  
    R = R / N 

    delta_mat = np.array([[-delta[0],0,0,0],[delta[0],-delta[1],0,0],[0,delta[1],-delta[2],0],[0,0,delta[2],0]])
    lambda_I = lambda_I_gen(I, n, beta)
    lambda_S = lambda_S_gen(S, I, n, beta)
    lambda_X = lambda_X_gen(beta, S, I, n)

    J_ss = lambda_I + delta_mat + mu_matrix(4,(0,3),mu[-1]) - mu_matrix(4,(3,3),mu[-1])
    J_si = -lambda_S + mu_matrix(4,(0,3),mu[-1])
    J_ii = lambda_S + delta_mat - gamma*np.eye(4) - mu_matrix(4,(3,3),mu[-1])
    J_sr = lambda_X + mu_matrix(4,(0,3),Sdeath)
    J_rs = np.zeros((4,4))
    J_ir = -lambda_X
    J_ri = gamma*np.eye(4)
    J_rr = delta_mat - mu_matrix(4,(3,3),mu[-1])

    J_is = np.zeros((4,4))
    J_se = lambda_X + mu_matrix(4,(0,3),mu[-1])
    J_es = -lambda_I
    J_ee = -lambda_X + delta_mat - sigma*np.eye(4) - mu_matrix(4,(3,3),mu[-1])
    J_ei = -lambda_X
    J_er = -lambda_X
    J_ie = sigma*np.eye(4)
    J_re = np.zeros((4,4))
    A = np.block([[J_ss,J_se,J_si,J_sr],[J_es,J_ee,J_ei,J_er],[J_is,J_ie,J_ii,J_ir],[J_rs,J_re,J_ri,J_rr]])

    # 41 Wiener processes: S1*4I, S2*4I, S3*4I, S4*4I, 4E onset, 4I recovery, 1 birth, 1S death, 1E death, 1I death, 1R death, 3S aging, 3E aging, 3I aging, 3R aging
    S1Infs = [np.sqrt(beta[0,0]*S[0]*I[0]/n[0]),np.sqrt(beta[0,1]*S[0]*I[1]/n[1]),np.sqrt(beta[0,2]*S[0]*I[2]/n[2]),np.sqrt(beta[0,3]*S[0]*I[3]/n[3])]
    S2Infs = [np.sqrt(beta[1,0]*S[1]*I[0]/n[0]),np.sqrt(beta[1,1]*S[1]*I[1]/n[1]),np.sqrt(beta[1,2]*S[1]*I[2]/n[2]),np.sqrt(beta[1,3]*S[1]*I[3]/n[3])]
    S3Infs = [np.sqrt(beta[2,0]*S[2]*I[0]/n[0]),np.sqrt(beta[2,1]*S[2]*I[1]/n[1]),np.sqrt(beta[2,2]*S[2]*I[2]/n[2]),np.sqrt(beta[2,3]*S[2]*I[3]/n[3])]
    S4Infs = [np.sqrt(beta[3,0]*S[3]*I[0]/n[0]),np.sqrt(beta[3,1]*S[3]*I[1]/n[1]),np.sqrt(beta[3,2]*S[3]*I[2]/n[2]),np.sqrt(beta[3,3]*S[3]*I[3]/n[3])]
    Irec = np.sqrt(gamma*I)
    birth = np.sqrt((mu*n)[::-1][0])
    Eonset = np.sqrt(sigma*E)
    Sdeath = np.sqrt(mu*S)[-1]
    Edeath = np.sqrt(mu*E)[-1]
    Ideath = np.sqrt(mu*I)[-1]
    Rdeath = np.sqrt(mu*R)[-1]
    Saging = np.sqrt(delta_prev*S)
    Eaging = np.sqrt(delta_prev*E)
    Iaging = np.sqrt(delta_prev*I)
    Raging = np.sqrt(delta_prev*R)
    G = np.array([
        [-S1Infs[0],-S1Infs[1],-S1Infs[2],-S1Infs[3], 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, birth,0,0,0,0,  -Saging[0],0,0, 0,0,0, 0,0,0, 0,0,0],
        [0,0,0,0, -S2Infs[0],-S2Infs[1],-S2Infs[2],-S2Infs[3], 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0,0,      Saging[0],-Saging[1],0, 0,0,0, 0,0,0, 0,0,0],
        [0,0,0,0, 0,0,0,0, -S3Infs[0],-S3Infs[1],-S3Infs[2],-S3Infs[3], 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0,0,      0,Saging[1],-Saging[2], 0,0,0, 0,0,0, 0,0,0],
        [0,0,0,0, 0,0,0,0, 0,0,0,0, -S4Infs[0],-S4Infs[1],-S4Infs[2],-S4Infs[3], 0,0,0,0, 0,0,0,0, 0,-Sdeath,0,0,0, 0,0,Saging[2], 0,0,0, 0,0,0, 0,0,0],

        [S1Infs[0],S1Infs[1],S1Infs[2],S1Infs[3], 0,0,0,0, 0,0,0,0, 0,0,0,0, -Eonset[0],0,0,0, 0,0,0,0, 0,0,0,0,0, 0,0,0, -Eaging[0],0,0, 0,0,0, 0,0,0],
        [0,0,0,0, S2Infs[0],S2Infs[1],S2Infs[2],S2Infs[3], 0,0,0,0, 0,0,0,0, 0,-Eonset[1],0,0, 0,0,0,0, 0,0,0,0,0, 0,0,0, Eaging[0],-Eaging[1],0, 0,0,0, 0,0,0],
        [0,0,0,0, 0,0,0,0, S3Infs[0],S3Infs[1],S3Infs[2],S3Infs[3], 0,0,0,0, 0,0,-Eonset[2],0, 0,0,0,0, 0,0,0,0,0, 0,0,0, 0,Eaging[1],-Eaging[2], 0,0,0, 0,0,0],
        [0,0,0,0, 0,0,0,0, 0,0,0,0, S4Infs[0],S4Infs[1],S4Infs[2],S4Infs[3], 0,0,0,-Eonset[3], 0,0,0,0, 0,0,-Edeath,0,0, 0,0,0, 0,0,Eaging[2], 0,0,0, 0,0,0],

        [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, Eonset[0],0,0,0, -Irec[0],0,0,0, 0,0,0,0,0, 0,0,0, 0,0,0, -Iaging[0],0,0, 0,0,0],
        [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,Eonset[1],0,0, 0,-Irec[1],0,0, 0,0,0,0,0, 0,0,0, 0,0,0, Iaging[0],-Iaging[1],0, 0,0,0],
        [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,Eonset[2],0, 0,0,-Irec[2],0, 0,0,0,0,0, 0,0,0, 0,0,0, 0,Iaging[1],-Iaging[2], 0,0,0],
        [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,Eonset[3], 0,0,0,-Irec[3], 0,0,0,-Ideath,0, 0,0,0, 0,0,0, 0,0,Iaging[2], 0,0,0],

        [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, Irec[0],0,0,0, 0,0,0,0,0, 0,0,0, 0,0,0, 0,0,0, -Raging[0],0,0],
        [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,Irec[1],0,0, 0,0,0,0,0, 0,0,0, 0,0,0, 0,0,0, Raging[0],-Raging[1],0],
        [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,Irec[2],0, 0,0,0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,Raging[1],-Raging[2]],
        [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,Irec[3], 0,0,0,0,-Rdeath, 0,0,0, 0,0,0, 0,0,0, 0,0,Raging[2]]
    ])

    B = G @ G.transpose()
    Θ = symmetric_matrix(Theta)
    K = A*Θ +Θ*A.transpose()+B
    return np.concatenate([dS,dE,dI,dR,extract_theta(K)])

def mu_matrix_na(n,inds,mu):
    temp = np.zeros((n,n))
    temp[inds] = mu
    return temp

def lambda_I_gen_na(I,beta):
    lambda_I = np.diag(-np.matmul(I,beta))
    return lambda_I

def lambda_S_gen_na(S,beta):
    lambda_S = np.zeros((len(S),len(S)))
    for i in range(len(S)):
        for j in range(len(S)):
            lambda_S[i,j] = beta[i,j]*S[i]
    return lambda_S

def mu_matrix_na(n,N,inds,mu):
    temp = np.zeros((n,n))
    temp[inds] = mu*N[inds[1]]/N[inds[0]]
    return temp

def sis_4_int_na(X, t, betat, gamma, mu, delta):
    S = X[:4]
    I = X[4:8]
    Theta = X[8:]
    N = S + I
    S = S / N
    I = I / N
    beta = betat(t)
    delta_mat = np.array([[-delta[0],0,0,0],[delta[0]*N[0]/N[1],-delta[1],0,0],[0,delta[1]*N[1]/N[2],-delta[2],0],[0,0,delta[2]*N[2]/N[3],0]])
    lambda_I = lambda_I_gen_na(I,beta)
    lambda_S = lambda_S_gen_na(S,beta)
    J_ss = lambda_I + delta_mat + mu_matrix_na(4,N,(0,3),mu[-1]) - mu_matrix(4,(3,3),mu[-1])
    J_si = -lambda_S + gamma*np.eye(4) + mu_matrix_na(4,N,(0,3),mu[-1])
    J_is = -lambda_I
    J_ii = lambda_S + delta_mat - gamma*np.eye(4) - mu_matrix(4,(3,3),mu[-1])
    for i in range(len(J_ss)):
        for j in range(len(J_ss)):
            J_ss[i,j] = J_ss[i,j]*np.sqrt(N[i]/N[j])
            J_si[i,j] = J_si[i,j]*np.sqrt(N[i]/N[j])
            J_is[i,j] = J_is[i,j]*np.sqrt(N[i]/N[j])
            J_ii[i,j] = J_ii[i,j]*np.sqrt(N[i]/N[j])
    A = np.block([[J_ss,J_si],[J_is,J_ii]])


    delta_prev = np.array([delta[1],delta[2],delta[3],0])
    # Compute the diffusion
    # 29 Wiener processes: S1*4I, S2*4I, S3*4I, S4*4I, 4I recovery, 1 birth, 1S death, 1I death, 3S aging, 3I aging
    S1Infs = [np.sqrt(beta[0,0]*S[0]*I[0]),np.sqrt(beta[0,1]*S[0]*I[1]),np.sqrt(beta[0,2]*S[0]*I[2]),np.sqrt(beta[0,3]*S[0]*I[3])]
    S2Infs = [np.sqrt(beta[1,0]*S[1]*I[0]),np.sqrt(beta[1,1]*S[1]*I[1]),np.sqrt(beta[1,2]*S[1]*I[2]),np.sqrt(beta[1,3]*S[1]*I[3])]
    S3Infs = [np.sqrt(beta[2,0]*S[2]*I[0]),np.sqrt(beta[2,1]*S[2]*I[1]),np.sqrt(beta[2,2]*S[2]*I[2]),np.sqrt(beta[2,3]*S[2]*I[3])]
    S4Infs = [np.sqrt(beta[3,0]*S[3]*I[0]),np.sqrt(beta[3,1]*S[3]*I[1]),np.sqrt(beta[3,2]*S[3]*I[2]),np.sqrt(beta[3,3]*S[3]*I[3])]
    Irec = np.sqrt(gamma*I)
    birth = np.sqrt((mu*N)[::-1][0])
    Sdeath = np.sqrt(mu*S)[-1]
    Ideath = np.sqrt(mu*I)[-1]
    Saging = np.sqrt(delta_prev*S)
    Iaging = np.sqrt(delta_prev*I)
    G = np.array([
        [-S1Infs[0],-S1Infs[1],-S1Infs[2],-S1Infs[3], 0,0,0,0, 0,0,0,0, 0,0,0,0, Irec[0],0,0,0, birth, 0, 0, -Saging[0],0,0, 0,0,0],
        [0,0,0,0, -S2Infs[0],-S2Infs[1],-S2Infs[2],-S2Infs[3], 0,0,0,0, 0,0,0,0, 0,Irec[1],0,0, 0, 0, 0, Saging[0],-Saging[1],0, 0,0,0],
        [0,0,0,0, 0,0,0,0, -S3Infs[0],-S3Infs[1],-S3Infs[2],-S3Infs[3], 0,0,0,0, 0,0,Irec[2],0, 0, 0, 0, 0,Saging[1],-Saging[2], 0,0,0],
        [0,0,0,0, 0,0,0,0, 0,0,0,0, -S4Infs[0],-S4Infs[1],-S4Infs[2],-S4Infs[3], 0,0,0,Irec[3], 0, -Sdeath, 0, 0,0,Saging[2], 0,0,0],

        [S1Infs[0],S1Infs[1],S1Infs[2],S1Infs[3], 0,0,0,0, 0,0,0,0, 0,0,0,0, -Irec[0],0,0,0, 0, 0, 0, 0,0,0, -Iaging[0],0,0],
        [0,0,0,0, S2Infs[0],S2Infs[1],S2Infs[2],S2Infs[3], 0,0,0,0, 0,0,0,0, 0,-Irec[1],0,0, 0, 0, 0, 0,0,0, Iaging[0],-Iaging[1],0,],
        [0,0,0,0, 0,0,0,0, S3Infs[0],S3Infs[1],S3Infs[2],S3Infs[3], 0,0,0,0, 0,0,-Irec[2],0, 0, 0, 0, 0,0,0, 0,Iaging[1],-Iaging[2]],
        [0,0,0,0, 0,0,0,0, 0,0,0,0, S4Infs[0],S4Infs[1],S4Infs[2],S4Infs[3], 0,0,0,-Irec[3], 0, 0, -Ideath, 0,0,0, 0,0,Iaging[2],]
    ])
    # Arranged [B_ss,B_si][B_is,B_ii]
    B = G @ G.transpose()
    Θ = symmetric_matrix(Theta)
    K = A*Θ + Θ*A.transpose()+B

    S = X[:4]
    I = X[4:8]
    Theta = X[8:]
    N = S + I
    beta = betat(t)
    s_temp = np.array([0,S[0],S[1],S[2]])
    i_temp = np.array([0,I[0],I[1],I[2]])
    dS = - np.matmul(I/N,beta)*S - mu*S + (mu*N)[::-1] + delta*s_temp - delta_prev*S  + gamma*I
    dI =  np.matmul(I/N,beta)*S - gamma*I - mu*I + delta*i_temp - delta_prev*I

    return np.concatenate([dS, dI, extract_theta(K)])