#! /usr/bin/env python
# -*- coding: utf-8 -*-

from numpy import dot, sum, tile, linalg
from numpy.linalg import inv 
import numpy as np


def kf_predict(X, P, A, Q, B, U):
    """  predict the mean X and the covariance P of the system state at the
    time step k

    Model : 
    x(k) = [ Ax(k-1 ) ]+ [Bu(k)] + [w(k-1)] 
    y(k) = [ Hx(k) ] + [ v(k) ]

    p(w) = probability distribution of w = Normal(0,Q)
    p(v) = probability distribution of v = Normal(0,R)

    Args : 
        X : The (predicted) mean state estimate at previous step (k −1) before seeing measurement (k-1).
        P : The (predicted) covariance at previous step (k −1) before seeing measurement (k-1).
        A : The transition [NxN] matrix.
        Q : The process noise covariance matrix.
        B : The input effect matrix.
        U : The control input. 

    """
    X = dot(A, X) + dot(B, U)
    P = dot(A, dot(P, A.T)) + Q
    return(X,P) 


def kf_update(X, P, Y, H, R):
    """update step computes the posterior mean X and covariance
    P of the system state given a new measurement Y

    Model : 
    X(k:predicted) = [ A(k-1)X(k-1:estimated) ]+ [B(k)U(k)]
    P(k:predicted) = [ A(k-1)P(k-1:estimated)A(k-1:transpose) + Q(k-1) ]

    Update : 
    IM(k:estimated) = Y(k:estimated) - H(k)X(k:predicted)
    IS(k:estimated) = H(k)P(k:predicted)H(k:transpose) + R(k)
    K(k) = P(k:predicted)H(k:transpose)S(k-1:inverse)
    X(k) = X(k:predicted) + K(k)V(k)
    P(k) = P(k:estimated) - K(k)S(k)K(k:tranpose)

    Args : 
    X : predicted state
    P : covariance of state vector(at previous time step)
    Y : measurement vector  
    H : measurement matrix  
    R : measurement covariance amtrix
    
    K : the Kalman Gain matrix(filter gain : how much redictions should be corrected at time k)
    IM : measurement residual: the Mean of predictive distribution of Y
    IS : measurement prediction covariance: the Covariance or predictive mean of Y
    LH : the Predictive probability (likelihood) of measurement which is
    computed using the Python function gauss_pdf. 

    """

    IM = dot(H, X)
    IS = R + dot(H, dot(P, H.T))
    K = dot(P, dot(H.T, inv(IS)))
    X = X + dot(K, (Y-IM))
    P = P - dot(K, dot(IS, K.T))
    LH = gauss_pdf(Y, IM, IS)
    return (X,P,K,IM,IS,LH)

def gauss_pdf(X, M, S):
    """
    Predefined function for calculating predictive probability(likelihood) of a measurement
    """
    if M.shape[1] == 1:
        DX = X - tile(M, X.shape[1])
        E = 0.5 * sum(DX * (dot(inv(S), DX)), axis=0)
        E = E + 0.5 * M.shape[0] * np.log(2 * np.pi) + 0.5 * np.log(np.linalg.det(S))
        P = np.exp(-E)
    elif X.shape[1] == 1:
        DX = tile(X, M.shape[1])- M
        E = 0.5 * sum(DX * (dot(inv(S), DX)), axis=0)
        E = E + 0.5 * M.shape[0] * np.log(2 * np.pi) + 0.5 * np.log(np.linalg.det(S))
        P = np.exp(-E)
    else:
        DX = X-M
        E = 0.5 * dot(DX.T, dot(inv(S), DX))
        E = E + 0.5 * M.shape[0] * np.log(2 * np.pi) + 0.5 * np.log(np.linalg.det(S))
        P = np.exp(-E)

    return (P[0],E[0]) 

    






if __name__ == "__main__" :

    #time step of mobile movement
    dt = 0.1

    # Initialization of state matrices
    X = np.array([[0.0], [0.0], [0.1], [0.1]])
    P = np.diag((0.01, 0.01, 0.01, 0.01))
    A = np.array([[1, 0, dt , 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0,1]])
    # import pdb; pdb.set_trace()
    Q = np.eye(X.shape[0])
    B = np.eye(X.shape[0])
    U = np.zeros((X.shape[0],1)) 

    # Measurement matrices
    Y = np.array([[X[0,0] + abs(np.random.randn(1)[0])], [X[1,0] +abs(np.random.randn(1)[0])]])
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
    R = np.eye(Y.shape[0])

    # Number of iterations in Kalman Filter
    N_iter = 50

    # Applying the Kalman Filter
    for i in range(0, N_iter):
        (X, P) = kf_predict(X, P, A, Q, B, U)
        (X, P, K, IM, IS, LH) = kf_update(X, P, Y, H, R)
        Y = np.array([[X[0,0] + abs(0.1 * np.random.randn(1)[0])],[X[1, 0] +\
        abs(0.1 * np.random.randn(1)[0])]])

    

