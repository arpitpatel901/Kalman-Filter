
from numpy import dot, sum, tile, linalg
from numpy.linalg import inv 

def kf_predict(X, P, A, Q, B, U):
    """  predict the mean X and the covariance P of the system state at the
    time step k

    Args : 
        X : The mean state estimate of the previous step ( k −1).
        P : The state covariance of previous step ( k −1).
        A : The transition n n × matrix.
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

    X : predicted state
    P : covariance of previous step
    Y : measurement vector  
    H : measurement matrix  
    R : measurement covariance amtrix
    
    K : the Kalman Gain matrix
    IM : the Mean of predictive distribution of Y
    IS : the Covariance or predictive mean of Y
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
    if M.shape()[1] == 1:
        DX = X - tile(M, X.shape()[1])
        E = 0.5 * sum(DX * (dot(inv(S), DX)), axis=0)
        E = E + 0.5 * M.shape()[0] * log(2 * pi) + 0.5 * log(det(S))
        P = exp(-E)
    elif X.shape()[1] == 1:
        DX = tile(X, M.shape()[1])- M
        E = 0.5 * sum(DX * (dot(inv(S), DX)), axis=0)
        E = E + 0.5 * M.shape()[0] * log(2 * pi) + 0.5 * log(det(S))
        P = exp(-E)
    else:
    DX = X-M
        E = 0.5 * dot(DX.T, dot(inv(S), DX))
        E = E + 0.5 * M.shape()[0] * log(2 * pi) + 0.5 * log(det(S))
        P = exp(-E)
    return (P[0],E[0]) 

