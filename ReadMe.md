
Basic Kalman filter application

Use this for :
    Determing system state from noisy data 

Assumption :
    *Linear System
    *Noise if uncorrelated and normally distributed
    *Noise covariance Q and measurement noise R covariance are assumed constant 
    
    Optimal if :
        1 . Model perfectly matches real system
        2 . Entering noise is uncorrelated , normally distributed(which if you have enough data should be fine due to law of large numbers)
        3 . Covariances of noise are exactly known


Concepts  :

    Weights : 

        The purpose of the weights is that values with better (i.e., smaller) estimated uncertainty are "trusted" more. Calculated from the covariance, a measure of the estimated uncertainty of the prediction of the system's state

        Kalman gain is the relative weight given to the measurements and current state estimate, and can be "tuned" to achieve particular performance. With a high gain, the filter places more weight on the most recent measurements, and thus follows them more responsively. With a low gain, the filter follows the model predictions more closely. At the extremes, a high gain close to one will result in a more jumpy estimated trajectory, while low gain close to zero will smooth out noise but decrease the responsiveness.


    
    
    
