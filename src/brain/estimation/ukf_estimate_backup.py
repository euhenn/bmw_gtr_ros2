import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
#Hard mode NO OTHER IMPORTS ALLOWED


def estInitialize(first_meas_x, first_meas_y, first_meas_yaw):
    x = first_meas_x
    y = first_meas_y
    theta = first_meas_yaw

    Pm = np.diag([0.05**2, 0.05**2, (5*np.pi/180)**2])
    return [x, y, theta, Pm]




def estRun(time, dt, internalStateIn, steeringAngle, speed, measurement):
    # In this function you implement your estimator. The function arguments are:
    #  dt: current time step [s]
    #  internalStateIn: the estimator internal state, definition up to you. 
    #  steeringAngle: the steering angle of the bike, gamma, [rad] 
    #  speed: longitudinal speed[m/s] 
    #  measurement: the position measurement valid at the current time step
    #
    # Note: the measurement is a 2D vector, of x-y position measurement.
    #  The measurement sensor may fail to return data, in which case the
    #  measurement is given as NaN (not a number).
    #
    # The function has four outputs:
    #  est_x: your current best estimate for the bicycle's x-position
    #  est_y: your current best estimate for the bicycle's y-position
    #  est_theta: your current best estimate for the bicycle's rotation theta
    #  internalState: the estimator's internal state, in a format that can be understood by the next call to this function

    # this internal state needs to correspond to your init function:
    B=0.267   # wheelbase [m]
 
    V=np.diag([0.004,0.002]) # process noise cov
#    W=np.array([[1.05,1.478],[1.478,2.873]]) # measurement noise cov
    W = np.diag([0.05**2, 0.05**2])

    V_mean = np.zeros((2,1))
    W_mean = np.array([[0],[0]])
    Last_state = np.array([[internalStateIn[0]],[internalStateIn[1]],[internalStateIn[2]]])
    Pm = internalStateIn[3]


    if (np.isnan(measurement[0]) or np.isnan(measurement[1])):

        tmp_state = np.concatenate((Last_state,V_mean,W_mean),axis=0) 
        tmp_var = scipy.linalg.block_diag(Pm, V, W)
        s_new=np.zeros((14,7,1)) # sigma point matrix
        s_xp=np.zeros((14,3,1)) # [x y theta] prediction sigma points
        # compute the sigma points
        for i in range(7):
            s_new[i]=tmp_state+ (scipy.linalg.sqrtm(7*tmp_var))[:,i,np.newaxis]
            s_new[i+7]=tmp_state- (scipy.linalg.sqrtm(7*tmp_var))[:,i,np.newaxis]
        # predict the sigma points trough the model
        for i in range(14):
            v_i = speed + s_new[i,3,0]                    # speed + process noise
            gamma_i = steeringAngle + s_new[i,4,0]        # steering + process noise

            s_xp[i,0,0] = s_new[i,0,0] + v_i * np.cos(s_new[i,2,0]) * dt
            s_xp[i,1,0] = s_new[i,1,0] + v_i * np.sin(s_new[i,2,0]) * dt
            s_xp[i,2,0] = s_new[i,2,0] + v_i * np.tan(gamma_i) * dt / B

        # mean and cov of predicted state
        xp=np.mean(s_xp,axis=0)
        Pp=np.zeros((3,3))
        for i in range(14):
            Pp += (s_xp[i]-xp)@ (s_xp[i]-xp).T/14
        x=xp[0,0]
        y=xp[1,0]
        theta=xp[2,0]
        Pm=Pp



    if not (np.isnan(measurement[0]) or np.isnan(measurement[1])):
        
        tmp_state = np.concatenate((Last_state,V_mean,W_mean),axis=0) 
        tmp_var = scipy.linalg.block_diag(Pm, V, W)
        s_new=np.zeros((14,7,1))
        s_xp=np.zeros((14,3,1))
        for i in range(7):
            s_new[i]=tmp_state+ (scipy.linalg.sqrtm(7*tmp_var))[:,i,np.newaxis]
            s_new[i+7]=tmp_state- (scipy.linalg.sqrtm(7*tmp_var))[:,i,np.newaxis]
        for i in range(14):
            v_i = speed + s_new[i,3,0]                     # speed + process noise
            gamma_i = steeringAngle + s_new[i,4,0]         # steering + process noise

            s_xp[i,0,0] = s_new[i,0,0] + v_i * np.cos(s_new[i,2,0]) * dt
            s_xp[i,1,0] = s_new[i,1,0] + v_i * np.sin(s_new[i,2,0]) * dt
            s_xp[i,2,0] = s_new[i,2,0] + v_i * np.tan(gamma_i) * dt / B

        xp=np.mean(s_xp,axis=0)
        Pp=np.zeros((3,3))
        
        for i in range(14):
            Pp += (s_xp[i]-xp)@ (s_xp[i]-xp).T/14
        # print(Pp)
        s_z=np.zeros((14,2,1))
        # prediction measurements sigma points
        for i in range(14):
            #s_z[i,0,0]= s_xp[i,0,0]+0.5*(B)*np.cos(s_xp[i,2,0])+s_new[i,5,0]
            #s_z[i,1,0]= s_xp[i,1,0]+0.5*(B)*np.sin(s_xp[i,2,0])+s_new[i,6,0]
            # if measurement is at rear axle:
            s_z[i,0,0] = s_xp[i,0,0] + s_new[i,5,0]
            s_z[i,1,0] = s_xp[i,1,0] + s_new[i,6,0]


        zp=np.mean(s_z,axis=0)
        Pzz=np.zeros((2,2))
        # covariance of prediction neasurements
        for i in range(14):
            Pzz += (s_z[i]-zp)@ (s_z[i]-zp).T/14
        Pxz=np.zeros((3,2))
        # cross covariance state - measurements
        for i in range(14):
            Pxz += (s_xp[i]-xp)@ (s_z[i]-zp).T/14

        # Kalman gain
        K=Pxz@ np.linalg.inv(Pzz)
        z=np.array([[measurement[0]],[measurement[1]]])
        xm=xp+K@(z-zp)
        Pm=Pp-K@Pzz@K.T
        x=xm[0,0]
        y=xm[1,0]
        theta=xm[2,0]
        # print(K)
        


    #### OUTPUTS ####
    # Update the internal state (will be passed as an argument to the function at next run)
    internalStateOut = [x,
                     y,
                     theta,
                     Pm
                     ]

    return x, y, theta, internalStateOut 


def main():

    #provide the index of the experimental run you would like to use.
    # Note that using "0" means that you will load the measurement calibration data.
    experimentalRun = 1

    print('Loading the data file #', experimentalRun)
    #experimentalData = np.genfromtxt ('data/run_{0:03d}.csv'.format(experimentalRun), delimiter=',')
    experimentalData = np.genfromtxt ('data/mpc_{0:03d}.csv'.format(experimentalRun), delimiter=',', skip_header=1)
    #===============================================================================
    # Here, we run your estimator's initialization
    #===============================================================================
    print('Running the initialization')
    internalState = estInitialize(experimentalData[0,2],
                                experimentalData[0,3],
                                experimentalData[0,4])


    numDataPoints = experimentalData.shape[0]

    #Here we will store the estimated position and orientation, for later plotting:
    estimatedPosition_x = np.zeros([numDataPoints,])
    estimatedPosition_y = np.zeros([numDataPoints,])
    estimatedAngle = np.zeros([numDataPoints,])

    print('Running the system')

    for k in range(numDataPoints):

        if k == 0:
            dt = 0.0
        else:
            dt = experimentalData[k, 0] - experimentalData[k-1, 0]
        
        t = experimentalData[k,0]
        measx   = experimentalData[k,2]   # x measurement (position of sensor)
        measy   = experimentalData[k,3]   # y measurement
        measyaw = experimentalData[k,4]   # yaw measurement (optional, not used yet)
        gamma   = experimentalData[k,6]   # steering angle
        speed   = experimentalData[k,5]   # linear speed

        #run the estimator:
        x, y, theta, internalState = estRun(t, dt, internalState, gamma, speed, (measx, measy))

        #keep track:
        estimatedPosition_x[k] = x
        estimatedPosition_y[k] = y
        estimatedAngle[k] = theta


    print('Done running')
    #make sure the angle is in [-pi,pi]
    estimatedAngle = np.mod(estimatedAngle+np.pi,2*np.pi)-np.pi

    posErr_x = estimatedPosition_x - experimentalData[:,2]  # skip header row
    posErr_y = estimatedPosition_y - experimentalData[:,3]
    angErr   = np.mod(estimatedAngle - experimentalData[:,4] + np.pi, 2*np.pi) - np.pi

    print('Final error: ')
    print('   pos x =',posErr_x[-1],'m')
    print('   pos y =',posErr_y[-1],'m')
    print('   angle =',angErr[-1],'rad')

    ax = np.sum(np.abs(posErr_x))/numDataPoints
    ay = np.sum(np.abs(posErr_y))/numDataPoints
    ath = np.sum(np.abs(angErr))/numDataPoints
    score = ax + ay + ath
    if not np.isnan(score):
        #this is for evaluation by the instructors
        print('average error:')

        print('   pos x =', ax, 'm')
        print('   pos y =', ay, 'm')
        print('   angle =', ath, 'rad')

        #our scalar score. 
        print('average score:',score)


    #===============================================================================
    # make some plots:
    #===============================================================================
    #feel free to add additional plots, if you like.
    print('Generating plots')

    figTopView, axTopView = plt.subplots(1, 1)
    #axTopView.plot(experimentalData[:,2], experimentalData[:,3], 'rx', label='Meas')
    axTopView.plot(estimatedPosition_x, estimatedPosition_y, 'bx', label='est')
    axTopView.plot(experimentalData[:,2], experimentalData[:,3], 'k:.', label='meas')
    axTopView.legend()
    axTopView.set_xlabel('x-position [m]')
    axTopView.set_ylabel('y-position [m]')

    figHist, axHist = plt.subplots(5, 1, sharex=True)
    axHist[0].plot(experimentalData[:,0], experimentalData[:,2], 'k:.', label='meas')
    #axHist[0].plot(experimentalData[:,0], experimentalData[:,2], 'rx', label='Meas')
    axHist[0].plot(experimentalData[:,0], estimatedPosition_x, 'b-', label='est')

    axHist[1].plot(experimentalData[:,0], experimentalData[:,3], 'k:.', label='meas')
    #axHist[1].plot(experimentalData[:,0], experimentalData[:,3], 'rx', label='Meas')
    axHist[1].plot(experimentalData[:,0], estimatedPosition_y, 'b-', label='est')

    axHist[2].plot(experimentalData[:,0], experimentalData[:,4], 'k:.', label='meas')
    axHist[2].plot(experimentalData[:,0], estimatedAngle, 'b-', label='est')

    axHist[3].plot(experimentalData[:,0], experimentalData[:,6], 'g-', label='meas')
    axHist[4].plot(experimentalData[:,0], experimentalData[:,5], 'g-', label='meas')

    axHist[0].legend()

    axHist[-1].set_xlabel('Time [s]')
    axHist[0].set_ylabel('Position x [m]')
    axHist[1].set_ylabel('Position y [m]')
    axHist[2].set_ylabel('Angle theta [rad]')
    axHist[3].set_ylabel('Steering [rad]')
    axHist[4].set_ylabel('Speed [m/s]')

    print('Done')
    plt.show()

if __name__ == "__main__":
    main()