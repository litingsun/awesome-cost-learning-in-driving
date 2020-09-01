import sys
import os
currentPath = os.path.dirname(os.path.abspath(__file__))
sys.path.append(currentPath)

import features
import featuresUspace

import autograd.numpy as np
from autograd.misc.optimizers import adam
from autograd import grad, jacobian

from scipy.optimize import minimize
import matplotlib.pyplot as plt
import time
from datetime import datetime
# SHORTCUT = False
SHORTCUT = True
maxTime = 1800

import signal

def setMaxtime(method):
    def signal_handler(signum, frame):
        raise Exception("Timed out! (%f)"%maxTime)
    def withMaxtime(*args, **kw):
        if(maxTime is not None):
            signal.signal(signal.SIGALRM, signal_handler)
            signal.alarm(maxTime)   # Ten seconds
        return method(*args, **kw)
    return withMaxtime

OPTCACHE = {} # store the optimization results to reuse the result when come across it next time

def clearCache():
    OPTCACHE = {}
CACHEDROPRATE = 0 # the probability of droping the cache

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print (datetime.now().strftime("%m/%d, %H:%M:%S"),'%2.2f ms' %((te - ts) * 1000))
        return result
    return timed

@timeit
# @setMaxtime # Note: this is only available in linux
def DSoptimizeSingleTraj(ftr, weights, featureList, fixHead = 2, divlim = 0.5,  feedGradient = True):
    """
    ##### arg traj_to_simu:   [n x 2], each point is (s, d)
    arg weights:        The weights of the features, (should consider the normalizer)
    arg featureList:    The features to compute, same order with weights
    arg ftr:            The feature object to compute features, which holds the refline and other information
    arg fixHead:        Add the constraint to fix the heading n time steps as the init condition
    arg divlim:         The limit of the diviation
    """
    traj_to_simu = ftr.spaceTransFromXY()
    FIXDIV = divlim ==0 # and False
    timescale = len(traj_to_simu)
    len_dsvec = timescale if FIXDIV else timescale*2
    assert(len(weights)==len(featureList)) 

    def objective(ds_vec):
        # print(ds_vec)
        if(FIXDIV):
            ds_vec = np.concatenate([ds_vec,np.zeros_like(ds_vec)],axis = 0)
        # ds_vec = ds_vec.reshape(-1,2)
        ds_vec = ds_vec.reshape(2,-1).T
        ftr.update(ds_vec)
        obj  = np.sum([ftr.featureValue(f) * w for w,f in zip(weights,featureList)])
        return obj

    def jac(ds_vec):
        if(FIXDIV):
            ds_vec = np.concatenate([ds_vec,np.zeros_like(ds_vec)],axis = 0)
        # ds_vec = ds_vec.reshape(-1,2)
        ds_vec = ds_vec.reshape(2,-1).T
        ftr.update(ds_vec)
        g = np.sum([ftr.featureGradJacobNormalizer(f, False, False)[0][:] * w for w,f in zip(weights, featureList)] ,axis = 0)
        if(FIXDIV):
            return g[:len(g)//2]
        # print(g.shape)
        return g
    
    def Hes(ds_vec):
        """
        This is not used by SQP
        """
        H = np.sum([ftr.featureGradJacobNormalizer(f, False)[1][:,:] * w for w,f in zip(weights, featureList)] ,axis = 0)
        if(FIXDIV):
            return H[:timescale,:timescale]
        # print(g.shape)
        return H

    def seq_cons(ds_vec):
        """
        this constraint ensures that points are one in a sequence.
        """
        MaxLengthFactor = 1.5 # the length of optimize result cannot be more than this times the original length
        if(FIXDIV):
            # return ds_vec[1:] - ds_vec[:-1]
            return np.array(list(ds_vec[1:] - ds_vec[:-1])+ [MaxLengthFactor * traj_to_simu[-1,0] - ds_vec[-1]])
        ds_vec = ds_vec.reshape(2,-1).T
        return np.array(list(ds_vec[1:,0] - ds_vec[:-1,0])+ [MaxLengthFactor * traj_to_simu[-1,0] - ds_vec[-1,0]])


    seq_jac = np.zeros((timescale,len_dsvec))
    seq_jac += np.eye(timescale) * (-1)
    seq_jac[:-1,1:timescale] += np.eye(timescale-1) * 1


    def init_cons(ds_vec):
        """
        This constraint on the x0 y0
        """
        # ds_vec = ds_vec.reshape(-1,2)
        if(FIXDIV):
            return ds_vec[:fixHead] - traj_to_simu[:fixHead,0]
        ds_vec = ds_vec.reshape(2,-1).T
        return (ds_vec[:fixHead] - traj_to_simu[:fixHead]).T.reshape(-1)

    if(FIXDIV):
        init_jac = np.zeros((fixHead,len_dsvec))
        init_jac[:fixHead,:fixHead] = np.eye(fixHead) * 1
    else:
        init_jac = np.zeros((fixHead*2,len_dsvec))
        init_jac[:fixHead,:fixHead] = np.eye(fixHead) * 1
        init_jac[fixHead:,timescale:timescale + fixHead] = np.eye(fixHead) * 1
    
    if(FIXDIV):
        lb , ub = 0, float("inf")
        bounds = np.tile(np.array([lb,ub]),(len(traj_to_simu),1))
    else:
        lb = (np.array([0,-divlim]) * np.ones_like(traj_to_simu)).T.reshape(-1)
        ub = (np.array([float("inf"),divlim])* np.ones_like(traj_to_simu)).T.reshape(-1)
        bounds = np.concatenate([lb[:,None],ub[:,None]],axis = 1)
    if(FIXDIV):
        x0 = traj_to_simu[:,0]
    else:
        x0 = traj_to_simu.T.reshape(-1)

    constraints = [{'type':'ineq','fun':seq_cons,'jac': lambda v: seq_jac }, 
                   {'type':'eq','fun':init_cons ,'jac': lambda v: init_jac}]
    options = {"maxiter" : 50000, "disp"    : 2}
    if(not feedGradient):
        jac = None
    # print(bounds[:,0])
    # print(x0)
    # assert(np.all(bounds[:,0]<=x0) and np.all(x0<=bounds[:,1]) and np.all(seq_cons(x0)>=0))
    res = minimize(objective, x0, bounds=bounds ,jac=jac,hess = Hes,
                constraints=constraints, options = options)
    if(FIXDIV):
        return np.concatenate([res.x,np.zeros_like(res.x)],axis = 0)
    # assert(res.success)
    return res.x

def DS_single_simulate(weights, featureList, traj, wholetraj = None, fixHead = 2, divlim = 0.5, 
                        feedGradient = True, Nshots = 1, cacheKey = None, shortcut = SHORTCUT,
                        Func = None, **kwargs):
    """
    (theta, features[str], traj[n x 2]) -> opt_traj[n x 2]
    input and output trajectories are in xy space while the optimization is done in ds space
    calls on `DSoptimizeSingleTraj` using featuresUspace.SDspaceFtr(features.Features) to do feature calculation
    arg weights:        The weights of the features, (should consider the normalizer)
    arg featureList:    The features to compute, same order with weights
    arg traj:           [n x 2] The trajectory to simulate in xy space            
    arg wholetraj:      [m x 2] The reference trajectory, can be longer or shorter than the traj, if None, then use traj itself instead
    arg fixHead:        Add the constraint to fix the heading n time steps as the init condition
    arg divlim:         The limit of the diviation. If to only optimize `s` rather than `d`, set this to 0
    arg feedGradient:   Whether or not to compute the jacobian for the optimizer
    arg Nshots:         The number of sections that the whole trajectory is divided into.
    arg cacheKey:       (Hashable) The key to reuse the optimization result from the last step, if None, then no result is stored
                                Note: This function will not check the correctness of cachekey. don't use the same cachekey with different arguments
    """
    # plt.plot(traj[:,0],traj[:,1],label = 'traj')
    # plt.plot(wholetraj[:,0],wholetraj[:,1],label = 'wholetraj')
    # plt.legend()
    # plt.show()
    # vec = np.concatenate([traj[:,0],traj[:,1]],axis = 0)
    shotLen = int(np.ceil((len(traj)+(Nshots-1)*fixHead)/Nshots))
    optxyres = []
    caches = OPTCACHE.get(cacheKey,[])
    if(np.random.random()<CACHEDROPRATE): # drop the cache according to cachedroprate
        print("Cache Dropped")
        caches = []
    for i in range(Nshots):
        if(cacheKey is None or i >= len(caches)):
            shottraj = np.copy(traj[i*shotLen - i*fixHead : min(len(traj), (i+1)*shotLen - i*fixHead)])
        else:
            shottraj = OPTCACHE[cacheKey][i].spaceTransToXY().reshape(2,-1).T
        if(i>0):
            shottraj[:fixHead] = optxyres[-1][-fixHead:]
            ## Brutly shift all the following trajectories
            shottraj[fixHead:] += shottraj[fixHead-1] - traj[i*shotLen - (i-1)*fixHead -1]
        vec = shottraj.T.reshape(-1)
        if Func is not None:
            ftr = featuresUspace.SDspaceFtr(features.featureFuncWrapper, vec,wholetraj, shortCut = shortcut,Func = Func,**kwargs)
        else:
            ftr = featuresUspace.SDspaceFtr(features.Features, vec,wholetraj, shortCut = shortcut,**kwargs)
        optDS = DSoptimizeSingleTraj(ftr, weights, featureList,fixHead,divlim, feedGradient)
        optDS = optDS.reshape(2,-1).T

        if(cacheKey is not None):
            ftr.update(optDS)
            optxyvec =  ftr.spaceTransToXY()
        else:
            optxyvec =  ftr.spaceTransToXY(optDS)
        optxyvec =  optxyvec.reshape(2,-1).T
        if(i>0):
            optxyvec = optxyvec[2:]
        optxyres.append(optxyvec)
        if(cacheKey is not None):
            if(i==len(caches)):
                caches.append(ftr)
            elif(i<len(caches)):
                caches[i] = ftr
            else:
                assert False, "Cache illegal access"
    if(cacheKey is not None):
        OPTCACHE[cacheKey] = caches
    return np.concatenate(optxyres,axis = 0)


def DS_double_simulate(weights, featureList, traj, othertraj,  wholetraj = None, otherWhole = None, 
                        actInd = None,fixHead = 2, divlim = 0.5, shortcut = SHORTCUT,
                        feedGradient = True, Nshots = 1, cacheKey = None, **kwargs):
    """
    (theta, features[str], traj[n x 2]) -> opt_traj[n x 2]
    input and output trajectories are in xy space while the optimization is done in ds space
    calls on `DSoptimizeSingleTraj` using featuresUspace.SDspaceFtr(features.Features) to do feature calculation
    arg weights:        The weights of the features, (should consider the normalizer)
    arg featureList:    The features to compute, same order with weights
    arg traj:           [n x 2] The trajectory to simulate in xy space            
    arg othertraj:      [n x 2] The trajectory of the other car
    arg wholetraj:      [m x 2] The reference trajectory, can be longer or shorter than the traj, if None, then use traj itself instead
    arg otherWhole:     [m' x 2] The whole trajectory of the other car, can be longer or shorter than the traj, if None, then use traj itself instead
    arg actInd:         (int, int) The indexes of the location of ego trajectories and other trajectories in the whole and other whole
    arg fixHead:        Add the constraint to fix the heading n time steps as the init condition
    arg divlim:         The limit of the diviation. If to only optimize `s` rather than `d`, set this to 0
    arg feedGradient:   Whether or not to compute the jacobian for the optimizer
    arg Nshots:         The number of sections that the whole trajectory is divided into.
    arg cacheKey:       (Hashable) The key to reuse the optimization result from the last step, if None, then no result is stored
                            Note: This function will not check the correctness of cachekey. don't use the same cachekey with different arguments
    """
    
    # vec = np.concatenate([traj[:,0],traj[:,1]],axis = 0)
    shotLen = int(np.ceil((len(traj)+(Nshots-1)*fixHead)/Nshots))
    optxyres = []
    caches = OPTCACHE.get(cacheKey,[])
    if(np.random.random()<CACHEDROPRATE): # drop the cache according to cachedroprate
        print("Cache Dropped")
        caches = []
    for i in range(Nshots):
        if(cacheKey is None or i >= len(caches)):
            shottraj = np.copy(traj[i*shotLen - i*fixHead : min(len(traj), (i+1)*shotLen - i*fixHead)])
        else:
            print("loaded cache")
            shottraj = OPTCACHE[cacheKey][i].spaceTransToXY().reshape(2,-1).T
        if(i>0):
            if(cacheKey is None or i >= len(caches)):
                initShift = optxyres[-1][-1] - traj[i*shotLen - (i-1)*fixHead -1]
            else:
                initShift = optxyres[-1][-1] - shottraj[fixHead -1]
            shottraj[:fixHead] = optxyres[-1][-fixHead:]
            ## Brutly shift all the following trajectories
            shottraj[fixHead:] += initShift
            # plt.plot(traj[:,0],traj[:,1],'.',alpha = 0.5, label = 'other_car')
            # plt.plot(optxyres[-1][:,0],optxyres[-1][:,1],'.',alpha = 0.5, label = 'last_shot')
            # plt.plot(shottraj[:,0],shottraj[:,1],"--",alpha = 0.5, label = "shot traj")
            # plt.legend()
            # plt.show()
        # visulizations.DoubleCarTrajectory(ego_car, other_car, shottraj,timeAlign =  False)
        vec = shottraj.T.reshape(-1)
        shotother = othertraj[i*shotLen - i*fixHead : min(len(traj), (i+1)*shotLen - i*fixHead)]
        # ftr = featuresUspace.SDspaceFtr(features.DoubleCarSRFeatures,vec,wholetraj,other_vec = shotother,shortCut = SHORTCUT)
        ftr = featuresUspace.SDspaceFtr(features.DoubleCarSRWrapper, vec, wholetraj, shortCut = shortcut, # args for SDspaceFtr
            other_vec = shotother, whole = wholetraj, otherwhole = otherWhole, actInd = actInd) # args for DoubleCarSRWrapper
        optDS = DSoptimizeSingleTraj(ftr, weights, featureList,fixHead,divlim,feedGradient)
        optDS = optDS.reshape(2,-1).T
        if(cacheKey is not None):
            ftr.update(optDS)
            optxyvec =  ftr.spaceTransToXY()
        else:
            optxyvec =  ftr.spaceTransToXY(optDS)
        optxyvec = optxyvec.reshape(2,-1).T
        if(i>0):
            optxyvec = optxyvec[2:]
        optxyres.append(optxyvec)
        if(cacheKey is not None):
            if(i==len(caches)):
                caches.append(ftr)
            elif(i<len(caches)):
                caches[i] = ftr
            else:
                assert False, "Cache illegal access"
    if(cacheKey is not None):
        OPTCACHE[cacheKey] = caches
    return np.concatenate(optxyres,axis = 0)

if __name__ == "__main__":
    sys.path.append(os.path.join(currentPath,"roundAboutDoubleCar"))
    import dataPreparation
    sys.path.append(os.path.join(os.path.dirname(currentPath),"util"))
    import visulizations
    ftrList = ["L2_a_lon","L2_v_des","L2_a_lat","L1_future_distance",]
    weights = np.array([0.4, 0.26, 0.16, 0.18 ])
    normalizers = np.array([1, 1, 1, 0.1])
    weights = weights / normalizers

    # extract the data
    # matfile = './raw/roundAboutDoubleCar/interact_data_whole_1003.mat'
    matfile = os.path.join(os.path.dirname(currentPath),"tasks","raw","roundAboutDoubleCar","interact_data_whole_1003.mat")
    dataPreparation.Wholetraj = "double"
    dataext = dataPreparation.extract(matfile)
    wholedata, data, datainfo = [d[0] for d in dataext] ,[d[1] for d in dataext], [d[2] for d in dataext]

    dataind = 2
    wholeego_car = wholedata[dataind][0][:,:2]
    wholeother_car = wholedata[dataind][1][:,:2]
    ego_car = data[dataind][0][:,:2]
    other_car = data[dataind][1][:,:2]
    actind = datainfo[dataind][1]
    # # test optimization of single car #
    # optres = DS_single_simulate(weights[:3],ftrList[:3],ego_car, Nshots = 3)

    # # visulization
    # plt.plot(ego_car[:,0],ego_car[:,1],label = 'ego_car')
    # plt.plot(optres[:,0],optres[:,1],label = 'optimized_other_car')
    # sampleN = 10
    # sampleT = np.arange(0,len(ego_car),len(ego_car)//sampleN)
    # simulPoints = np.concatenate([ego_car[sampleT].T,optres[sampleT].T],axis = 0)
    # plt.plot(simulPoints[[0,2],:],simulPoints[[1,3],:],'y',linestyle='--',lw = 0.5)
    # plt.legend()
    # plt.title('the traj between old ego_car and optimized ego_car')
    # # plt.savefig("./tmp/%s_trajComp.png"%time.strftime("%m%d%H%M"))
    # plt.show()
    

    # test optimization of double car #
    # optres = DS_double_simulate(weights[:3],ftrList[:3],ego_car,other_car,wholeego_car)

    # optres = DS_double_simulate(weights,ftrList,other_car,ego_car, wholeother_car,feedGradient = False)
    print(len(ego_car))
    optres = DS_double_simulate(weights,ftrList, other_car, ego_car, wholeother_car, wholeego_car, actind,
                                    feedGradient = True, Nshots = 1, cacheKey=None)
    visulizations.DoubleCarTrajectory(ego_car, other_car, optres)
    # optres = DS_double_simulate(weights,ftrList,other_car,ego_car, wholeother_car,feedGradient = True, Nshots = 1,cacheKey=0)
    # visulizations.DoubleCarTrajectory(ego_car, other_car, optres)
