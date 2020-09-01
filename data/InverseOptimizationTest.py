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
import pickle as pkl

import optimize
import pdb
def checkFeatureCountSingle(ftrList, weights, normalizers, data, wholedata, ftrList_for_check = None, 
            sampleTrajLists = [], savepath = None, Func = None, data_dict = None,**kwargs):
    data_dict = {} if data_dict is None else data_dict
    if(ftrList_for_check is None):
        ftrList_for_check = ftrList
    weights = weights / normalizers
    ErrorList = []
    originFtrVecs = []
    newFtrVects = []
    sampledTrajs = []
    
    percentErrors = []
    spatialErrors = []
    data_args = [{k:v[i] for k,v in data_dict.items()} for i in range(len(list(data_dict.values())[0]))]
    for i,(wholetraj,traj,darg) in enumerate( zip(wholedata, data,data_args)):
        print("Traj:", i)
        try:
            if Func is None:
                Ftr = features.Features(traj.T.reshape(-1),**darg)
            else:
                Ftr = features.featureFuncWrapper(traj.T.reshape(-1),Func =Func,**darg)
            originFtrVec = np.array([Ftr.featureValue(f) for f in ftrList_for_check])
            
            optres = optimize.DS_single_simulate(weights,ftrList, traj, wholetraj, Func = Func,
                                         feedGradient = True, cacheKey=None, **kwargs,**darg)
           # pdb.set_trace()
            if Func is None:
                Ftr = features.Features(optres.T.reshape(-1),**darg)
            else:
                Ftr = features.featureFuncWrapper(optres.T.reshape(-1),Func =Func,**darg)
            newFtrVec = np.array([Ftr.featureValue(f) for f in ftrList_for_check])
            print("originFtrvec:",originFtrVec,"\n new Ftr Vec:", newFtrVec)
            print("len optres:", len(optres))
        except Exception as ex:
            print(str(ex))
            ErrorList.append((i,ex))
            continue

        # print(originFtrVec,newFtrVec)
        originFtrVecs.append(originFtrVec)
        newFtrVects.append(newFtrVec)
        if(i in sampleTrajLists):
            sampledTrajs.append(optres)
        percentErrors += list(abs(newFtrVec - originFtrVec)/originFtrVec)
        spatialErrors.append(np.linalg.norm(optres - traj))
        
    err = np.linalg.norm(np.array(percentErrors))
    spatialErr = np.mean(spatialErrors)
    if(savepath is not None):
        with open(savepath,"wb") as f:
            pkl.dump([ftrList, weights, normalizers, err, originFtrVecs, newFtrVects, sampledTrajs,sampleTrajLists,ErrorList],f)
    return (err,spatialErr), originFtrVecs, newFtrVects, sampledTrajs



def checkFeatureCountDouble(ftrList, weights, normalizers, wholedata, data, actInds, ftrList_for_check = None, sampleTrajLists = [], savepath = None, **kwargs):
    """
    return:
        err: the average feature count diviation 
        spatialErr: The average spatial error, spatial error is the eucliden distance between the trajectory
    """
    if(ftrList_for_check is None):
        ftrList_for_check = ftrList
    weights = weights / normalizers

    originFtrVecs = []
    newFtrVects = []
    sampledTrajs = []
    
    percentErrors = []
    sampleTrajLists_ = sampleTrajLists # make this backup because not all indexes in this list gets appended (they may be passed by exception)
    sampleTrajLists = []
    ErrorList = []
    spatialErrors = []
    for i,((wholeego_car, wholeother_car), (ego_car, other_car),actind) in enumerate( zip(wholedata, data, actInds)):
        print("Traj:", i)
        try:
            print("len(ego_car)",len(ego_car))
            wholeego_car    = wholeego_car[:,:2]
            wholeother_car  = wholeother_car[:,:2]
            ego_car         = ego_car[:,:2]
            other_car       = other_car[:,:2]

            Ftr = features.DoubleCarSRWrapper(ego_car.T.reshape(-1), other_vec = other_car, 
                            whole = wholeego_car, otherwhole = wholeother_car, actInd = actind)
            originFtrVec = np.array([Ftr.featureValue(f) for f in ftrList_for_check])
            #pdb.set_trace()            
            optres = optimize.DS_double_simulate(weights,ftrList, ego_car, other_car, wholeego_car, wholeother_car, actind,
                                        feedGradient = True, cacheKey=None,**kwargs)
            #pdb.set_trace()
            Ftr.update(optres.T.reshape(-1))

            newFtrVec = np.array([Ftr.featureValue(f) for f in ftrList_for_check])
            print("originFtrvec:",originFtrVec,"\n new Ftr Vec:", newFtrVec)
            print("len optres:", len(optres))
        except Exception as ex:
            print(str(ex))
            ErrorList.append((i,ex))
            continue

        # print(originFtrVec,newFtrVec)
        originFtrVecs.append(originFtrVec)
        newFtrVects.append(newFtrVec)
        if(i in sampleTrajLists_):
            sampledTrajs.append(optres)
            sampleTrajLists.append(i)
        percentErrors += list(abs(newFtrVec - originFtrVec)/originFtrVec)
        spatialErrors.append(np.linalg.norm(optres - ego_car))

    err = np.linalg.norm(np.array(percentErrors))/len(wholedata)
    spatialErr = np.mean(spatialErrors) # the euclidean distance between optresult and ground truth
    if(savepath is not None):
        with open(savepath,"wb") as f:
            pkl.dump([ftrList, weights, normalizers, err, originFtrVecs, newFtrVects, sampledTrajs,sampleTrajLists, ErrorList],f)
    return (err,spatialErr), originFtrVecs, newFtrVects, sampledTrajs

## UTIL Functions


def get_most_diviative_data_index(datavec):
    return np.argmax(abs(datavec-datavec.mean()))


def filter_abnormal_data(datavec, thresh = 1.1):
    """
        return the indexs in the list that has a value too different from the other values
    """
    datavec = np.array(datavec)
    ind = get_most_diviative_data_index(datavec)
    tmp = np.delete(datavec,ind)
    getRidSet = []
    while( datavec.std() > tmp.std() * thresh):
        getRidind = ind
        while(np.sum(np.array(getRidSet)<ind)>getRidind - ind):
            getRidind = getRidind + np.sum(np.array(getRidSet)<ind)
        getRidSet.append(getRidind)
        print("get rid of:",getRidind)
        datavec = tmp
        ind = get_most_diviative_data_index(datavec)
        tmp = np.delete(datavec,ind)
        
    return datavec, getRidSet

def trajs_percent_diviation(originFtrVecs, newFtrVects,ftrList, savepath = None, doc = None, show=True):
    """
        savePath will be a folder to save the pictures
    """
    relativeErrors = [ [(fn[i]-fo[i])/fo[i]  for fo, fn in zip(originFtrVecs, newFtrVects) ] for i in range(len(ftrList))]
    getRidSets = set()
    for rE, fName in zip(relativeErrors,ftrList):
        plt.hist(rE,bins = int(np.sqrt(len(originFtrVecs)))*2)
        plt.title(fName)
        plt.xlabel("relative Error")
        plt.ylabel("count")
        if(savepath is not None):
            os.makedirs(savepath, exist_ok=True)
            plt.savefig(os.path.join(savepath,"diviation_distribution_{}.png".format(fName)))
        if(doc is not None):
            doc.addplt(savepath)
        if(show):
            plt.show()
        else:
            plt.clf()
        
        rE,getrid = filter_abnormal_data(rE)
        print(getrid)
        getRidSets = getRidSets.union( set(list(getrid)))
        plt.hist(rE,bins = int(np.sqrt(len(originFtrVecs)))*2)
        plt.title(fName+" after get rid of anormalies")
        plt.xlabel("relative Error")
        plt.ylabel("count")
        if(savepath is not None):
            plt.savefig(os.path.join(savepath,"diviation_distribution_{}_get_out_of_anormaly.png".format(fName)))
        if(doc is not None):
            doc.addplt(savepath)
        if(show):
            plt.show()
        else:
            plt.clf()
        
    
    

if __name__ == "__main__":
    import argparse

    sys.path.append(os.path.join(currentPath,"roundAboutDoubleCar"))
    import dataPreparation
    sys.path.append(os.path.join(os.path.dirname(currentPath),"util"))
    import visulizations
    # extract the data

    matfile = './raw/roundAboutDoubleCar/interact_data_whole_1003.mat'
    dataPreparation.Wholetraj = "double"
    dataext = dataPreparation.extract(matfile)
    wholedata, data, dataInds = [d[0] for d in dataext] ,[d[1] for d in dataext], [d[2][1] for d in dataext]

    ftrList = ["L2_a_lon","L2_v_des","L2_a_lat","L1_future_distance",]
    weights = np.array([0.4, 0.26, 0.16, 0.18 ])
    normalizers = np.array([0.44179708, 1, 1, 0.1])

    checkFeatureCountDouble(ftrList,weights,normalizers,wholedata,data,dataInds)

