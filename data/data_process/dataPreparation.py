# -^- coding:utf-8 -^-
"""
the codes for calculate and dump the features and jacob and gradients

Author: Chenyu Yang: github.com/yangcyself
change log:
2019-09-08 16:56:01:  added the ref_select a callback function
"""
import sys
sys.path.append("../../reference_code/ver3_autograd/")
import autograd.numpy as np
from autograd import grad, jacobian
# https://stackoverflow.com/questions/13855677/how-do-i-approximate-the-jacobian-and-hessian-of-a-function-numerically

import pandas
import features
from  features import Features, HighDFeatures, featureFuncWrapper

import pickle as pkl


EXTRACTSOURCE = False # wether the raw pickle file contains the source of the data
DATASET_MIN_MAX_NORMALIZE = False
DEBUG_SHOT = False # just return 4 trajectories for debugging
ref_select = lambda xl,yl: None # an call back function that selectref line lambda x_list, y_list : ref

def getSetting():
    """
        return a string about all the flags in this file
    """
    return "### dataPreparation Setting:\n\n" + "\n".join(["%s : %s"%(k,str(v)) for k,v in
       [("**DATASET_MIN_MAX_NORMALIZE**",DATASET_MIN_MAX_NORMALIZE)]])


def column_select(df,returnList = ["x","y"]):
    """
    get from the dataframe the desired columns
    REMEMBER TO REORDER THE DATAFRAME BEFORE HAND
    """
    df = df.sort_values(by = 'frame_id')
    return [ list(df[k]) for k in returnList]

def column_select_HighD(dfs):
    """
    get from the dataframe the desired datastructures:
    return: othercarDict: the information about other car
    """
    traj = dfs[0]
    traj = np.array([traj["x"].values,traj["y"].values]).T
    turnPoint = traj[int(len(traj)/2)]
    traj = traj - turnPoint
    x_list = traj[:,0]
    y_list = traj[:,1]
    othercar = ["pcd_sor","flw_sor","pcd_tar","flw_tar"]
    othercarDict = {}
    for i,c in enumerate(othercar):
        trj = dfs[i+1]
        trj,w,h = np.array([trj["x"].values,trj["y"].values]).T - turnPoint, trj["width"].values[0],trj["height"].values[0]
        othercarDict[c] = (trj,w,h)
    return  x_list,y_list,othercarDict

perparation_called_times = 0
calculation_called_times = 0

def preparation(x_list, y_list,feature_symbols,Func = None, ref = None, **kwargs):
    """
    feature symbols are a list of interested feature names (Not with the prefix g_ or H_ )
    """
    global perparation_called_times
    features.DT = 0.1
    print(perparation_called_times)
    xy_vector = np.array(list(x_list) + list(y_list))

    assert(len(x_list)==len(y_list))
    # ref = ref_bottom if x_list[0]<1000 else ref_top
    ref = ref_select(x_list,y_list) if ref is None else ref
    if(Func is None):
        feature = Features(vec=xy_vector, referenceCurv=ref,**kwargs)
    else:
        feature = featureFuncWrapper(vec=xy_vector,  Func = Func, referenceCurv=ref,**kwargs)
    ftrvalues = [np.array([feature.featureValue(f) for f in feature_symbols])]
    return_list = []
    for k in feature_symbols:
        print("symbol:",k)
        g,H,N = feature.featureGradJacobNormalizer(k)
        return_list.append(g)
        return_list.append(H)
        return_list.append(N)
    perparation_called_times +=1
    return return_list, ftrvalues
    # return [ locals()[sym] for sym in dump_symbols] DO NOT work!


def preparationHighD(x_list,y_list,othercarDict,feature_symbols,**kwargs):
    """
    feature symbols are a list of interested feature names (Not with the prefix g_ or H_ )
    """
    global perparation_called_times
    features.DT = 0.04
    print(perparation_called_times)
    xy_vector = np.array(list(x_list)+list(y_list))
    assert(len(x_list)==len(y_list))
    # ref = ref_bottom if x_list[0]<1000 else ref_top
    ref = ref_select(x_list,y_list)
    feature = HighDFeatures(vec=xy_vector, referenceCurv=ref,**othercarDict)
    return_list = []
    for k in feature_symbols:
        print("symbol:",k)
        g,H,N = feature.featureGradJacobNormalizer(k)
        return_list.append(g)
        return_list.append(H)
        return_list.append(N)
    perparation_called_times +=1
    return return_list


def calculateFeature(x_list, y_list,feature_symbols, Func = None, **kwargs):
    """
    feature symbols are a list of interested feature names (Not with the prefix g_ or H_ )
    """
    global calculation_called_times
    features.DT = 0.1
    print(calculation_called_times)
    xy_vector = np.array(list(x_list) + list(y_list))
    assert(len(x_list)==len(y_list))
    ref = ref_select(x_list,y_list)
    if(Func is not None):
        feature = featureFuncWrapper(vec=xy_vector, referenceCurv=ref, Func = Func, **kwargs)
    else:
        feature = Features(vec=xy_vector, referenceCurv=ref,**kwargs)
    return_list = []
    for k in feature_symbols:
        print("symbol:",k)
        return_list.append(feature.featureValue(k))
    calculation_called_times +=1
    return return_list

def calculateFeatureHighD(x_list,y_list,othercarDict,feature_symbols,**kwargs):
    """
    feature symbols are a list of interested feature names (Not with the prefix g_ or H_ )
    """
    global calculation_called_times
    features.DT = 0.04
    print(calculation_called_times)
    xy_vector = np.array(list(x_list) + list(y_list))
    assert(len(x_list)==len(y_list))
    ref = ref_select(x_list,y_list)
    feature = HighDFeatures(vec=xy_vector, referenceCurv=ref,**othercarDict)
    return_list = []
    for k in feature_symbols:
        print("symbol:",k)
        return_list.append(feature.featureValue(k))
    calculation_called_times +=1
    return return_list

def process(DFfile, outfile,feature_symbols,**kwargs):
    # saves the g and H and normalizer of each feature
    NormalizeInfo = {}
    with open(DFfile,"rb") as f:
        if(EXTRACTSOURCE):
            trajs,_ = pkl.load(f)
        else:
            trajs = pkl.load(f)
    trajs = trajs[:10] if DEBUG_SHOT else trajs
    features = [preparation(*column_select(df),feature_symbols,**kwargs) for df in trajs]
    ftrvals = [f[1] for f in features]
    features = [f[0] for f in features]
    if(DATASET_MIN_MAX_NORMALIZE): # normalize over the dataset
        ftrvals = np.array(ftrvals)
        ftr_max = np.max(ftrvals,axis = 0,keepdims = False).flatten()
        ftr_min = np.min(ftrvals,axis = 0,keepdims = False).flatten()
        ftr_dif = ftr_max - ftr_min
        features = [[(ff-ftr_min[i//3])/ftr_dif[i//3] if(i%3!=2) else ff for i, ff in enumerate(f)  ] for f in features]
        NormalizeInfo = {"ftr_max":ftr_max,"ftr_min":ftr_min}
    dump_symbols = []
    for k in feature_symbols:
        dump_symbols.append("g_"+k)
        dump_symbols.append("H_"+k)
        dump_symbols.append(k+"_normalizer")
    with open(outfile,"wb") as f:
        pkl.dump((dump_symbols, features, NormalizeInfo),f)

def process_fromXYtrajs(trajs, outfile,feature_symbols,dump = True,**kwargs):
    # saves the g and H and normalizer of each feature
    trajs = trajs[:10] if DEBUG_SHOT else trajs
    NormalizeInfo = {}
    features = [preparation(*xytuple,feature_symbols,**kwargs) for xytuple in trajs]
    ftrvals = [f[1] for f in features]
    features = [f[0] for f in features]
    if(DATASET_MIN_MAX_NORMALIZE): # normalize over the dataset
        ftrvals = np.array(ftrvals)
        ftr_max = np.max(ftrvals,axis = 0,keepdims = False).flatten()
        ftr_min = np.min(ftrvals,axis = 0,keepdims = False).flatten()
        ftr_dif = ftr_max - ftr_min
        features = [[(ff-ftr_min[i//3])/ftr_dif[i//3] if(i%3!=2) else ff for i, ff in enumerate(f)  ] for f in features]
        NormalizeInfo = {"ftr_max":ftr_max,"ftr_min":ftr_min}
    dump_symbols = []
    for k in feature_symbols:
        dump_symbols.append("g_"+k)
        dump_symbols.append("H_"+k)
        dump_symbols.append(k+"_normalizer")
    if(dump):
        with open(outfile,"wb") as f:
            pkl.dump((dump_symbols, features, NormalizeInfo),f)
    return (dump_symbols,features)


def process_fromXYtrajs_WithFunc(trajs, outfile,feature_symbols,Func,**kwargs):
    # saves the g and H and normalizer of each feature
    trajs = trajs[:10] if DEBUG_SHOT else trajs
    NormalizeInfo = {}
    features = [preparation(*xytuple,feature_symbols,Func = Func, **kwargs) for xytuple in trajs]
    ftrvals = [f[1] for f in features]
    features = [f[0] for f in features]
    if(DATASET_MIN_MAX_NORMALIZE): # normalize over the dataset
        ftrvals = np.array(ftrvals)
        ftr_max = np.max(ftrvals,axis = 0,keepdims = False).flatten()
        ftr_min = np.min(ftrvals,axis = 0,keepdims = False).flatten()
        ftr_dif = ftr_max - ftr_min
        features = [[(ff-ftr_min[i//3])/ftr_dif[i//3] if(i%3!=2) else ff for i, ff in enumerate(f)  ] for f in features]
        NormalizeInfo = {"ftr_max":ftr_max,"ftr_min":ftr_min}
    dump_symbols = []
    for k in feature_symbols:
        dump_symbols.append("g_"+k)
        dump_symbols.append("H_"+k)
        dump_symbols.append(k+"_normalizer")
    with open(outfile,"wb") as f:
        pkl.dump((dump_symbols,features,NormalizeInfo),f)


def processHighD(DFfile, outfile,feature_symbols,**kwargs):
    # saves the g and H and normalizer of each feature
    with open(DFfile,"rb") as f:
            trajs,_ = pkl.load(f)
    features = [preparationHighD(*column_select_HighD(df),feature_symbols,**kwargs) for df in trajs]
    dump_symbols = []
    for k in feature_symbols:
        dump_symbols.append("g_"+k)
        dump_symbols.append("H_"+k)
        dump_symbols.append(k+"_normalizer")
    with open(outfile,"wb") as f:
        pkl.dump((dump_symbols,features),f)


def saveFeature(DFfile, outfile,feature_symbols,Func = None,**kwargs):
    with open(DFfile,"rb") as f:
        if(EXTRACTSOURCE):
            trajs,_ = pkl.load(f)
        else:
            trajs = pkl.load(f)
    features = [calculateFeature(*column_select(df),feature_symbols, Func = Func, **kwargs) for df in trajs]
    with open(outfile,"wb") as f:
        pkl.dump((feature_symbols,features),f)

def saveFeatureHighD(DFfile, outfile,feature_symbols,**kwargs):
    with open(DFfile,"rb") as f:
        trajs,source = pkl.load(f)
    features = [calculateFeatureHighD(*column_select_HighD(dfs),feature_symbols,**kwargs) for dfs in trajs]
    with open(outfile,"wb") as f:
        pkl.dump((feature_symbols,features),f)

if __name__ == "__main__":
    filename = "TRAJ_US_RD_SR_indep_8-22"
    feature_symbols = [ "L2_a_lat",
                        "L2_v_des",
                        "L2_a_lon"]
    process(filename+".pkl",filename+"_Procs_831.pkl",feature_symbols)
    # with open(filename+".pkl","rb") as f:
    #     trajs = pkl.load(f)
    # features = [column_select(df) for df in trajs]
    # with open("xyList.pkl","wb") as f:
    #     pkl.dump(features,f)


