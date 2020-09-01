"""
    The interface that returns IRL loss in one function call, given feature list and the trajectory
"""
import sys
import os
currentPath = os.path.dirname(os.path.abspath(__file__))
sys.path.append(currentPath)


import learnFactors
import USA_roundaboutSR_independent.dataPreparation as independent_dataPreparation
import roundAboutDoubleCar.dataPreparation as doubleCar_dataPreparation
import numpy as np


independent_dataPreparation.features.SINGULARVLAUENORM = True


doubleCar_dataPreparation.DATASET_NORMALIZE = False


#### Configure the LearnFactor module
learnFactors.SURROGATEOBJ = False # whether or not to use g^T·g to replace g^T·H^{-1}·g
learnFactors.SUM1 = True # whether or not all the factors are sumed to 1
learnFactors.NEGFACTORLOSS = True

def main_independent(featureList, weights, trajectories, reftrajs = None):
    """
    get the IRL training loss of the weight given the bunch of the trajectories
    args FeatureList:   A list of Features, which is a string like `L2_a_lon`
    args weights:       A list of weights, each element is a weight with same length as the feature List
    args trajectories:  A list of trajectories, each trajectory is an nx2 array
    args reftrajs:      A list of reference trajectories, each trajectory is an nx2 array
    """
    if reftrajs is None:
        features = [independent_dataPreparation.preparation(traj[:,0],traj[:,1],featureList) for traj in trajectories]
    else:
        features = [independent_dataPreparation.preparation(traj[:,0],traj[:,1],featureList, ref = r) for traj,r in zip(trajectories,reftrajs)]
    dump_symbols = []
    for k in featureList:
        dump_symbols.append("g_"+k)
        dump_symbols.append("H_"+k)
        dump_symbols.append(k+"_normalizer")
    features = np.array(features)
    loss, w_ = learnFactors.testWeights(dump_symbols,features,featureList,weights)
    return loss



def main_doubleCar(featureList, weights, trajectories, whole_reftrajs, inter_points):
    """
    get the IRL training loss of the weight given the bunch of the trajectories
    args FeatureList:   A list of Features, which is a string like `L2_a_lon`
    args weights:       A list of weights, each element is a weight with same length as the feature List
    args trajectories:  A list of trajectories, each trajectory is a pair(ego, other), where each element is an nx2 array
    args whole_reftrajs: A list of Whole trajectories, each trajectory is a pair(ego, other), where each element is an nx2 array
    args inter_point:   A list of pairs of integers, (egocar_interaction_index, othercar_interaction_index), the `interaction_index` means 
            the start point of interaction_trajectory in whole trajectory
    """
    
    features = [doubleCar_dataPreparation.preparation(whole, data, (None,inter), featureList)[0] for whole, data, inter in zip(trajectories,whole_reftrajs,inter_points)]
    dump_symbols = []
    for k in featureList:
        dump_symbols.append("g_"+k)
        dump_symbols.append("H_"+k)
        dump_symbols.append(k+"_normalizer")
    features = np.array(features)
    loss, w_ = learnFactors.testWeights(dump_symbols,features,featureList,weights)
    return loss



if __name__ == '__main__':

    ## test the independent main
    # import pickle as pkl
    # import pandas as pd
    # featureList= [
    #             "L2_a_lon",
    #             "L2_a_lat",
    #             "L2_j_lon",
    #             "L2_v_des",
    #             "L2_ref_d",
    #             "L2_ref_sinphi"]

    # weight =  [ [0.16785511708046885,0.21255474025513357,0.15425785557437988,0.16705139464428284,0.16735411855356885,0.13092677389216598],
    #             [0.166551, 0.166069, 0.166784, 0.166741, 0.16601, 0.167845 ]]

    # trajs, _ = pkl.load(open("./generatedFiles/DR_USA_Intersection_GL/TRAJ_US_IS_GL_indep_9-14.pkl","rb"))
    # xy_vecs = [dataPreparation.column_select(df)  for df in trajs ]

    # # select the first 4 trajectories and references
    # trajectories = [np.array(xy).T for xy in xy_vecs][:4]
    # ref = [pd.read_csv("./raw/DR_USA_Intersection_GL/reference_path_results_DR_USA_Intersection_GL_ref_from_1A_to_10.csv",header = None).values for _ in trajectories]
    
    # print(main(featureList, weight, trajectories, ref))
    

    ## test the double_car main
    matfile = os.path.join(os.path.dirname(currentPath),"tasks","raw","roundAboutDoubleCar","interact_data_whole_1003.mat")
    doubleCar_dataPreparation.Wholetraj = "double"
    dataext = doubleCar_dataPreparation.extract(matfile)
    wholedata, data, datainfo = [d[0] for d in dataext] ,[d[1] for d in dataext], [d[2][1] for d in dataext]
    featureList= [
                "L2_a_lon",
                "L2_j_lon",
                "L2_a_lat",
                "L2_abs_v_des",
                "L1_future_distance",
                "L1_future_inter_dist"]
    
    # get a subset of the dataset
    wholedata, data, datainfo = wholedata[:4], data[:4], datainfo[:4]
    weight =  [ [0.13669053015975755, 0.35535477216563277, 0.5073111166330064, 0.0002777776107272573, 0.0002943335478645073, 7.1469883011408e-05]
                ]
    print(main_doubleCar(featureList, weight,  wholedata, data, datainfo))