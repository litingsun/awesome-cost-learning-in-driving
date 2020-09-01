"""
use IRL to learn the weights of features in a stopping scenario (GL) based on a L2 functional
"""
import sys
sys.path.append("../data/data_process")
sys.path.append("../util")
sys.path.append("../data")
import learnFactors
import dataPreparation
import pickle as pkl
import MDlogger
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import pandas as pd

## STEP2 Process data calculate features and feature gradients
filename = "../data/DR_USA_Roundabout_EP/TRAJ_US_RD_EP_indep_unlabel9-9.pkl"
dataFile = "../data/DR_USA_Roundabout_EP/TRAJ_US_RD_EP_indep_unlabel9-9_L2_symbol_311.pkl"
all_symbols = [
    "L2_a_lon",
    "L2_j_lon",
    "L2_v_des",
    "L2_final_v"]

ref1 = pd.read_csv("../data/US_roundabout_EP/reference_path_results_DR_USA_Roundabout_EP_ref_1.csv",header = None).values
ref2 = pd.read_csv("../data/US_roundabout_EP/reference_path_results_DR_USA_Roundabout_EP_ref_2.csv",header = None).values
ref3 = pd.read_csv("../data/US_roundabout_EP/reference_path_results_DR_USA_Roundabout_EP_ref_3.csv",header = None).values

def ref_select(x_list,y_list):
    # manually kick out some outliers
    if(x_list[0]<1032):
            ref = ref3
    else:
        if(y_list[-1]<1018):
            ref = ref2
        else:
            ref = ref1
    return ref
dataPreparation.ref_select =  ref_select
dataPreparation.features.SINGULARVLAUENORM = True
dataPreparation.process(filename,dataFile,all_symbols, v_lim = 1.4) 




# Configure the LearnFactor module
learnFactors.SURROGATEOBJ = False # whether or not to use g^T·g to replace g^T·H^{-1}·g
learnFactors.SUM1 = True # whether or not all the factors are sumed to 1
# learnFactors.TOTALEPOCHS = 1000
learnFactors.TOTALEPOCHS = 200
learnFactors.NEGFACTORLOSS = True
learnFactors.INITLR = 0.0000001
learnFactors.LRDECAY = 0.9


with open(dataFile,"rb") as f:
    dump_symbols,features = pkl.load(f)


#### Filter the dataset
features = np.array(features)
rows = set(np.arange(features.shape[0])) 
rows.remove(24) # 24 is not the traj of stoping
rows.remove(34)
rows.remove(54)
rows.remove(62)
rows.remove(159)
rows.remove(163) # theses value besides 24 are trajs that makes the det >0 in the first epoch
rows = list(rows)
features = features[rows,:]

#### Initialize the automatic logger
doc = MDlogger.Doc(filename = "TrainEP.md")

try:
    #### STEP3 LEARN FACTOR
    learnHistory = learnFactors.main(dump_symbols,features,all_symbols)
    os.makedirs("learnHistories",exist_ok=True)
    pkl.dump(learnHistory,open("learnHistories/{}.pkl".format(os.path.basename(__file__)),"wb"))
    learnFactors.plotres(*learnHistory,doc)
finally:
    doc.generate()