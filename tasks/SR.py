"""
use IRL to explore the features in a free driving scenario (SR), and investigate the learning variance
"""

import pickle as pkl
# import numpy as np
import autograd.numpy as np
import sys
sys.path.append("../data/data_process")
sys.path.append("../util")
sys.path.append("../data")
from features import MetaFunc
import learnFactors
import dataPreparation
import visulizations
import MDlogger
import os

import matplotlib.pyplot as plt
from  features import Features, HighDFeatures, featureFuncWrapper
import pandas as pd
from tabulate import tabulate
import random

#### Declare the symbols(features)
# Because here we are using custom norm functions so the head is named as `Func`
feed_symbols = ['Func_j_lon', 'Func_v_des', 'Func_a_lon', 'Func_a_lat',
         "Func_ref_d", 
         "Func_ref_sinphi"]


#### Prepare and configure the dataPreparation module
trajs = pkl.load(open("../data/DR_USA_Roundabout_SR/TRAJ_US_RD_SR_indep_8-22.pkl","rb"))
xy_vecs = [dataPreparation.column_select(df)  for df in trajs ]

with open("../data/DR_USA_Roundabout_SR/ref_bottom.pkl","rb") as f:
    ref_bottom = np.array(pkl.load(f))
with open("../data/DR_USA_Roundabout_SR/ref_top.pkl","rb") as f:
    ref_top = np.array(pkl.load(f))

ref_select = lambda x_list,y_list: ref_bottom if x_list[0]<1000 else ref_top
dataPreparation.ref_select =  lambda x_list,y_list: ref_bottom if x_list[0]<1000 else ref_top # ref_select: given the trajectory return corresponding reference path
dataPreparation.features.SINGULARVLAUENORM = True


#### Configure the LearnFactor module
learnFactors.SURROGATEOBJ = False # whether or not to use g^T·g to replace g^T·H^{-1}·g
learnFactors.SUM1 = True # whether or not all the factors are sumed to 1
learnFactors.TOTALEPOCHS = 1
learnFactors.NEGFACTORLOSS = True
learnFactors.INITLR = 0.0000001
learnFactors.LRDECAY = 0.9


'''
    Define the body and names of the feature norms. 
    Some examples include:
    1）use feature=sum(exp(abs(value))):
    MetaFunc(lambda a,b,c : np.mean(np.exp(c*np.abs(a - b))), {
            "j_lon":(0.002,), "v_des":(0.0025,), "a_lon":(0.02,), "a_lat":(0.02,), 
            "ref_d":(0.03,), "ref_a_d":(0.0035,), "ref_a_s":(1.,), "ref_sinphi":(2.5,)}),
    2) use feature=exp(abs(L_inf)):
    MetaFunc(lambda a,b,c: np.exp(c*np.max(np.abs(a - b))), {
            "j_lon":(0.002,), "v_des":(0.0025,), "a_lon":(0.02,), "a_lat":(0.02,), 
            "ref_d":(0.03,), "ref_a_d":(0.0035,), "ref_a_s":(1.,), "ref_sinphi":(2.5,)}),
    3) use feature=exp(abs((mean(value)))
    MetaFunc(lambda a,b,c: np.exp(c*np.abs(np.mean(a - b))), {
            "j_lon":(0.002,), "v_des":(0.0025,), "a_lon":(0.02,), "a_lat":(0.02,), 
            "ref_d":(0.03,), "ref_a_d":(0.0035,), "ref_a_s":(1.,), "ref_sinphi":(2.5,)}),
'''
FuncList = [
    MetaFunc(lambda a,b: np.max(np.abs(a - b)) **2, {} ),
    MetaFunc(lambda a,b: (np.mean(a - b))**2, {}),
    MetaFunc(lambda a,b: np.abs(np.mean(a - b)), {}),
    MetaFunc(lambda a,b : np.mean(np.abs(a - b)), {}),
    MetaFunc(lambda a,b : np.sqrt(np.mean((a - b)**2)), {}),
    MetaFunc(lambda a,b : np.max(np.abs(a - b)), {})
    ]

### you can select from the following library: 
    # "sum(exp(abs(value)))",
    # "(L_inf)^2",
    # "exp(abs(L_inf))",
    # "(mean(value))^2",
    # "exp(abs((mean(value)))",
    # "abs(mean(value))",

FuncNames = [
        "(L_inf)^2",
        "(mean(value))^2",
        "abs(mean(value))",
        "L1","L2","Linf"
        ]

THETA_SAMPLE = 2000
DATASET_SIZE = 20
DATASET_SAMPLE = 10


#### Define an util funtion to output result
def plotres(losss, weights, doc, symbols = feed_symbols):
    # Util function to generate the experiment results
    losss = np.array(losss)
    binrange = (np.min(losss), np.max(losss))

    for l in losss:
        print(l)
        plt.hist(l, bins = 20, alpha = 0.2, range = binrange)
    plt.title("distribution of losses")
    doc.addplt()
    plt.clf()
    doc.addparagraph("> each color is a subset of dataset")

    loss = np.transpose(losss,(1,0))
    weight = np.transpose(weights, (1,0,2))
    return loss, weight


#### The main body of training and analysising of one feature norm
def runSubsetTraining(func,funcname,doc):

    #### STEP2 Dataprocess: calculate features
    dataFile = "../data/DR_USA_Roundabout_SR/Ftr%s.pkl"%funcname
    try:
        dump_symbols,features = pkl.load(open(dataFile,"rb"))
        print("loaded from generated file")
    except FileNotFoundError as ex:
        print(ex)
        dataPreparation.process_fromXYtrajs_WithFunc(xy_vecs,dataFile, feed_symbols, Func = func)
        dump_symbols,features = pkl.load(open(dataFile,"rb"))

    #### STEP3 TRAIN
    try:
        features = np.array(features)

        # manually kick out outliers
        removelist = [60, 107,  64,  32, 33, 48] 
        rows = [i for i in np.arange(features.shape[0]) if i not in removelist] # selected rows and reruned in 9-9
        features = features[rows,:]

        rows = set(np.arange(features.shape[0]))
        learnHistorys = []

        init_ws = [np.random.random(len(feed_symbols)) for i in range(THETA_SAMPLE)]
        init_ws = np.array(init_ws)
        init_ws /= init_ws.sum(axis = 1,keepdims=True)
        for i in range(DATASET_SAMPLE):
            #### Randomly select data
            # select_rows = rows
            select_rows = set(random.sample(list(rows),DATASET_SIZE))
            rows = rows - select_rows
            feed_features = features[list(select_rows),:]
            if(len(rows)<21):
                rows = set(np.arange(features.shape[0]))
            #### Train body
            lw = learnFactors.testWeights(dump_symbols,feed_features,feed_symbols,init_ws)
            learnHistorys.append((lw))

        #### STEP4 ANALYSIS
        loss, weight=plotres([h[0] for h in learnHistorys], [h[1] for h in learnHistorys],  doc)

        weight = np.array(weight)

        
        if(len(func.coefDict)):
            doc.addparagraph("### exp coefficient \n```python\n{ ")
            
            for k, v in func.coefDict.items():
                doc.addparagraph("{}:{},".format(k,v))
            doc.addparagraph("}\n```")
            doc.addparagraph("> The exp coefficient is the factor $\alpha$ on the operand of the exp operation $\exp(\alpha f(x))$")

        doc.addparagraph("### The mean and std of each dimension")
        doc.addparagraph(tabulate(df, tablefmt="pipe", headers="keys", showindex=False))

        meanloss = loss.mean(axis = 1)

        doc.addparagraph("the minimum loss is %s, with variance %s, its weight is %s"%(str(min(meanloss)),str(loss[np.argmin(meanloss)].std()) ,str(weight[np.argmin(meanloss)][0])))
        doc.addparagraph("### The original data: (weight, loss)")
        for l,w in zip(loss,weight):
            for ll, ww in zip(l,w):
                doc.addparagraph(str(ww)+" : "+str(ll))
        
    finally:
        doc.generate()


for func, funcname in zip(FuncList,FuncNames):
    try:
        doc = MDlogger.Doc(filename = "0404LearnMeanStdSR%s.md"%funcname)
        doc.addparagraph("## "+funcname)
        doc.addparagraph('THETA_SAMPLE: '+ str(THETA_SAMPLE))
        doc.addparagraph('DATASET_SIZE: '+ str(DATASET_SIZE))
        doc.addparagraph('DATASET_SAMPLE: '+ str(DATASET_SAMPLE))

        runSubsetTraining(func,funcname,doc)
    except Exception as ex:
        doc = MDlogger.Doc(filename = "0404LearnMeanStdSR%s.md"%funcname)
        doc.addparagraph(str(ex))
        doc.generate()
    