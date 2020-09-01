# visulization
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from collections.abc import Iterable


def figOutputer(func,**outKwarg):
    def wrap_func(*args, savepath = None,doc = None,show = True, **kwargs):
        res = func(*args,**kwargs)
        if(savepath is not None):
            if ("." not in savepath): # the save path is not a file name, it is a path name
                os.makedirs(savepath, exist_ok=True)
                try:
                    plt.savefig(os.path.join(savepath,"{}.png".format(kwargs["title"])))
                except:
                    plt.savefig(os.path.join(savepath,"{}.png".format(int(time.time()*10))))
            else:
                plt.savefig(savepath)
        if(doc is not None):
            if(savepath is None):
                doc.addplt()
            else:
                doc.addplt(savepath)
        if(show):
            plt.show()
        else:
            plt.clf()
        return res
    return wrap_func

@figOutputer
def plotRawFeatures(funcs,ftr,titles,legends = None,Nsubplot = (3,3)):
    """
    print plots in a 3 x 3 subplot
    """
    if not isinstance(ftr, Iterable):
        ftr = [ftr]
    for i,(f,t) in enumerate( zip(funcs,titles)):
        ax = plt.subplot(*Nsubplot,i+1)
        if (legends is not None):
            for featr,led in zip(ftr,legends):
                func = getattr(featr,f)
                ax.plot(func(featr.vec),label = led)
        else:
            for featr in ftr:
                func = getattr(featr,f)
                ax.plot(func(featr.vec))
        ax.title.set_text(t)
        plt.legend()
        plt.xlabel("t")
        plt.ylabel("d")

@figOutputer
def DoubleCarTrajectory(ego_car, *otherTrajList, timeAlign = True, alignlw = 0.5, labels = ['ego_car','other_car','optimized_other_car']):
    plt.plot(ego_car[:,0],ego_car[:,1],'.',label = labels[0])
    for i,traj in enumerate(otherTrajList):
        plt.plot(traj[:,0],traj[:,1],'.',label = labels[i+1])
    
    sampleN = 10
    sampleT = np.arange(0,len(ego_car)-1,len(ego_car)//sampleN)
    if(timeAlign):
        simulPoints = np.concatenate(list([ego_car[sampleT].T])+list([traj[sampleT].T for traj in otherTrajList]),axis = 0)
        for i in range(1,len(otherTrajList)+1):
            plt.plot(simulPoints[[0,2*i],:],simulPoints[[1,2*i+1],:],plt.rcParams['axes.prop_cycle'].by_key()['color'][i],linestyle='--',lw = alignlw)
        
    plt.legend()
    plt.title('traj_between_old other_car and optimized other_car')
