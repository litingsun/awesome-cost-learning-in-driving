import pickle as pkl
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
import numpy as np

from scipy.optimize import minimize
from scipy.optimize import Bounds
import sys
sys.path.append("C:\\Users\\11657\\AppData\\Local\\conda\\conda\\envs\\myroot\\lib\\site-packages")
from autograd import grad, jacobian

Curv_name = "bottom" # which curve to output, the lower or the higher
# Curv_name = "top" # which curve to output, the lower or the higher

with open("CurbPts.pkl","rb") as f:
    CurbPts = pkl.load(f)
CurbPts = CurbPts.transpose()

"""
Cluster to get different part of the lanes
"""
from sklearn.cluster import SpectralClustering,AgglomerativeClustering
clustering = AgglomerativeClustering(n_clusters=7,linkage="single").fit(CurbPts)


curvs = []
for g in range(7):
    curv = np.array([x for i,x in zip(clustering.labels_, CurbPts) if i == g])
    curvs.append(curv)

def referenceedge(points,x,side,direction = 0,k = 6):
    """
    x is the x axis, side is 1 if upper, -1 if lower
    direction is 0 means default find the flat lane
    k is the nearst points take into consider
    """
    distance = [abs(p[0]-x) for p in points] 
    select = np.argpartition(distance,k)[:k]
    points = points[select]
    ys = [side*p[1] for p in points]
    return side * np.partition(ys,2)[:2].mean()


Lanes = {"low": np.concatenate([ c for c in curvs if c.mean(axis=0)[1]<1010]),
         "mid": np.concatenate([ c for c in curvs if 1010 < c.mean(axis=0)[1]<1030]),
         "top": np.concatenate([ c for c in curvs if 1030 < c.mean(axis=0)[1]])}

if(Curv_name=="bottom"):
    X = np.arange(min(Lanes["low"][:,0]),max(Lanes["low"][:,0]),0.5)
    bounds = Bounds([referenceedge(Lanes["low"],x,-1) for x in X ],[referenceedge(Lanes["mid"],x,1) for x in X ])
elif(Curv_name=="top"):
    X = np.arange(min(Lanes["top"][:,0]),max(Lanes["top"][:,0]),0.5)
    bounds = Bounds([referenceedge(Lanes["mid"],x,-1) for x in X ],[referenceedge(Lanes["top"],x,1) for x in X ])

lowb = bounds.lb
upb = bounds.ub

# plt.plot(Lanes["top"][:,0],Lanes["top"][:,1],".")
# for curv in curvs:
#     plt.plot(curv[:,0],curv[:,1],".")
# plt.plot(X,bounds.lb)
# plt.plot(X,bounds.ub)
# plt.show()
# exit()

bound_fac = 1e-3

def cost_refLine(Y):
    # add the curvature cost of the line
    k = Y[1:]-Y[:-1]
    cost = np.sum((k[1:]-k[:-1])**2)
    cost += - np.sum(np.log(Y-lowb)) * bound_fac
    cost += - np.sum(np.log(upb - Y)) * bound_fac
    return cost

def cost_jab(x):
    dev = 2*np.array([0]+list(x[1:]-x[:-1]))
    dev += 2*np.array(list(x[:-1]-x[1:])+[0])
    dev += 1/(x-lowb) * bound_fac
    dev += 1/(x-upb)  * bound_fac
    return dev

Y = (np.array(bounds.lb )+ np.array(bounds.ub))/2
res = minimize(cost_refLine, Y, method='trust-constr',jac=cost_jab ,
                options={'verbose': 1}, bounds=bounds)

ref_curve = res.x

points = np.concatenate((X[:,None],ref_curve[:,None]),axis = 1)
with open("ref_{}.pkl".format(Curv_name),"wb") as f:
    pkl.dump(points,f)

for curv in curvs:
    plt.plot(curv[:,0],curv[:,1],".")
plt.plot(X,bounds.lb)
plt.plot(X,bounds.ub)
Y = (np.array(bounds.lb )+ np.array(bounds.ub))/2
# plt.plot(X,Y)
plt.plot(X,res.x)
plt.show()