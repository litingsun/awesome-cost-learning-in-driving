# -^- coding:utf-8 -^-
import pickle as pkl
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.cluster import SpectralClustering,AgglomerativeClustering
import numpy as np
# import sklearn
# print('The scikit-learn version is {}.'.format(sklearn.__version__))
with open("roadFeature/CurbPts.pkl","rb") as f:
    CurbPts = pkl.load(f)
CurbPts = CurbPts.transpose()

clustering = AgglomerativeClustering(n_clusters=7,linkage="single").fit(CurbPts)
# clustering = SpectralClustering(n_clusters=7,n_neighbors=2).fit(CurbPts)
for g in range(7):
    curv = np.array([x for i,x in zip(clustering.labels_, CurbPts) if i == g])
    # plt.plot(CurbPts[:,0],CurbPts[:,1],'.')
    plt.plot(curv[:,0],curv[:,1],".")

plt.show()