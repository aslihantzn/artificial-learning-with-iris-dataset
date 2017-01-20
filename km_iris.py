#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
=========================================================
K-means Sınıflandırma / İRİS ÇİÇEĞİ
=========================================================
"""
print(__doc__)

import numpy as np      
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


from sklearn.cluster import KMeans
from sklearn import datasets


merkezler = [[1, 1], [-1, -1], [1, -1]]
iris = datasets.load_iris()
X = iris.data
y = iris.target

tahmin =     {'k_means_iris_3': KMeans(n_clusters=3),
              'k_means_iris_5': KMeans(n_clusters=5),
              'k_means_iris_enkotu': KMeans(n_clusters=3, n_init=1,
                                              init='random')}


grafik_num = 1
for isim, est in tahmin.items():
    grafik = plt.figure(grafik_num, figsize=(8, 6))
    plt.clf()
    ax = Axes3D(grafik, elev=-150, azim=110)

    plt.cla()
    est.fit(X)
    labels = est.labels_

    ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=labels.astype(np.float))

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel('TacYaprak Genislik')
    ax.set_ylabel('CanakYaprak Uzunluk')
    ax.set_zlabel('TacYaprak  Uzunluk')

grafik_num = grafik_num + 1


grafik = plt.figure(grafik_num, figsize=(8, 6))
plt.clf()    
# grafiği temizler = clear figure

ax = Axes3D(grafik, elev=-150, azim=110)    
#azimuth = yatay (horizon) x ekseni , elev(ation) = yükseklik-y ekseni

plt.cla()     
#ekseni temizler  =  clear axis

for isim, label in [('Setosa', 0),
                    ('Versicolour', 1),
                    ('Virginica', 2)]:

    ax.text3D(X[y == label, 2].mean(),
              X[y == label, 0].mean()+1,
              X[y == label, 1].mean(), isim,
              horizontalalignment='center', 
                  #merkezden itibaren dağıtıyor
              bbox=dict(boxstyle="sawtooth", edgecolor='pink', facecolor='w')
            # bbox=dict(boxstyle="roundtooth", fc="cyan", ec="b",lw=2)
             )

# renklerine göre sınıflandırmaar eşleştiriliyor.
y = np.choose(y, [0, 1, 2]).astype(np.float)
ax.scatter(X[:, 2], X[:, 0], X[:, 1], c=y)

#x-y-z eksenleri & isimleri;

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel('TacYaprak Genislik')
ax.set_ylabel('CanakYaprak Uzunluk')
ax.set_zlabel('TacYaprak  Uzunluk')

plt.show()  #grafiği gösterir
