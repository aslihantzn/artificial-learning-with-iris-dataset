#!/usr/bin/python


"""
=========================================================
LOJİSTİK REGRESYON İLE SINIFLANDIRMA
=========================================================

"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets

iris = datasets.load_iris()
X = iris.data[:, :2]  
Y = iris.target

h = .02  # adım mesafe

logreg = linear_model.LogisticRegression(C=1000)    #c=1e4   dizi boyutu gösterimi
						    # aslında 10^4 demek = 1000

# örnek komşu sınıfar oluşturuluyor grafik için
logreg.fit(X, Y)

x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])     #z=lr tahmini 
# x ve y yeniden değerlendiriliyor

# renklendirme    && dizi boyutu
Z = Z.reshape(xx.shape)
plt.figure(1, figsize=(6, 5))
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.rainbow)
#http://matplotlib.org/examples/color/colormaps_reference.html  
#renklerin numaraları ve isimleri

# noktalandırma
plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='black', cmap=plt.cm.plasma)
#noktaların renkleri ve dağılımı
plt.xlabel('canak yaprak uzunlugu')
plt.ylabel('canak yaprak genisligi')

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())

plt.show()
