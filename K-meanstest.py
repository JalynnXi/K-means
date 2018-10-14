import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors,datasets

iris = datasets.load_iris()
X = iris.data[:, : 2]
Y = iris.target
clf = neighbors.KNeighborsClassifier(n_neighbors=15, weights='uniform').fit(X,Y)

h = .02
x_min, x_max = X[:, 0].min()-1, X[:, 0].max()+1
y_min, y_max = X[:, 1].min()-1, X[:, 1].max()+1
xx,yy = np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])


cmap_light = ListedColormap(['#FFAAAA','#AAFFAA','#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
Z = Z.reshape(xx.shape)

plt.figure()
plt.pcolormesh(xx,yy,Z,cmap=cmap_light)
plt.scatter(X[:,0],X[:,1],c=Y,cmap=cmap_bold,marker='o')
plt.xlim(xx.min(),xx.max())
plt.ylim(yy.min(),yy.max())
plt.title("3-Class classification(k=%i,weights='distance')%k_neighbors")
plt.show()

correct = 0
for i in range(len(iris.data)):
    if Z[i] == iris.target[i]:
        correct += 1
        print (correct/len(iris.data))
