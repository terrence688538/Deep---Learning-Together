from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt

# 载入数据
data = np.genfromtxt(r'D:\资源\数据科学\机器学习实战\聚类\kmeans.txt', delimiter=" ")
# 训练模型
# eps距离阈值，min_samples核心对象在eps领域的样本数阈值
model = DBSCAN(eps=1.5, min_samples=4)
model.fit(data)

result = model.fit_predict(data)
result
# 画出各个数据点，用不同颜色表示分类
mark = ['or', 'ob', 'og', 'oy', 'ok', 'om']
for i, d in enumerate(data):
    plt.plot(d[0], d[1], mark[result[i]])

plt.show()


###例二

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

x1, y1 = datasets.make_circles(n_samples=2000, factor=0.5, noise=0.05)
x2, y2 = datasets.make_blobs(n_samples=1000, centers=[[1.2,1.2]], cluster_std=[[.1]])

x = np.concatenate((x1, x2))
plt.scatter(x[:, 0], x[:, 1], marker='o')
plt.show()

from sklearn.cluster import KMeans
y_pred = KMeans(n_clusters=3).fit_predict(x)
plt.scatter(x[:, 0], x[:, 1], c=y_pred)
plt.show()

from sklearn.cluster import DBSCAN
y_pred = DBSCAN().fit_predict(x)
plt.scatter(x[:, 0], x[:, 1], c=y_pred)
plt.show()

y_pred = DBSCAN(eps = 0.2).fit_predict(x)
plt.scatter(x[:, 0], x[:, 1], c=y_pred)
plt.show()

y_pred = DBSCAN(eps = 0.2, min_samples=50).fit_predict(x)
plt.scatter(x[:, 0], x[:, 1], c=y_pred)
plt.show()