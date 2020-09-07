# 导入算法包以及数据集
from sklearn import neighbors
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import random

# 载入数据
iris = datasets.load_iris()
print(iris)

# 打乱数据切分数据集
x_train,x_test,y_train,y_test = train_test_split(iris.data, iris.target, test_size=0.2) #分割数据0.2为测试数据，0.8为训练数据

#切分数据集
test_size = 40
x_train = iris.data[test_size:]
x_test =  iris.data[:test_size]
y_train = iris.target[test_size:]
y_test = iris.target[:test_size]

# 构建模型
model = neighbors.KNeighborsClassifier(n_neighbors=3)                   #neighbors中的K近临分类器              里面的参数是自己选的就是临近的几个点，默认是5
model.fit(x_train, y_train)
prediction = model.predict(x_test)

print(classification_report(y_test, prediction))
