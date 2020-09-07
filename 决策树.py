from sklearn.feature_extraction import DictVectorizer
from sklearn import tree
from sklearn import preprocessing
import csv

# 读入数据
Dtree = open(r'D:\资源\数据科学\机器学习实战\决策树\AllElectronics.csv', 'r')
reader = csv.reader(Dtree)

# 获取第一行数据
headers = reader.__next__()
print(headers)

# 定义两个列表
featureList = []
labelList = []

#
for row in reader:
    # 把label存入list
    labelList.append(row[-1])
    rowDict = {}
    for i in range(1, len(row)-1):
        #建立一个数据字典
        rowDict[headers[i]] = row[i]
    # 把数据字典存入list
    featureList.append(rowDict)

print(featureList)


# 把数据转换成01表示
vec = DictVectorizer()
x_data = vec.fit_transform(featureList).toarray()
print("x_data: " + str(x_data))

# 打印属性名称
print(vec.get_feature_names())

# 打印标签
print("labelList: " + str(labelList))

# 把标签转换成01表示
lb = preprocessing.LabelBinarizer()
y_data = lb.fit_transform(labelList)
print("y_data: " + str(y_data))

# 创建决策树模型
model = tree.DecisionTreeClassifier(criterion='entropy')                #entropy 代表熵        C4.5算法
# 输入数据建立模型
model.fit(x_data, y_data)

# 测试
x_test = x_data[0]
print("x_test: " + str(x_test))

predict = model.predict(x_test.reshape(1,-1))
print("predict: " + str(predict))

# 导出决策树
# pip install graphviz
# http://www.graphviz.org/
import graphviz

dot_data = tree.export_graphviz(model,                        #我们训练好的模型
                                out_file = None,                                       #没有输出文件
                                feature_names = vec.get_feature_names(),                   #属性的名字
                                class_names = lb.classes_,                                 #标签      yes no
                                filled = True,                                          #填充色
                                rounded = True,                              #圆型
                                special_characters = True)      #特殊形状
graph = graphviz.Source(dot_data)
graph.render('computer')              #保存到当前目录下

graph

vec.get_feature_names()

lb.classes_