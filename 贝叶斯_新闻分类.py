from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split

news = fetch_20newsgroups(subset='all')
print(news.target_names)
print(len(news.data))
print(len(news.target))

print(len(news.target_names))

news.data[0]

print(news.target[0])
print(news.target_names[news.target[0]])

x_train,x_test,y_train,y_test = train_test_split(news.data,news.target)
# train = fetch_20newsgroups(subset='train')
# x_train = train.data
# y_train = train.target
# test = fetch_20newsgroups(subset='test')
# x_test = test.data
# y_test = test.target


#CountVectorizer方法构建单词的字典，每个单词实例被转换为特征向量的一个数值特征，每个元素是特定单词在文本中出现的次数
from sklearn.feature_extraction.text import CountVectorizer                         #词袋模型
texts=["dog cat fish","dog cat cat","fish bird", 'bird']
cv = CountVectorizer()                           #类
cv_fit=cv.fit_transform(texts)                 #类中的方法

#
print(cv.get_feature_names())
print(cv_fit.toarray())

print(cv_fit.toarray().sum(axis=0))

from sklearn import model_selection
from sklearn.naive_bayes import MultinomialNB

cv = CountVectorizer()
cv_data = cv.fit_transform(x_train)
mul_nb = MultinomialNB()

scores = model_selection.cross_val_score(mul_nb, cv_data, y_train, cv=3, scoring='accuracy')
print("Accuracy: %0.3f" % (scores.mean()))



#TfidfVectorizer使用了一个高级的计算方法，称为Term Frequency Inverse Document
#Frequency (TF-IDF)。这是一个衡量一个词在文本或语料中重要性的统计方法。直觉上讲，该方法通过比较在整个语料库的词的频率，寻求在当前文档中频率较高的词。这是一种将结果进行标准化的方法，可以避免因为有些词出现太过频繁而对一个实例的特征化作用不大的情况(我猜测比如a和and在英语中出现的频率比较高，但是它们对于表征一个文本的作用没有什么作用)
from sklearn.feature_extraction.text import TfidfVectorizer
# 文本文档列表
text = ["The quick brown fox jumped over the lazy dog.","The dog.","The fox"]
# 创建变换函数
vectorizer = TfidfVectorizer()
# 词条化以及创建词汇表
vectorizer.fit(text)
# 总结
print(vectorizer.vocabulary_)
print(vectorizer.idf_)
# 编码文档
vector = vectorizer.transform([text[0]])
# 总结编码文档
print(vector.shape)
print(vector.toarray())

# 创建变换函数
vectorizer = TfidfVectorizer()
# 词条化以及创建词汇表
tfidf_train = vectorizer.fit_transform(x_train)

scores = model_selection.cross_val_score(mul_nb, tfidf_train, y_train, cv=3, scoring='accuracy')
print("Accuracy: %0.3f" % (scores.mean())


def get_stop_words():
    result = set()
    for line in open('stopwords_en.txt', 'r').readlines():
        result.add(line.strip())
    return result

# 加载停用词
stop_words = get_stop_words()
# 创建变换函数
vectorizer = TfidfVectorizer(stop_words=stop_words)


mul_nb = MultinomialNB(alpha=0.01)

# 词条化以及创建词汇表
tfidf_train = vectorizer.fit_transform(x_train)

scores = model_selection.cross_val_score(mul_nb, tfidf_train, y_train, cv=3, scoring='accuracy')
print("Accuracy: %0.3f" % (scores.mean()))

# 切分数据集
tfidf_data = vectorizer.fit_transform(news.data)
x_train,x_test,y_train,y_test = train_test_split(tfidf_data,news.target)

mul_nb.fit(x_train,y_train)
print(mul_nb.score(x_train, y_train))

print(mul_nb.score(x_test, y_test))