import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import numpy as np
from sklearn.datasets import load_iris  # 数据集
from sklearn.model_selection import train_test_split  # 数据集的分割函数
from sklearn.preprocessing import StandardScaler  # 数据预处理
from sklearn import metrics
from sklearn.model_selection import cross_validate
from sklearn import metrics


class HiddenLayer:
    def __init__(self, x, num):
        row = x.shape[0]
        columns = x.shape[1]
        rnd = np.random.RandomState(4444)
        self.w = rnd.uniform(-1, 1, (columns, num))
        self.b = np.zeros([row, num], dtype=float)
        for i in range(num):
            rand_b = rnd.uniform(-0.4, 0.4)
            for j in range(row):
                self.b[j, i] = rand_b
        h = self.sigmoid(np.dot(x, self.w) + self.b)
        self.H_ = np.linalg.pinv(h)
        # print(self.H_.shape)

    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

    def regressor_train(self, T):
        T = T.reshape(-1, 1)
        self.beta = np.dot(self.H_, T)
        return self.beta

    def classifisor_train(self, T):
        en_one = OneHotEncoder()
        T = en_one.fit_transform(T.reshape(-1, 1)).toarray()  # 独热编码之后一定要用toarray()转换成正常的数组
        # T = np.asarray(T)
        print(self.H_.shape)
        print(T.shape)
        self.beta = np.dot(self.H_, T)
        print(self.beta.shape)
        return self.beta

    def regressor_test(self, test_x):
        b_row = test_x.shape[0]
        h = self.sigmoid(np.dot(test_x, self.w) + self.b[:b_row, :])
        result = np.dot(h, self.beta)
        return result

    def classifisor_test(self, test_x):
        b_row = test_x.shape[0]
        h = self.sigmoid(np.dot(test_x, self.w) + self.b[:b_row, :])
        result = np.dot(h, self.beta)
        result = [item.tolist().index(max(item.tolist())) for item in result]
        return result


stdsc = StandardScaler()
iris = load_iris()
x, y = stdsc.fit_transform(iris.data), iris.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

a = HiddenLayer(x_train, 20)
a.classifisor_train(y_train)
result = a.classifisor_test(x_test)

print(result)
print(metrics.accuracy_score(y_test, result))