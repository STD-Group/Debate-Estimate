import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import cross_val_score


def SVM(x):
    # 测试集的读取
    n_mf = 20
    n_k = 20
    x = x.reshape(-1, 20)
    xx = np.load("B_train_feat.npy")
    xx = xx.reshape(4000,20)
    yy = np.zeros(shape=200 * n_k)
    yy[100 * n_k:] = np.ones(shape=100 * n_k)

    # 将测试集的顺序随机打乱
    ll = [i for i in range(0, 200 * n_k)]
    random.shuffle(ll)
    xx = xx[ll]
    yy = yy[ll]

    # 高斯核SVM模型(目前效果比较好）
    rbf_kernel_svm_clf = Pipeline((
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="rbf", gamma=0.25, C=1))  # 0.24
    ))
    rbf_kernel_svm_clf.fit(xx, yy)
    y= rbf_kernel_svm_clf.predict(x)
    y = y.reshape(100, -1)
    y = np.sum(y, axis=1)
    test = np.zeros(100)
    test[y > n_k / 2] = 1
    return test
    # 画学习曲线
    # train_sizes, train_score, test_score = learning_curve(
    #     rbf_kernel_svm_clf, xx, yy, cv=10, scoring='accuracy',
    #     train_sizes=[0.1, 0.2, 0.4, 0.6, 0.8, 1]  # [0.1, 0.25, 0.5, 0.75, 1]
    # )
    # train_error = 1 - np.mean(train_score, axis=1)
    # test_error = 1 - np.mean(test_score, axis=1)
    # plt.plot(train_sizes, train_error, 'o-', color='r', label='training')
    # plt.plot(train_sizes, test_error, 'o-', color='g', label='testing')
    # plt.legend(loc='best')
    # plt.xlabel('traing examples')
    # plt.ylabel('error')
    # plt.show()


# 读取特征
xTest = np.load("B_feat.npy")
yTest = SVM(xTest)
np.save("TestAns/SVM.npy", yTest)
