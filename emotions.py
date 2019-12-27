import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.model_selection import learning_curve
# GBDT
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
# PolynomialFeatures
from sklearn.preprocessing import PolynomialFeatures
# LR
from sklearn.linear_model import LogisticRegression as LR
from sklearn.linear_model import LogisticRegressionCV as LRCV
# from sklearn.mode_selection import train_test_split
from sklearn.model_selection import KFold

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

num = 100
frames = 30
times = 10

numEmotion = np.zeros(num*2)
x = np.zeros((num*2, 7))
for i in range(num):
    for j in range(frames):
        fPath = "../dataset/train/positive/" + str(i) + "/" + str(j) + "_emotions.npy"
        if os.path.exists(fPath):
            data = np.load(fPath)
            x[i, :] += data
            numEmotion[i] += 1
    x[i, :] = x[i, :]/numEmotion[i]

for i in range(num):
    for j in range(frames):
        fPath = "../dataset/train/negative/" + str(i) + "/" + str(j) + "_emotions.npy"
        if os.path.exists(fPath):
            data = np.load(fPath)
            x[i+num, :] += data
            numEmotion[i+num] += 1
    x[i+num, :] = x[i+num, :] / numEmotion[i+num]

yNeg = np.zeros(num)
yPos = np.ones(num)
y = np.concatenate((yNeg, yPos), 0)

# PolynomialFeatures生成多项式特征
poly = PolynomialFeatures(degree=3, interaction_only=True)
xPoly = poly.fit_transform(x)

state = np.random.get_state()
np.random.shuffle(xPoly)
np.random.set_state(state)
np.random.shuffle(y)

scaler = preprocessing.StandardScaler().fit(xPoly)
xscaled = scaler.transform(xPoly)

lr = LR(solver='newton-cg', penalty='l2', max_iter=1e5, tol=1e-5)
lrcv = LRCV(solver='newton-cg', penalty='l2', max_iter=1e5, tol=1e-5)

nSplit = 5

kf = KFold(n_splits=nSplit)
for train_index, test_index in kf.split(xscaled):
    x_train, x_test = xscaled[train_index], xscaled[test_index]
    y_train, y_test = y[train_index], y[test_index]

    lrcv.fit(x_train, y_train)
    coef = lrcv.coef_

    yPredict = lrcv.predict(x_train)
    acc = (yPredict == y_train).sum() / y_train.size
    print('LR train acc is {:.4f}'.format(acc))  # 精度

    yPredict = lrcv.predict(x_test)
    acc = (yPredict == y_test).sum() / y_test.size
    print('LR test acc is {:.4f}'.format(acc))  # 精度

    rbf_kernel_svm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="rbf", gamma=0.015, C=0.001))
    ])
    rbf_kernel_svm_clf.fit(x_train, y_train)

    yPredict = rbf_kernel_svm_clf.predict(x_train)
    acc = (yPredict == y_train).sum() / y_train.size
    print('SVM train acc is {:.4f}'.format(acc))  # 精度

    yPredict = rbf_kernel_svm_clf.predict(x_test)
    acc = (yPredict == y_test).sum() / y_test.size
    print('SVM test acc is {:.4f}'.format(acc))  # 精度

