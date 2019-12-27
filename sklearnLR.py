import numpy as np
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

# sklearn官网说明地址：
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

# feat.npy文件读取，结果为长度13的浮点list
num = 100
xLen = 13

xNeg = np.zeros((num, xLen+1))
for i in range(num):
    data = np.load("../dataset/train/negative/" + str(i) + "/feat.npy")
    for j in range(xLen):
        xNeg[i][j] = data[j]
    xNeg[i][xLen] = 1

xPos = np.zeros((num, xLen+1))
for i in range(num):
    data = np.load("../dataset/train/positive/" + str(i) + "/feat.npy")
    for j in range(xLen):
        xPos[i][j] = data[j]
    xPos[i][xLen] = 1

xTest = np.zeros((num, xLen+1))
for i in range(num):
    data = np.load("../dataset/test/" + str(i) + "/feat.npy")
    for j in range(xLen):
        xTest[i][j] = data[j]
    xTest[i][xLen] = 1

yNeg = np.zeros(num)
yPos = np.ones(num)
x = np.concatenate((xNeg, xPos), 0)
y = np.concatenate((yNeg, yPos), 0)

# 同序打乱
state = np.random.get_state()
np.random.shuffle(x)
np.random.set_state(state)
np.random.shuffle(y)

# # PolynomialFeatures生成多项式特征
# poly = PolynomialFeatures(degree=2, interaction_only=True)
# xPoly = poly.fit_transform(x)

# 标准化向量，按列处理
scaler = preprocessing.StandardScaler().fit(x)
xscaled = scaler.transform(x)
scaler = preprocessing.StandardScaler().fit(xTest)
xTestScaled = scaler.transform(xTest)
# # 正则化向量，按行处理
# normalizer = preprocessing.Normalizer().fit(x)
# xnormalized = normalizer.transform(x)


times = 1
nSplit = 5

# solver = liblinear, newton-cg, lbfgs, sag, saga
lr = LR(solver='newton-cg', penalty='l2', max_iter=1e5, tol=1e-5)
lrcv = LRCV(solver='newton-cg', penalty='l2', max_iter=1e5, tol=1e-5)

# # 利用learning_curve分析LR描述能力
# numSection = 10
# trainS = np.zeros(numSection)
# testS = np.zeros(numSection)
#
# for t in range(times):
#     state = np.random.get_state()
#     np.random.shuffle(xscaled)
#     np.random.set_state(state)
#     np.random.shuffle(y)
#
#     train_sizes, train_score, test_score = \
#         learning_curve(lrcv, xscaled, y, train_sizes=np.linspace(0.2, 1.0, numSection),\
#                        cv=nSplit, scoring='accuracy')
#     train_score = np.mean(train_score, axis=1)
#     test_score = np.mean(test_score, axis=1)
#     trainS += train_score
#     testS += test_score
# trainS = trainS/times
# testS = testS/times
# plt.plot(train_sizes, trainS, label='Train')
# plt.plot(train_sizes, testS, label='Test')
# plt.xlabel('Train examples')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()

# 输出最终测试集结果
yTest = np.zeros((xTestScaled.shape[0], times))
for t in range(times):
    state = np.random.get_state()
    np.random.shuffle(xscaled)
    np.random.set_state(state)
    np.random.shuffle(y)
    lrcv.fit(xscaled, y)
    yTest[:, t] = lrcv.predict(xTestScaled)

yTestMean = np.mean(yTest, axis=1)
Ans = np.zeros(yTestMean.size)
for t in range(yTestMean.size):
    if yTestMean[t] >= 0.5:
        Ans[t] = 1
    else:
        Ans[t] = 0

np.save("TestAns/LR.npy", Ans)

