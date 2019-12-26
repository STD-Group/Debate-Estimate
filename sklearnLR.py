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

Ans1 = np.load("TestAns/LR.npy")
Ans2 = np.load("TestAns/GBDT_LR.npy")
res = (Ans1 != Ans2).sum()

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

# # 随机数测试
# x = np.random.random((200, 14))-0.5
# tmp = np.random.random((200, 1))
# for i in range(tmp.size):
#     if tmp[i] >= 0.5:
#         y[i] = 1

# 同序打乱
state = np.random.get_state()
np.random.shuffle(x)
np.random.set_state(state)
np.random.shuffle(y)

# PolynomialFeatures
poly = PolynomialFeatures(degree=2, interaction_only=True)
xPoly = poly.fit_transform(x)

# 标准化向量，按列处理
scaler = preprocessing.StandardScaler().fit(x)
xscaled = scaler.transform(x)
scaler = preprocessing.StandardScaler().fit(xTest)
xTestScaled = scaler.transform(xTest)
# # 正则化向量，按行处理
# normalizer = preprocessing.Normalizer().fit(x)
# xnormalized = normalizer.transform(x)


times = 100
nSplit = 5
gbdtTrainAcc = 0
gbdtTestAcc = 0
lrTrainAcc = 0
lrTestAcc = 0

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

# GBDT测试部分
for t in range(times):
    state = np.random.get_state()
    np.random.shuffle(xscaled)
    np.random.set_state(state)
    np.random.shuffle(y)
    kf = KFold(n_splits=nSplit)
    for train_index, test_index in kf.split(xscaled):
        # print("TRAIN:", train_index, "TEST:", test_index)
        x_train, x_test = xscaled[train_index], xscaled[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # GBDT
        # n_estimators为最大子树个数
        # learning_rate为迭代步长，[0, 1]
        # subsample为子采样，可减少过拟合，[0.5, 0.8]
        # random_state为伪随机，便于复现
        numTrees = 100
        gbc = GradientBoostingClassifier(n_estimators=numTrees, learning_rate=0.01, subsample=0.7, random_state=0,\
                                         max_depth=5, min_samples_split=2)
        gbc.fit(x_train, y_train)
        yPredict = gbc.predict(x_train)
        acc = (yPredict == y_train).sum() / y_train.size
        # print('GBDT train acc is {:.4f}'.format(acc))  # 精度
        gbdtTrainAcc += acc

        yPredict = gbc.predict(x_test)
        acc = (yPredict == y_test).sum() / y_test.size
        # print('GBDT test acc is {:.4f}'.format(acc))  # 精度
        gbdtTestAcc += acc

        tmpTrain = gbc.apply(x_train)
        tmpTrain = tmpTrain.reshape(-1, numTrees)  # 得到200*20矩阵
        encoder = OneHotEncoder()
        xNewTrain = np.array(encoder.fit_transform(tmpTrain).toarray())

        tmpTest = gbc.apply(x_test)
        tmpTest = tmpTest.reshape(-1, numTrees)  # 得到200*20矩阵
        xNewTest = np.array(encoder.transform(tmpTest).toarray())

        realTest = gbc.apply(xTestScaled)
        realTest = realTest.reshape(-1, numTrees)
        xRealTest = np.array(encoder.transform(realTest).toarray())

        lrcv.fit(xNewTrain, y_train)
        coef = lrcv.coef_

        yPredict = lrcv.predict(xNewTrain)
        acc = (yPredict == y_train).sum() / y_train.size
        # print('LR train acc is {:.4f}'.format(acc))  # 精度
        lrTrainAcc += acc

        yPredict = lrcv.predict(xNewTest)
        acc = (yPredict == y_test).sum() / y_test.size
        # print('LR test acc is {:.4f}'.format(acc))  # 精度
        lrTestAcc += acc

        if t == 0:
            # 输出为文件
            Ans = lrcv.predict(xRealTest)
            np.save("TestAns/GBDT_LR.npy", Ans)

gbdtTrainAcc = gbdtTrainAcc/(nSplit*times)
gbdtTestAcc = gbdtTestAcc/(nSplit*times)
lrTrainAcc = lrTrainAcc/(nSplit*times)
lrTestAcc = lrTestAcc/(nSplit*times)

print('GBDT train acc is {:.4f}'.format(gbdtTrainAcc))  # 精度
print('GBDT test acc is {:.4f}'.format(gbdtTestAcc))  # 精度
print('LR train acc is {:.4f}'.format(lrTrainAcc))  # 精度
print('LR test acc is {:.4f}'.format(lrTestAcc))  # 精度

