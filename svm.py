import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.model_selection import learning_curve

from sklearn.model_selection import KFold
# SVM
from sklearn.svm import SVC

# sklearn官网说明地址：
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

# audio.npy文件读取，结果为长度13的浮点list
num = 100
xLen = 240

xNeg = np.zeros((num, xLen+1))
for i in range(num):
    data = np.load("../dataset/train/negative/" + str(i) + "/audio.npy")
    for j in range(xLen):
        xNeg[i][j] = data[j]
    xNeg[i][xLen] = 1

xPos = np.zeros((num, xLen+1))
for i in range(num):
    data = np.load("../dataset/train/positive/" + str(i) + "/audio.npy")
    for j in range(xLen):
        xPos[i][j] = data[j]
    xPos[i][xLen] = 1

xTest = np.zeros((num, xLen+1))
for i in range(num):
    data = np.load("../dataset/test/" + str(i) + "/audio.npy")
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

xtmp = x[100: y.size]
x = x[0: 99]
ytmp = y[100: y.size]
y = y[0: 99]

# # PolynomialFeatures生成多项式特征
# poly = PolynomialFeatures(degree=2, interaction_only=True)
# xPoly = poly.fit_transform(x)

# 标准化向量，按列处理
scaler = preprocessing.StandardScaler().fit(x)
xscaled = scaler.transform(x)
scaler = preprocessing.StandardScaler().fit(xtmp)
xTmpScaled = scaler.transform(xtmp)
scaler = preprocessing.StandardScaler().fit(xTest)
xTestScaled = scaler.transform(xTest)

times = 10
nSplit = 3

# for t in range(times):
#     state = np.random.get_state()
#     np.random.shuffle(xscaled)
#     np.random.set_state(state)
#     np.random.shuffle(y)
#     kf = KFold(n_splits=nSplit)
#     for train_index, test_index in kf.split(xscaled):
#         # print("TRAIN:", train_index, "TEST:", test_index)
#         x_train, x_test = xscaled[train_index], xscaled[test_index]
#         y_train, y_test = y[train_index], y[test_index]
#
#         clf = SVC(kernel='rbf', class_weight='balanced')
#         clf.fit(x_train, y_train)
#         yPred = clf.predict(x_train)
#         acc = (yPred == y_train).sum() / y_train.size
#         print('SVM train acc is {:.4f}'.format(acc))  # 精度
#
#         yPred = clf.predict(x_test)
#         acc = (yPred == y_test).sum() / y_test.size
#         print('SVM test acc is {:.4f}'.format(acc))  # 精度

clf = SVC(kernel='rbf', class_weight='balanced')
clf.fit(xscaled, y)
yTmp = clf.predict(xTmpScaled)
acc = (yTmp == ytmp).sum() / yTmp.size
print('SVM test acc is {:.4f}'.format(acc))# 精度
Ans = clf.predict(xTestScaled)
np.save("TestAns/SVM.npy", Ans)
