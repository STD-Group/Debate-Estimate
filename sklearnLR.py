import numpy as np

from sklearn import preprocessing
# GBDT
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
# LR
from sklearn.linear_model import LogisticRegression as LR
from sklearn.linear_model import LogisticRegressionCV as LRCV
# from sklearn.mode_selection import train_test_split
from sklearn.model_selection import KFold

# sklearn官网地址：
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

yNeg = np.zeros(num)
yPos = np.ones(num)
x = np.concatenate((xNeg, xPos), 0)
y = np.concatenate((yNeg, yPos), 0)
# 标准化向量，按列处理
scaler = preprocessing.StandardScaler().fit(x)
xscaled = scaler.transform(x)
# 正则化向量，按行处理
normalizer = preprocessing.Normalizer().fit(x)
xnormalized = normalizer.transform(x)
# xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.3)

times = 20
nSplit = 5
gbdtTrainAcc = 0
gbdtTestAcc = 0
lrTrainAcc = 0
lrTestAcc = 0

# 随机数测试
# x = np.random.random((200, 14))-0.5
# tmp = np.random.random((200, 1))
# for i in range(tmp.size):
#     if tmp[i] >= 0.5:
#         y[i] = 1

for t in range(times):
    kf = KFold(n_splits=nSplit, shuffle=True)
    for train_index, test_index in kf.split(xscaled):
        # print("TRAIN:", train_index, "TEST:", test_index)
        x_train, x_test = xscaled[train_index], xscaled[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # GBDT
        # n_estimators为最大子树个数
        # learning_rate为迭代步长，[0, 1]
        # subsample为子采样，可减少过拟合，[0.5, 0.8]
        # random_state为伪随机，便于复现
        numTrees = 50
        gbc = GradientBoostingClassifier(n_estimators=numTrees, learning_rate=0.01, subsample=0.7, random_state=0,\
                                         min_samples_leaf=1, max_depth=3, min_samples_split=2)
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

        # solver = liblinear, newton-cg, lbfgs, sag, saga
        lr = LR(solver='liblinear', penalty='l2', max_iter=1e5, tol=1e-5)
        lrcv = LRCV(solver='liblinear', penalty='l2', max_iter=1e5, tol=1e-5)

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


gbdtTrainAcc = gbdtTrainAcc/(nSplit*times)
gbdtTestAcc = gbdtTestAcc/(nSplit*times)
lrTrainAcc = lrTrainAcc/(nSplit*times)
lrTestAcc = lrTestAcc/(nSplit*times)

print('GBDT train acc is {:.4f}'.format(gbdtTrainAcc))  # 精度
print('GBDT test acc is {:.4f}'.format(gbdtTestAcc))  # 精度
print('LR train acc is {:.4f}'.format(lrTrainAcc))  # 精度
print('LR test acc is {:.4f}'.format(lrTestAcc))  # 精度

