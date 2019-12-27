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

num = 1000
x = np.random.random((num, 14))-0.5
xTest = np.random.random((num, 14))-0.5
tmp = np.random.random(num)
tmpTest = np.random.random(num)
y = np.zeros(num)
yTest = np.zeros(num)
for i in range(tmp.size):
    if tmp[i] >= 0.5:
        y[i] = 1
    if tmpTest[i] >= 0.5:
        yTest[i] = 1

# 同序打乱
state = np.random.get_state()
np.random.shuffle(x)
np.random.set_state(state)
np.random.shuffle(y)

scaler = preprocessing.StandardScaler().fit(x)
xscaled = scaler.transform(x)
scaler = preprocessing.StandardScaler().fit(xTest)
xTestScaled = scaler.transform(xTest)

lr = LR(solver='newton-cg', penalty='l2', max_iter=1e5, tol=1e-5)
lrcv = LRCV(solver='newton-cg', penalty='l2', max_iter=1e5, tol=1e-5)

lrcv.fit(xscaled, y)
yPred = lrcv.predict(xscaled)
yTestPred = lrcv.predict(xTestScaled)

acc1 = (yPred == y).sum()/yPred.size
acc2 = (yTestPred == yTest).sum()/yTestPred.size

print('LR train acc is {:.4f}'.format(acc1))  # 精度
print('LR test acc is {:.4f}'.format(acc2))  # 精度