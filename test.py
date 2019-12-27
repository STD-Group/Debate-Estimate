import pandas as pd
import numpy as np

df = pd.read_excel('manual.xlsx')
data1 = df.Human

ansLR = np.load("TestAns/LR.npy")
ansGBDT = np.load("TestAns/GBDT.npy")
ansGBDT_LR = np.load("TestAns/GBDT_LR.npy")
ansSVM = np.load("TestAns/SVM.npy")
ansGBDT_MFCC = np.load("TestAns/GBDT_MFCC.npy")
ansGBDT_LR_MFCC = np.load("TestAns/GBDT_LR_MFCC.npy")

tmp = np.zeros((data1.size, 4))
tmp[:, 0] = data1
tmp[:, 1] = ansSVM
tmp[:, 2] = ansGBDT_MFCC
tmp[:, 3] = ansGBDT_LR_MFCC

for i in range(4):
    for j in range(i+1, 4):
        ans1 = tmp[:, i]
        ans2 = tmp[:, j]
        res = (ans1 != ans2).sum()
        print("i = {:d}, j = {:d}, res = {:d}".format(i, j, res))
