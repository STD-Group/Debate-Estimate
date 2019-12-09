import numpy as np
import json
import torch

def LR(x):
    y = []
    return y


def train(xNeg, xPos):
    tmp = []
    w = []
    b = 0
    tmp.append(w)
    tmp.append(b)
    return tmp


def sigmoid(x):
    ans = 1 / (1 + np.exp(-x))
    return ans


def sigmoid_derivative(x):
    s = 1 / (1 + np.exp(-x))
    ans = s * (1-s)
    return ans


# # i_keypoints.json文件读取，结果为一个结构体，包含version版本，people各种标识点
# with open("../../dataset/train/negative/0/0_keypoints.json", 'r') as f:
#     keyPoints = json.load(f)
# print(keyPoints)

# feat.npy文件读取，结果为长度13的浮点list
num = 100
xLen = 13

xNeg = np.zeros((num, xLen))
for i in range(num):
    data = np.load("../../dataset/train/negative/" + str(i) + "/feat.npy")
    for j in range(xLen):
        xNeg[i][j] = data[j]

xPos = np.zeros((num, xLen))
for i in range(num):
    data = np.load("../../dataset/train/positive/" + str(i) + "/feat.npy")
    for j in range(xLen):
        xPos[i][j] = data[j]

tmp = train(xNeg, xPos)
w = tmp[0]
b = tmp[1]

for i in range(num):
    data = np.load("../../dataset/test/" + str(i) + "/feat.npy")
    x = np.array(data)
    y = sigmoid(w*x+b)
