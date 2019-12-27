import numpy as np
import json
import torch
from torch import nn
from torch.autograd import Variable
import matplotlib.pyplot as plt

class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.lr = nn.Linear(xLen+1, 1)
        self.sm = nn.Sigmoid()

    def forward(self, x):
        x = self.lr(x)
        x = self.sm(x)
        return x


class LR_Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y):
        loss = -torch.sum(y*y_pred-torch.log(torch.exp(y_pred)+1))
        return loss


# 3.pytorch手动梯度下降；
def LR(X):
    num = 100

    xLen = 13

    # 载入数据
    xNeg = torch.zeros((num, xLen+1))
    for i in range(num):
        data = np.load("../dataset/train/negative/" + str(i) + "/feat.npy")
        for j in range(xLen):
            xNeg[i][j] = data[j]
        xNeg[i][xLen] = 1

    xPos = torch.zeros((num, xLen+1))
    for i in range(num):
        data = np.load("../dataset/train/positive/" + str(i) + "/feat.npy")
        for j in range(xLen):
            xPos[i][j] = data[j]
        xPos[i][xLen] = 1


    yNeg = torch.zeros(num, 1)
    yPos = torch.ones(num, 1)
    x = torch.cat((xNeg, xPos), 0).type(torch.FloatTensor)
    y = torch.cat((yNeg, yPos), 0).type(torch.FloatTensor)

    lr = 1e-4
    lr_loss = LR_Loss()
    w = Variable((torch.rand(xLen+1, 1)*10-5).type(torch.FloatTensor), requires_grad=True)
    loss_all = []
    accuracy = []

    for epoch in range(100000):
        wx = x.mm(w)
        y_pred = 1 / (1 + torch.exp(-wx))
        loss = lr_loss.forward(wx, y)
        print_loss = loss.data.item()
        loss_all.append(print_loss)
        mask = y_pred.ge(0.5).float()
        correct = (mask == y).sum()
        acc = correct.item() / x.size(0)
        accuracy.append(acc)

        grad_w = torch.zeros(xLen+1, 1)
        for i in range(xLen+1):
            w_x = torch.exp(wx)
            tmp = torch.div(w_x, w_x+1)
            xtmp = x[:, i]
            grad_w[i] = -torch.sum(xtmp.mul(y[:, 0]-tmp[:, 0]))

        w.data -= lr*grad_w
        if (epoch + 1) % 1000 == 0:
                print('*' * 10)
                print('epoch {}'.format(epoch + 1))  # 训练轮数
                print('loss is {:.4f}'.format(print_loss))  # 误差
                print('acc is {:.4f}'.format(acc))  # 精度
    wx = X.mm(w)
    y_pred = 1 / (1 + torch.exp(-wx))
    Y = y_pred.ge(0.5).float()
    return Y

num = 100

xLen = 13

# 载入数据
X = torch.zeros((num, xLen+1))
for i in range(num):
    data = np.load("../dataset/test/" + str(i) + "/feat.npy")
    for j in range(xLen):
        X[i][j] = data[j]
    X[i][xLen] = 1

Y = LR(X)
