import numpy as np
import json
import torch
from torch import nn
from torch.autograd import Variable


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


# 1.pytorch自带优化；
def train_torch(xNeg, xPos):
    yNeg = torch.zeros(num, 1)
    yPos = torch.ones(num, 1)
    x = torch.cat((xNeg, xPos), 0).type(torch.FloatTensor)
    y = torch.cat((yNeg, yPos), 0).type(torch.FloatTensor)
    logistic_model = LogisticRegression()
    criterion = LR_Loss()

    optimizer = torch.optim.SGD(logistic_model.parameters(), lr=1e-3, momentum=0.9)
    # optimizer = torch.optim.Adam(logistic_model.parameters(), lr=1e-2, betas=(0.9, 0.99))
    if torch.cuda.is_available():
        logistic_model.cuda()
    loss_all = []
    accuracy = []

    for epoch in range(100000):
        if torch.cuda.is_available():
            x_data = Variable(x).cuda()
            y_data = Variable(y).cuda()
        else:
            x_data = Variable(x)
            y_data = Variable(y)

        out = logistic_model(x_data)
        loss = criterion(out, y_data)
        print_loss = loss.data.item()
        loss_all.append(print_loss)

        mask = out.ge(0.5).float()
        correct = (mask == y_data).sum()
        acc = correct.item() / x_data.size(0)
        accuracy.append(acc)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (epoch + 1) % 1000 == 0:
            print('*' * 10)
            print('epoch {}'.format(epoch + 1))  # 训练轮数
            print('loss is {:.4f}'.format(print_loss))  # 误差
            print('acc is {:.4f}'.format(acc))  # 精度

        print('*' * 10)
        print('epoch {}'.format(epoch + 1))  # 训练轮数
        print('loss is {:.4f}'.format(print_loss))  # 误差
        print('acc is {:.4f}'.format(acc))  # 精度


    return logistic_model


# 2.pytorch自动梯度下降；
def train_torchGD(xNeg, xPos):
    yNeg = torch.zeros(num, 1)
    yPos = torch.ones(num, 1)
    x = torch.cat((xNeg, xPos), 0).type(torch.FloatTensor)
    y = torch.cat((yNeg, yPos), 0).type(torch.FloatTensor)

    lr = 1e-4
    lr_loss = LR_Loss()
    w = Variable(torch.zeros(xLen+1, 1).type(torch.FloatTensor), requires_grad=True)
    loss_all = []
    accuracy = []
    for epoch in range(100000):
        y_pred = x.mm(w)
        loss = lr_loss.forward(y_pred, y)
        print_loss = loss.data.item()
        loss_all.append(print_loss)
        mask = y_pred.ge(0.5).float()
        correct = (mask == y).sum()
        acc = correct.item() / x.size(0)
        accuracy.append(acc)

        loss.backward()
        w.data -= lr*w.grad.data
        w.grad.data.zero_()
        if (epoch + 1) % 1000 == 0:
                print('*' * 10)
                print('epoch {}'.format(epoch + 1))  # 训练轮数
                print('loss is {:.4f}'.format(print_loss))  # 误差
                print('acc is {:.4f}'.format(acc))  # 精度

    return w



# 3.pytorch手动梯度下降
def train_GD(xNeg, xPos):
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

        loss.backward()
        grad_w = torch.zeros(xLen+1, 1)
        grad_grad_w = torch.zeros(xLen + 1, 1)
        for i in range(xLen+1):
            w_x = torch.exp(wx)
            tmp = torch.div(w_x, w_x+1)
            xtmp = x[:, i]
            xsqrt = xtmp.mul(xtmp)
            tmp2 = torch.div(tmp, w_x+1)
            grad_w[i] = -torch.sum(xtmp.mul(y[:, 0])-xtmp.mul(tmp[:, 0]))
            grad_grad_w[i] = torch.sum(xsqrt.mul(tmp2[:, 0]))

        grad = torch.div(grad_w, grad_grad_w)
        w.data -= lr*grad_w/grad
        w.grad.data.zero_()
        if (epoch + 1) % 1000 == 0:
                print('*' * 10)
                print('epoch {}'.format(epoch + 1))  # 训练轮数
                print('loss is {:.4f}'.format(print_loss))  # 误差
                print('acc is {:.4f}'.format(acc))  # 精度

    return w

# 4.pytorch手动牛顿迭代
def train_Newton(xNeg, xPos):
    yNeg = torch.zeros(num, 1)
    yPos = torch.ones(num, 1)
    x = torch.cat((xNeg, xPos), 0).type(torch.FloatTensor)
    y = torch.cat((yNeg, yPos), 0).type(torch.FloatTensor)

    lr = 1e-3
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

        loss.backward()
        grad_w = torch.zeros(xLen+1, 1)
        grad_grad_w = torch.zeros(xLen + 1, 1)
        for i in range(xLen+1):
            w_x = torch.exp(wx)
            tmp = torch.div(w_x, w_x+1)
            xtmp = x[:, i]
            xsqrt = xtmp.mul(xtmp)
            tmp2 = torch.div(tmp, w_x+1)
            grad_w[i] = torch.sum(xtmp.mul(y[:, 0])-xtmp.mul(tmp[:, 0]))
            grad_grad_w[i] = -torch.sum(xsqrt.mul(tmp2[:, 0]))

        grad = torch.div(grad_w, grad_grad_w)
        w.data -= lr*grad
        w.grad.data.zero_()
        if (epoch + 1) % 1000 == 0:
                print('*' * 10)
                print('epoch {}'.format(epoch + 1))  # 训练轮数
                print('loss is {:.4f}'.format(print_loss))  # 误差
                print('acc is {:.4f}'.format(acc))  # 精度

    return w


# def sigmoid(x):
#     ans = 1 / (1 + np.exp(-x))
#     return ans
#
#
# def sigmoid_derivative(x):
#     s = 1 / (1 + np.exp(-x))
#     ans = s * (1-s)
#     return ans


# # i_keypoints.json文件读取，结果为一个结构体，包含version版本，people各种标识点
# with open("../../dataset/train/negative/0/0_keypoints.json", 'r') as f:
#     keyPoints = json.load(f)
# print(keyPoints)

# feat.npy文件读取，结果为长度13的浮点list
num = 100

# xLen = 13
#
# xNeg = torch.zeros((num, xLen+1))
# for i in range(num):
#     data = np.load("../../dataset/train/negative/" + str(i) + "/feat.npy")
#     for j in range(xLen):
#         xNeg[i][j] = data[j]
#     xNeg[i][xLen] = 1
#
# xPos = torch.zeros((num, xLen+1))
# for i in range(num):
#     data = np.load("../../dataset/train/positive/" + str(i) + "/feat.npy")
#     for j in range(xLen):
#         xPos[i][j] = data[j]
#     xPos[i][xLen] = 1
#
# train_torch(xNeg, xPos)

xLen = 348

xNeg = torch.zeros((num, xLen+1))
for i in range(num):
    data = np.load("../../dataset/train/negative/" + str(i) + "/audio.npy")
    for j in range(xLen):
        xNeg[i][j] = data[j, 0]
    xNeg[i][xLen] = 1

xPos = torch.zeros((num, xLen+1))
for i in range(num):
    data = np.load("../../dataset/train/positive/" + str(i) + "/audio.npy")
    for j in range(xLen):
        xPos[i][j] = data[j, 0]
    xPos[i][xLen] = 1

train_torch(xNeg, xPos)


for i in range(num):
    data = np.load("../../dataset/test/" + str(i) + "/feat.npy")