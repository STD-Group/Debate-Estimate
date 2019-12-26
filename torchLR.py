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


# 1.pytorch自带优化；
def train_torch(xNeg, xPos):
    yNeg = torch.zeros(num, 1)
    yPos = torch.ones(num, 1)
    x = torch.cat((xNeg, xPos), 0).type(torch.FloatTensor)
    y = torch.cat((yNeg, yPos), 0).type(torch.FloatTensor)
    logistic_model = LogisticRegression()
    criterion = LR_Loss()

    # optimizer = torch.optim.SGD(logistic_model.parameters(), lr=1e-3, momentum=0.9)
    optimizer = torch.optim.Adam(logistic_model.parameters(), lr=1e-2, betas=(0.9, 0.99))
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

    return logistic_model, loss_all, accuracy

# 2.pytorch自动梯度下降；
def train_torchGD(xNeg, xPos):
    yNeg = torch.zeros(num, 1)
    yPos = torch.ones(num, 1)
    x = torch.cat((xNeg, xPos), 0).type(torch.FloatTensor)
    y = torch.cat((yNeg, yPos), 0).type(torch.FloatTensor)

    lr = 1e-3
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

    return w, loss_all, accuracy

# 3.pytorch手动梯度下降；
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

    return w, loss_all, accuracy

# 4.pytorch手动牛顿迭代；
def train_Newton(xNeg, xPos):
    yNeg = torch.zeros(num, 1)
    yPos = torch.ones(num, 1)
    x = torch.cat((xNeg, xPos), 0).type(torch.FloatTensor)
    y = torch.cat((yNeg, yPos), 0).type(torch.FloatTensor)

    lr = 1e-2
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

    return w, loss_all, accuracy

# 5.cascading LR手动梯度下降；
def train_CLR(xNeg, xPos):
    yNeg = torch.zeros(num, 1)
    yPos = torch.ones(num, 1)
    x = torch.cat((xNeg, xPos), 0).type(torch.FloatTensor)
    y = torch.cat((yNeg, yPos), 0).type(torch.FloatTensor)
    maxX = torch.max(x, 1)
    for i in range(x.size(1)):
        x[:, i] = x[:, i].div(maxX[0])

    numSig = 10
    lr = 1e-4
    lrSig = 1e-6
    lr_loss = LR_Loss()
    w = Variable((torch.rand(numSig, 1)-0.5).type(torch.FloatTensor), requires_grad=True)
    wSig = Variable((torch.rand(xLen + 1, numSig)-0.5).type(torch.FloatTensor), requires_grad=True)
    # w = Variable((torch.rand(numSig, 1)/100).type(torch.FloatTensor), requires_grad=True)
    # wSig = Variable((torch.rand(xLen + 1, numSig)/100).type(torch.FloatTensor), requires_grad=True)

    loss_all = []
    accuracy = []

    grad_w = torch.zeros(numSig, 1)
    grad_wSig = torch.zeros(xLen + 1, numSig)
    y_predSig = torch.zeros(x.size(0), numSig)

    lossH = 0

    for epoch in range(100000):
        wxSig = x.mm(wSig)
        relu = nn.ReLU(inplace=True)
        y_predSig = relu(wxSig)
        wx = y_predSig.mm(w)
        y_pred = 1 / (1 + torch.exp(-wx))

        loss = lr_loss.forward(wx, y)

        print_loss = loss.data.item()
        loss_all.append(print_loss)
        mask = y_pred.ge(0.5).float()
        correct = (mask == y).sum()
        acc = correct.item() / x.size(0)
        accuracy.append(acc)

        tmp = torch.exp(wx)
        tmp = torch.div(tmp, tmp + 1)

        for s in range(numSig):

            xtmp = y_predSig[:, s]
            grad_w[s] = -torch.sum(xtmp.mul(y[:, 0] - tmp[:, 0]))

            wtmp = w[s]
            grad_x = -wtmp*torch.sum(y[:, 0] - tmp[:, 0])

            gtmp = torch.zeros(xLen + 1)
            n = 0
            for i in range(x.size(0)):
                if wxSig[i, s] > 0:
                    gtmp += x[i, :]
                    n += 1

            grad_wSig[:, s] = gtmp/n*grad_x

        if lossH > loss.data:
            w.data -= lr*grad_w*min((lossH-loss.data)*5000, 1)
            wSig.data -= lrSig*grad_wSig*min((lossH-loss.data)*5000, 1)
        else:
            w.data -= lr * grad_w
            wSig.data -= lrSig * grad_wSig
        lossH = loss.data

        if (epoch + 1) % 100 == 0:
                print('*' * 10)
                print('epoch {}'.format(epoch + 1))  # 训练轮数
                print('loss is {:.4f}'.format(print_loss))  # 误差
                print('acc is {:.4f}'.format(acc))  # 精度

    return w


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

# 训练并存入文件
x = range(100000)
model1, l1, a1 = train_torch(xNeg, xPos)
Dict = {'l1': l1, 'a1': a1}
model2, l2, a2 = train_torchGD(xNeg, xPos)
Dict['l2'] = l2
Dict['a2'] = a2
model3, l3, a3 = train_GD(xNeg, xPos)
Dict['l3'] = l3
Dict['a3'] = a3
model4, l4, a4 = train_Newton(xNeg, xPos)
Dict['l4'] = l4
Dict['a4'] = a4

np.save("LR.npy", Dict)

# 读出数据并绘图
Dict = np.load("LR.npy", allow_pickle=True).item()
l1 = Dict['l1']
l2 = Dict['l2']
l3 = Dict['l3']
l4 = Dict['l4']

a1 = Dict['a1']
a2 = Dict['a2']
a3 = Dict['a3']
a4 = Dict['a4']

x = range(100000)
plt.title("Loss")
plt.semilogx()
plt.semilogy()
plt.plot(x, l1, label='Adam')
plt.plot(x, l2, label='TorchGD')
plt.plot(x, l3, label='GD')
plt.plot(x, l4, label='Newton')
plt.legend()
plt.xlabel('Iteration Times')
plt.ylabel('Loss')
plt.show()

plt.title("Accuracy")
plt.semilogx()
plt.plot(x, a1, label='Adam')
plt.plot(x, a2, label='TorchGD')
plt.plot(x, a3, label='GD')
plt.plot(x, a4, label='Newton')
plt.legend()
plt.xlabel('Iteration Times')
plt.ylabel('Accuracy')
plt.show()
