import numpy as np
import json
import torch
from torch import nn
import torch.nn.functional as F
from sklearn import preprocessing
from torch.autograd import Variable
from sklearn.model_selection import KFold

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        layer = 10
        self.layer1 = torch.nn.Linear(xLen+1, layer)
        self.layer2 = torch.nn.Linear(layer, 1)
        # self.sm = torch.nn.Sigmoid()
        # self.layer1 = torch.nn.Linear(xLen+1, 1, bias=False, nonlinearity='tanh')
        self.sm = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.sm(x)
        # return x
        # x = self.layer1(x)
        # x = self.sm(x)
        return x


def train(x, y):

    model = Model()
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    if torch.cuda.is_available():
        model.cuda()
    loss_all = []
    accuracy = []
    for epoch in range(1000):
        if torch.cuda.is_available():
            x_data = Variable(x).cuda()
            y_data = Variable(y).cuda()
        else:
            x_data = Variable(x)
            y_data = Variable(y)

        y_pred = model(x_data)
        loss = criterion(y_pred,y_data)
        print_loss = loss.data.item()
        loss_all.append(print_loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for i in range(len(y_pred)):
            if y_pred[i] >= 0.5:
                y_pred[i] = 1.0
            else:
                y_pred[i] = 0.0

        correct = (y_pred == y).sum()
        acc = correct.item() / x.size(0)
        accuracy.append(acc)

        if (epoch + 1) % 1000 == 0:
            print('*' * 10)
            print('epoch {}'.format(epoch + 1))  # 训练轮数
            print('loss is {:.4f}'.format(print_loss))  # 误差
            print('acc is {:.4f}'.format(acc))  # 精度

    torch.save(model, 'model.pkl')
    return model


num = 100
xLen = 120
x = np.load("C_train_feat.npy")
xTest = np.load("C_feat.npy")

yNeg = torch.zeros(num, 1)
yPos = torch.ones(num, 1)
y = torch.cat((yNeg, yPos), 0).type(torch.FloatTensor)

scaler = preprocessing.StandardScaler().fit(x)
xscaled = torch.from_numpy(scaler.transform(x)).float()

kf = KFold(n_splits=5, shuffle=True)
for train_index, test_index in kf.split(xscaled):
    # print("TRAIN:", train_index, "TEST:", test_index)
    x_train, x_test = xscaled[train_index], xscaled[test_index]
    y_train, y_test = y[train_index], y[test_index]

    train(x_train, y_train)
    model = torch.load('model.pkl')

    out = model(x_test)
    mask = out.ge(0.5).float()
    correct = (mask == y_test).sum()
    accuracy = correct.item() / y_test.size(0)
    print("accuracy = {:.4f}".format(accuracy))

# accuracy = 0
# for i in range(num):
#     data = np.load("../dataset/test/" + str(i) + "/audio.npy")
#     out = model(data)





