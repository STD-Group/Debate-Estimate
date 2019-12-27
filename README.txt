文件清单：
设计报告.pdf	实验报告

sklearnLR.py为A部分主要文件
SVM.py为B部分主要文件
GBDT_MFCC.py为C部分主要文件

Ans\A.npy	A.A部分主要输出结果
Ans\B.npy	A.B部分主要输出结果
Ans\C.npy	A.C部分主要输出结果

具体内容：
TestAns\GBDT	C.GBDT_LR.py文件GBDT方法输出结果
TestAns\GBDT_LR	C.GBDT_LR.py文件GBDT+LR方法输出结果
TestAns\GBDT_LR_MFCC	C.GBDT_MFCC.py文件GBDT+LR方法输出结果
TestAns\GBDT_MFCC	C.GBDT_MFCC.py文件GBDT方法输出结果(C部分主要输出结果）
TestAns\LR.npy	A.sklearnLR.py文件LR方法输出结果(A部分主要输出结果）
TestAns\SVM.npy	B.SVM.py文件MFCC+SVM方法输出结果(B部分主要输出结果）

B_feat.npy	B.B部分测试集MFCC特征
B_train_feat.npy	B.B部分训练集MFCC特征
C_feat.npy	C.C部分测试集MFCC特征
C_train_feat.npy	C.C部分训练集MFCC特征

emotions.py	C.尝试提取表情特征辅助LR分类
GBDT_LR.py	C.采用GBDT和GBDT+LR方式对feat.npy特征进行分类
GBDT_MFCC.py	C.采用GBDT和GBDT+LR方式对C_train_feat.npy特征进行分类
LR_first_order.py	A.采用pytorch与梯度下降实现的LR模型
LR_second_order.py	A.采用pytorch与牛顿迭代法实现的LR模型
neuralNetwork.py	C.采用两层神经网络对C_train_feat.npy特征进行分类
randomTest.py	A.使用随机数据检查是否有数据泄漏
sklearnLR.py	A.主要输出文件，使用sklearn实现LR模型并输出测试结果
SVM.py	B.利用SVM对B_train_feat.npy特征进行分类
svm_source.py	B/C.提取各个音频文件的mfcc特征，采用k-means聚类输出
torchLR.py	A.采用pytorch实现了多个方法，用于预调研比较


运行代码路径要求：
xxx\视听导第三次大作业（提交文件夹）
xxx\dataset	数据文件夹

运行代码库包要求：
所有上述主要文件只需要sklearn，librosa和python基本库包即可
其他部分文件需要torch（版本1.3.1）