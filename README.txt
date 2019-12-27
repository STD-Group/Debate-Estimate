文件清单：
GBDT_LR.py	C.采用GBDT和GBDT+LR方式对feat.npy特征进行分类
LR_first_order.py	A.采用pytorch与梯度下降实现的LR模型
LR_second_order.py	A.采用pytorch与牛顿迭代法实现的LR模型
mfcc.py	B.提取各个音频文件的mfcc特征，采用k-means聚类输出
randomTest.py	A.使用随机数据检查是否有数据泄漏
sklearnLR.py	A.主要输出文件，使用sklearn实现LR模型并输出测试结果
torch.py	A.采用pytorch实现了多个方法，用于预调研比较


运行代码路径要求：
xxx\Debate-Estimate	git项目文件夹
xxx\dataset	数据文件夹