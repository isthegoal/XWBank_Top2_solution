# coding=gbk
'''

这里想要解决样本不均衡问题，方式尝试有：
     1.过采样
     2.改进形式的过采样
     3.代价敏感方式
     4.欠采样
'''
import pandas as pd
from imblearn.over_sampling import SMOTE       #过度抽样处理库SMOTE
from imblearn.under_sampling import RandomUnderSampler
def do_guocaiyang():
    print('开始了,第一种方式(可选择：选择计算的距离依据是最重要的哪几个特征？ 往往更多的特征依据会起到更好的采样生成效果，可以多做尝试吧 )')
    x= pd.read_csv('../feature_data/train_feature.csv')[1:]
    feature_columns=[i for i in x.columns]
    y = pd.read_csv('../feature_data/train_label.csv',header=None)[1]
    #groupby_data_orginal = y.groupby('label').count()
    print(x)
    print(y)
    model_smote = SMOTE()  # 建立smote模型对象
    x_smote_resampled, y_smote_resampled = model_smote.fit_sample(x, y)
    x_smote_resampled = pd.DataFrame(x_smote_resampled, columns=feature_columns)
    y_smote_resampled = pd.DataFrame(y_smote_resampled, columns=['label'])
    smote_resampled = pd.concat([x_smote_resampled, y_smote_resampled], axis=1)
    groupby_data_smote = smote_resampled.groupby('label').count()


def do_qiancaiyang():
    x= pd.read_csv('../feature_data/train_feature.csv')[1:]
    feature_columns=[i for i in x.columns]
    y = pd.read_csv('../feature_data/train_label.csv',header=None)[1]
    model_RandomUnderSampler=RandomUnderSampler()  #建立RandomUnderSample模型对象
    x_RandomUnderSample_resampled,y_RandomUnderSample_resampled=model_RandomUnderSampler.fit_sample(x,y)#输入数据并进行欠抽样处理
    x_RandomUnderSample_resampled=pd.DataFrame(x_RandomUnderSample_resampled,columns=feature_columns)
    y_RandomUnderSample_resampled=pd.DataFrame(y_RandomUnderSample_resampled,columns=['label'])
    RandomUnderSampler_resampled=pd.concat([x_RandomUnderSample_resampled,y_RandomUnderSample_resampled],axis=1)
    groupby_data_RandomUnderSampler=RandomUnderSampler_resampled.groupby('label').count()

if __name__=='__main__':

    print('###################    方式一.SMOTE方式过采样    #################')
    do_guocaiyang()
    print('###################    方式二.欠采样方案    #################')
    do_qiancaiyang()
    print('###################    方式三.代价敏感方案    #################')
    #这种方案就不单独写了，因为对这种方式的调节可以在参数is_unbalance 设置为True,这样就可以起到对不均衡问题处理的效果。