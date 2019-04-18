# coding=gbk
'''

������Ҫ����������������⣬��ʽ�����У�
     1.������
     2.�Ľ���ʽ�Ĺ�����
     3.�������з�ʽ
     4.Ƿ����
'''
import pandas as pd
from imblearn.over_sampling import SMOTE       #���ȳ��������SMOTE
from imblearn.under_sampling import RandomUnderSampler
def do_guocaiyang():
    print('��ʼ��,��һ�ַ�ʽ(��ѡ��ѡ�����ľ�������������Ҫ���ļ��������� ����������������ݻ��𵽸��õĲ�������Ч�������Զ������԰� )')
    x= pd.read_csv('../feature_data/train_feature.csv')[1:]
    feature_columns=[i for i in x.columns]
    y = pd.read_csv('../feature_data/train_label.csv',header=None)[1]
    #groupby_data_orginal = y.groupby('label').count()
    print(x)
    print(y)
    model_smote = SMOTE()  # ����smoteģ�Ͷ���
    x_smote_resampled, y_smote_resampled = model_smote.fit_sample(x, y)
    x_smote_resampled = pd.DataFrame(x_smote_resampled, columns=feature_columns)
    y_smote_resampled = pd.DataFrame(y_smote_resampled, columns=['label'])
    smote_resampled = pd.concat([x_smote_resampled, y_smote_resampled], axis=1)
    groupby_data_smote = smote_resampled.groupby('label').count()


def do_qiancaiyang():
    x= pd.read_csv('../feature_data/train_feature.csv')[1:]
    feature_columns=[i for i in x.columns]
    y = pd.read_csv('../feature_data/train_label.csv',header=None)[1]
    model_RandomUnderSampler=RandomUnderSampler()  #����RandomUnderSampleģ�Ͷ���
    x_RandomUnderSample_resampled,y_RandomUnderSample_resampled=model_RandomUnderSampler.fit_sample(x,y)#�������ݲ�����Ƿ��������
    x_RandomUnderSample_resampled=pd.DataFrame(x_RandomUnderSample_resampled,columns=feature_columns)
    y_RandomUnderSample_resampled=pd.DataFrame(y_RandomUnderSample_resampled,columns=['label'])
    RandomUnderSampler_resampled=pd.concat([x_RandomUnderSample_resampled,y_RandomUnderSample_resampled],axis=1)
    groupby_data_RandomUnderSampler=RandomUnderSampler_resampled.groupby('label').count()

if __name__=='__main__':

    print('###################    ��ʽһ.SMOTE��ʽ������    #################')
    do_guocaiyang()
    print('###################    ��ʽ��.Ƿ��������    #################')
    do_qiancaiyang()
    print('###################    ��ʽ��.�������з���    #################')
    #���ַ����Ͳ�����д�ˣ���Ϊ�����ַ�ʽ�ĵ��ڿ����ڲ���is_unbalance ����ΪTrue,�����Ϳ����𵽶Բ��������⴦���Ч����