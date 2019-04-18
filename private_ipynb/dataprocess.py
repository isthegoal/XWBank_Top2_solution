# coding=gbk
import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas import Series
import time
import h5py
def process_sesu():
    train_x = pd.read_csv("data/train_x.csv",header=0,sep=",")

    #���Ȼ�ȡ�ֿ�������
    numerical_features = []
    categorical_features = []
    for i in range(157):
        feat = "x_" + str(i + 1)
        if i <= 94:  # 1-95
            numerical_features.append(feat)
        else:
            categorical_features.append(feat)
    print("���õ���ֵ��������", len(numerical_features))
    print("���õ������������", len(categorical_features))

    #�ڶ���.��ȱʧֵ����ͳ��
    def get_nan_count(data):
        df = data.copy()
        df = df.replace(-99,np.nan)
        df['nan_count'] = df.shape[1] - df.count(axis = 1).values  # ���� - ��nan��
        dummy = pd.get_dummies(pd.cut(df['nan_count'],7),prefix = 'nan') # ��ȱʧ���ݽ�����ɢ��,����Ϊ7������
        print(dummy.shape)
        res = pd.concat([data,dummy],axis = 1) # �ϲ���ԭ��������
        print(res.shape)
        return res
    data = get_nan_count(train_x)
    # ������.��ȱʧֵ�������
    # ��Ҫ��top24
    imp_feat = ['x_80', 'x_2', 'x_81', 'x_95', 'x_1',
                'x_52', 'x_63', 'x_54', 'x_43', 'x_40',
                'x_93', 'x_42', 'x_157', 'x_62', 'x_29',
                'x_61', 'x_55', 'x_79', 'x_59', 'x_69',
                'x_48', 'x_56', 'x_7', 'x_64']
    print("��Ҫ������������", len(imp_feat))
    # ��һЩ��Ҫ������������䣬
    for feat in imp_feat[:10]:  # ���top 10 ,����������
        if feat in numerical_features:  # ��ֵ���þ�ֵ
            data[feat] = data[feat].replace(-99, np.nan)
            data[feat] = data[feat].fillna(data[feat].mean())  # ��nan��ֵ
        if feat in categorical_features:  # ����ͣ���������λ�� ������
            print("�������������", feat)
    pass

    #���Ĳ�.��֯������
    no_features = ['cust_id','cust_group','y']
    features = [feat for feat in data.columns.values if feat not in no_features]
    print("����������ά�ȣ�",len(features))
    test_data = data[features].values
    print('����ǲ�����������ɸѡ֮��Ľ����',data[features])

    #################  ���淽ʽΪ�˰�ලѧϰ����������������Ϣ  ##################
    no_features = ['cust_id','y']
    features = [feat for feat in data.columns.values if feat not in no_features]
    pd.DataFrame(data[features]).to_csv('../feature_data/suse_test_data.csv')
    return test_data
def process():
    # ��ȡ�ļ�
    train_xy = pd.read_csv("data/train_xy.csv",header=0,sep=",")
    train_x = pd.read_csv("data/train_x.csv",header=0,sep=",")
    test_all = pd.read_csv("data/test_all.csv",header=0,sep=",")

    print(train_xy.shape)
    print(train_x.shape)
    print(test_all.shape)

    train = train_xy.copy()
    test = test_all.copy()
    test['y'] = -1
    # �ϲ�һ��train �� test
    data = pd.concat([train,test],axis = 0) # train_xy��test_all����������
    print(train.shape)
    print(test.shape)
    print(data.shape)
    ################################################   ��������  #################################################################
    # ��ʣ�µ��������з�������Ϊ��ֵ�� �� �����
    numerical_features = []
    categorical_features = []
    for i in range(157):
        feat = "x_" + str(i+1)
        if i <= 94: # 1-95
            numerical_features.append(feat)
        else:
            categorical_features.append(feat)
    print("���õ���ֵ��������",len(numerical_features))
    print("���õ������������",len(categorical_features))

    #################################################  ȱʧֵͳ��  ################################################################
    # ͳ��ÿ���û�ȱʧֵ�ĸ���
    def get_nan_count(data):
        df = data.copy()
        df = df.replace(-99,np.nan)
        df['nan_count'] = df.shape[1] - df.count(axis = 1).values  # ���� - ��nan��
        dummy = pd.get_dummies(pd.cut(df['nan_count'],7),prefix = 'nan') # ��ȱʧ���ݽ�����ɢ��,����Ϊ7������
        print(dummy.shape)
        res = pd.concat([data,dummy],axis = 1) # �ϲ���ԭ��������
        print(res.shape)
        return res
    data = get_nan_count(data)
    #######################################################  ȱʧ���   ##########################################################
    # ��Ҫ��top24
    imp_feat = [ 'x_80', 'x_2', 'x_81', 'x_95', 'x_1',
                 'x_52', 'x_63', 'x_54', 'x_43', 'x_40',
                 'x_93', 'x_42', 'x_157', 'x_62', 'x_29',
                 'x_61', 'x_55', 'x_79', 'x_59', 'x_69',
                 'x_48', 'x_56', 'x_7', 'x_64']
    print("��Ҫ������������",len(imp_feat))
    # ��һЩ��Ҫ������������䣬
    for feat in imp_feat[:10]: # ���top 10 ,����������
        if feat in numerical_features:   # ��ֵ���þ�ֵ
            data[feat] = data[feat].replace(-99,np.nan)
            data[feat] = data[feat].fillna(data[feat].mean()) # ��nan��ֵ
        if feat in categorical_features: # ����ͣ���������λ�� ������
            print("�������������",feat)

    pass
    #################################################################################################################
    train = data.loc[data['y']!=-1,:] # train set
    test = data.loc[data['y']==-1,:]  # test set
    print(train.shape)
    print(test.shape)

    # ��ȡ�����У�ȥ��id��group, y
    no_features = ['cust_id','cust_group','y']
    features = [feat for feat in train.columns.values if feat not in no_features]
    print("����������ά�ȣ�",len(features))


    # �õ�����X �����y
    train_id = train['cust_id'].values
    y = train['y'].values
    X = train[features].values
    print("X shape:",X.shape)
    print("y shape:",y.shape)

    test_id = test['cust_id'].values
    test_data = test[features].values
    print("test shape",test_data.shape)

    #################  ���淽ʽΪ�˰�ලѧϰ����������������Ϣ  ##################
    train.to_csv('../feature_data/suse_all_train.csv')


    return y,X,test_data


########################################    �ڶ���Ԥ�������Ľ��   ###################################
def  process_for_sel():
    # ��ȡ�ļ�
    train_xy = pd.read_csv("data/train_xy.csv",header=0,sep=",")
    train_x = pd.read_csv("data/train_x.csv",header=0,sep=",")
    test_all = pd.read_csv("data/test_all.csv",header=0,sep=",")

    print(train_xy.shape)
    print(train_x.shape)
    print(test_all.shape)

    train = train_xy.copy()
    test = test_all.copy()
    test['y'] = -1
    # �ϲ�һ��train �� test
    data = pd.concat([train,test],axis = 0) # train_xy��test_all����������
    print(train.shape)
    print(test.shape)
    print(data.shape)
    ################################################   ��������  #################################################################
    # ��ʣ�µ��������з�������Ϊ��ֵ�� �� �����
    numerical_features = []
    categorical_features = []
    for i in range(157):
        feat = "x_" + str(i+1)
        if i <= 94: # 1-95
            numerical_features.append(feat)
        else:
            categorical_features.append(feat)
    print("���õ���ֵ��������",len(numerical_features))
    print("���õ������������",len(categorical_features))

    #################################################  ȱʧֵͳ��  ################################################################
    # ͳ��ÿ���û�ȱʧֵ�ĸ���
    def get_nan_count(data):
        df = data.copy()
        df = df.replace(-99,np.nan)
        df['nan_count'] = df.shape[1] - df.count(axis = 1).values  # ���� - ��nan��
        dummy = pd.get_dummies(pd.cut(df['nan_count'],7),prefix = 'nan') # ��ȱʧ���ݽ�����ɢ��,����Ϊ7������
        print(dummy.shape)
        res = pd.concat([data,dummy],axis = 1) # �ϲ���ԭ��������
        print(res.shape)
        return res
    data = get_nan_count(data)
    #######################################################  ȱʧ���   ##########################################################
    # ��Ҫ��top24
    imp_feat = [ 'x_80', 'x_2', 'x_81', 'x_95', 'x_1',
                 'x_52', 'x_63', 'x_54', 'x_43', 'x_40',
                 'x_93', 'x_42', 'x_157', 'x_62', 'x_29',
                 'x_61', 'x_55', 'x_79', 'x_59', 'x_69',
                 'x_48', 'x_56', 'x_7', 'x_64']
    print("��Ҫ������������",len(imp_feat))
    # ��һЩ��Ҫ������������䣬
    for feat in imp_feat[:10]: # ���top 10 ,����������
        if feat in numerical_features:   # ��ֵ���þ�ֵ
            data[feat] = data[feat].replace(-99,np.nan)
            data[feat] = data[feat].fillna(data[feat].mean()) # ��nan��ֵ
        if feat in categorical_features: # ����ͣ���������λ�� ������
            print("�������������",feat)
    #######################################################  ��������  ##########################################################
    # ����ֵ�͵�����������Ϊrank������³���Ժ�һ�㣩     ��ֵ����ʹ����С����˼�����ﹹ���������������й�һ����Ч�������³��һЩ��
    for feat in numerical_features:
        # print('rankǰ��',data[feat])
        data['rank_'+feat] = data[feat].rank() / float(data.shape[0])  # ���򣬲��ҽ��й�һ��        ����Ҳ�У�
        # print('rank��',data[feat])
    print('ѵ�����������У�', data.columns)

    #################################################################################################################
    train = data.loc[data['y']!=-1,:] # train set
    test = data.loc[data['y']==-1,:]  # test set
    print(train.shape)
    print(test.shape)

    # ��ȡ�����У�ȥ��id��group, y
    no_features = ['cust_id','cust_group','y']
    features = [feat for feat in train.columns.values if feat not in no_features]
    print("����������ά�ȣ�",len(features))


    # �õ�����X �����y
    train_id = train['cust_id'].values
    y = train['y'].values
    X = train[features].values
    print("X shape:",X.shape)
    print("y shape:",y.shape)

    test_id = test['cust_id'].values
    test_data = test[features].values
    print("test shape",test_data.shape)

    #�Եڶ���Ԥ�������Ľ�����б���
    train.to_csv('../feature_data/process_have_sort_train.csv')
    test.to_csv('../feature_data/process_have_sort_test.csv')
    pass

if __name__=='__main__':
    print('-------  ��һ��Ԥ����������ʵ������������ʺ�lgb,��û����������������  -------')
    y, X, test_data=process()
    sesu_pro_data=process_sesu()
    with h5py.File('do_pro_data/pro_data.hdf5','w') as f:
        f['y']=y
        f['X']=X
        f['test_data']=test_data
        f['sesu_pro_data'] = sesu_pro_data

    print('-------  Ϊ������ѡ�񷽰��е���������ɸѡ�����Ĵ���(�ڶ���Ԥ������������������ǰ������������)  -------')
    process_for_sel()