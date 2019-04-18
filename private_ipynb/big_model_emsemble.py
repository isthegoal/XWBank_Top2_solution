# coding=gbk
import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas import Series
import time
import h5py
from private_ipynb.lgb import lgb_model
from private_ipynb.xgb import xgb_method
'''
������������ʹ�����·�ʽ��ģ�ͽ��м�Ȩ�ںϣ�
      ����Ԥ�����������������ʽ(ע��������Դ)��
           1.��ʼ����
           2.��ʼ����+��ȱʧ+ͳ������
           3.��ʼ����+��ȱʧ+ͳ������+��������
           4.��ʼ����+ɸѡ��ͳ������+ɸѡ����������(����)
      ���õĲ���ģ���ںϵ�ģ��(ע��ģ�������ϵĲ���)��
           lgb��xgb��cab
           
    �˴����̾��ܹ����ִ���֮�������Զ���Ҫ�ˣ�����Ҫ��ÿ��������̶��ܹ���ʱ����ȥ���������Ļ������ܼ��ٷǳ���Ĵ����ظ�����
    �����ں����������ȶ��Ը�ǿһЩ����Ϊ�ܹ�Ӧ�Զ���ͻ������� ��Ϊÿһ�ַ�������������������������ۺϷ������ø��ӹ���
'''
#######################################   ����ʽһ   ##################################
def do_process1():
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
    #########   ��������  ########
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
    ##########
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

    return y,X,test_data

#######################################   ����ʽ��   ##################################
def do_process2():
    # ��ȡ�ļ�
    train_xy = pd.read_csv("data/train_xy.csv",header=0,sep=",")
    test_all = pd.read_csv("data/test_all.csv",header=0,sep=",")

    print(train_xy.shape)
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
#######################################   ����ʽ��   ##################################

def do_process3():
    # ��ȡ�ļ�
    train_xy = pd.read_csv("data/train_xy.csv", header=0, sep=",")
    test_all = pd.read_csv("data/test_all.csv", header=0, sep=",")
    train = train_xy.copy()
    test = test_all.copy()
    test['y'] = -1
    # �ϲ�һ��train �� test    ������ͬ���ķ�ʽ���д�������ʡ��Щ
    data = pd.concat([train, test], axis=0)  # train_xy��test_all����������
    # ɾ��һЩ����Ҫ��������������ȱʧ���ء���ֵ���ظ��ȣ�     ɾ����ѵ�����Ͳ��Լ���ͬʱȱʧ���ڰٷ�֮95��
    # train ,test �ֿ�����
    # ����һ��ȱʧֵ���ص������У�ɾ��
    def get_nan_feature(train, rate=0.95):
        total_num = train.shape[0]
        train_nan_feats = []
        for i in range(157):
            feat = 'x_' + str(i + 1)
            nan_num = train.loc[train[feat] == -99, :].shape[0]
            nan_rate = nan_num / float(total_num)

            if nan_rate == 1.0:  # ֻ��nan
                train_nan_feats.append(feat)
            if nan_rate > rate:  # ��ȱʧֵ nan,����ȱʧ����
                if len(train[feat].unique()) == 2:  # ֻ��nan + һ������ֵ
                    train_nan_feats.append(feat)
        print("һ���� %d �������е�ȱʧֵ���أ�����%f " % (len(train_nan_feats), rate))
        return train_nan_feats

    train_nan_feats = get_nan_feature(train)
    test_nan_feats = get_nan_feature(test)
    print("ȱʧ���ص�������train =?= test------", np.all(train_nan_feats == test_nan_feats))

    # ����Щ����ȡ����:28��
    nan_feats = list(set(train_nan_feats) | set(test_nan_feats))  # ����train | test�Ľ��,������������A or B ��һ��     ������ʽ�ģ���ʵ���Զೢ��
    print('����ȱʧ�������� %d ����' % (len(nan_feats)))

    # �ܵ�ɾ��������������ɾ�ظ���5��������Ч�����ã�����ֻɾ��28��ȱʧ���ص�����,�����г��Թ��ã�
    drop_feats = nan_feats
    print('һ��ɾ���������� %d ����' % (len(drop_feats)))
    print(drop_feats)

    # ɾ��ȱʧֵ���ص�������
    train = train.drop(drop_feats, axis=1)
    test = test.drop(drop_feats, axis=1)
    data = data.drop(drop_feats, axis=1)
    print(data.shape)
    print(train.shape)
    print(test.shape)
    # ɾ����������ȫ����������Ҫ��=0������������ɾ�� = ԭʼ����������Ӱ�쾫ȷЧ�������ǿ��Լ���������
    # ɾ����x_92 , x_94 ����ֵ�͵ģ����� 24 �� ȫ���� �����
    # ��ʣ�µ��������з�������Ϊ��ֵ�� �� �����        (���������ʣ��ѵ�Ĭ��0-94����������ֵ����������ܲ�����)
    numerical_features = []
    categorical_features = []
    for i in range(157):
        feat = "x_" + str(i+1)
        if feat not in drop_feats:
            if i <= 94: # 1-95
                numerical_features.append(feat)
            else:
                categorical_features.append(feat)
    # ͳ��ÿ������ȱʧֵ�ĸ���                  ͳ��ȱʧ������
    def get_nan_count(data, feats, bins=7):
        df = data[feats].copy()
        df = df.replace(-99, np.nan)
        print('������:', df.shape[1])  # ����չʾ��ÿһ��ȱʧ������������
        print('ÿ�зǿ�����:', df.count(axis=1).values)  # ����չʾ��ÿһ��ȱʧ������������
        df['nan_count'] = df.shape[1] - df.count(axis=1).values  # ���� - ��nan��
        print('ÿ�п�����:', df['nan_count'])  # ����չʾ��ÿһ��ȱʧ������������
        dummy = pd.get_dummies(pd.cut(df['nan_count'], bins),
                               prefix='nan')  # ��ÿ�п���������7����ɢ����תone-hot���� ��ȱʧ���ݽ�����ɢ��,����Ϊ7������,���ڻ������䣬������ݿ�ֵ�������dummies����
        print(dummy.shape)
        res = pd.concat([data, dummy], axis=1)  # �ϲ���ԭ��������
        print(res.shape)
        return res

    # ��ȫ����������ͳ��ȱʧֵ      �¼����˶�ȱʧֵͳ�Ƶ�7��
    data = get_nan_count(data, data.columns.values, 7)
    print('ѵ�����������У�', data.columns)

    # ��ȡȱʧ���ٵ���ֵ�͵�����         ȱʧ�ٵ���ֵ���þ�ֵ
    def get_little_nan_feats(df, numerical_features, rate=0.1):
        total_num = df.shape[0]
        little_nan_feats = []
        for feat in numerical_features:
            nan_num = df.loc[df[feat] == -99, :].shape[0]
            nan_rate = nan_num / float(total_num)
            if nan_rate <= rate:
                little_nan_feats.append(feat)
                # print("feature:",feat,"nan_num = ",nan_num,"nan_rate = ",nan_rate)
        print("һ���� %d �������е�ȱʧֵ���٣�����%f " % (len(little_nan_feats), rate))
        return little_nan_feats

    little_nan_feats = get_little_nan_feats(data, numerical_features)
    # ����ֵ�͵�����������Ϊrank������³���Ժ�һ�㣩     ��ֵ����ʹ����С����˼�����ﹹ���������������й�һ����Ч�������³��һЩ��
    for feat in numerical_features:
        #print('rankǰ��',data[feat])
        data[feat] = data[feat].rank() / float(data.shape[0]) # ���򣬲��ҽ��й�һ��        ����Ҳ�У�
        #print('rank��',data[feat])
    print('ѵ�����������У�',data.columns)

    # ����Ҫ������������䣬֮�󻹻����Ŷ�����Ҫ��������������в�������
    imp_feat = ['x_80', 'x_2', 'x_81', 'x_95', 'x_1', 'x_52', 'x_63', 'x_54', 'x_43', 'x_40', 'x_93', 'x_42', 'x_157',
                'x_62', 'x_29', 'x_61', 'x_55']
    print('�ٶ�����Ҫ��������Ϊ��', len(imp_feat))
    for feat in imp_feat[:10]:
        if feat in numerical_features:
            print('��������')
            data[feat] = data[feat].replace(-99, np.nan)
            data[feat] = data[feat].fillna(data[feat].mean())
        if feat in categorical_features:
            print('�������������', feat)
    train = data.loc[data['y'] != -1, :]  # train set
    test = data.loc[data['y'] == -1, :]  # test set
    no_features = ['cust_id', 'cust_group', 'y']
    features = [feat for feat in train.columns.values if feat not in no_features]

    train_id = train.pop('cust_id')
    y = train['y'].values
    X = train[features].values
    print('X features :', features)
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    test_id = test.pop('cust_id')
    test_data = test[features].values
    return y, X, test_data

if __name__=='__main__':

    print('--------  �������ַ�ʽ�Ĵ�����ʽ  ---------')
    print('��һ����ʽ���ļ�����')
    y, X, test_data=do_process1()
    print('�ڶ�����ʽ���ļ�����')
    y2, X2, test_data2=do_process2()
    print('��������ʽ���ļ�����')
    y3, X3, test_data3 = do_process3()
    print('--------  ������ͬģ�͵���������  ---------')
    the_score_lg1,r_lg1=lgb_model(X, y, test_data)
    the_score_lg2,r_lg2=lgb_model(X2, y2, test_data2)
    the_score_lg3,r_lg3=lgb_model(X3, y3, test_data3)
    the_score_xg1,r_xg1=xgb_method(X, y, test_data)
    the_score_xg2,r_xg2=xgb_method(X2, y2, test_data2)
    the_score_xg3,r_xg3=xgb_method(X3, y3, test_data3)
    print('--------  �ںϴ���  ---------')
    print('���µĵ÷������ǣ�',the_score_lg1,'   ',the_score_lg2,'   ',the_score_lg3,'   ',the_score_xg1,'   ',the_score_xg2,'   ',the_score_xg3)

    the_record_score = (the_score_lg1 + the_score_lg2 + the_score_lg3 + the_score_xg1 + the_score_xg2 + the_score_xg3) / 6
    the_avg_sub=0*r_lg1+0*r_lg2+1*r_lg3+0*r_xg1+0*r_xg2+0*r_xg3


    filepath = '../result/�����ݼ���������ģ���ںϷ���' + str(the_record_score) + '.csv'  # ����ƽ������
    # תΪarray
    print('result shape:', the_avg_sub.shape)

    sub_sample = pd.read_csv('../result/xgb_nan.csv')
    result = DataFrame()
    result['cust_id'] = sub_sample['cust_id']
    result['pred_prob'] = the_avg_sub
    result.to_csv(filepath, index=False, sep=",")


'''

��6�ַ�ʽ�ںϵ�Ч����
    ��һ�֣�
        0.05*r_lg1+0.15*r_lg2+0.3*r_lg3+0.1*r_xg1+0.3*r_xg2+0.1*r_xg3
        mean auc: 0.8190709923066477
        �ܵĽ���� (5, 10000)
        result shape: (10000,)
        --------  �ںϴ���  ---------
        ���µĵ÷������ǣ� 0.8191922695131673     0.8206291690409669     0.8192571164144196     0.8194091860993773     0.819775776724755     0.8190709923066477
        result shape: (10000,)
        ���ϣ�0.75188
    �ڶ��֣�
        0.1*r_lg1+0.2*r_lg2+0.2*r_lg3+0.1*r_xg1+0.2*r_xg2+0.2*r_xg3
    
'''