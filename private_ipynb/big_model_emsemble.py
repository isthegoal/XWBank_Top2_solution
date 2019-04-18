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
初步方案打算使用以下方式的模型进行加权融合：
      几种预处理和特征构建的形式(注重数据来源)：
           1.初始特征
           2.初始特征+补缺失+统计特征
           3.初始特征+补缺失+统计特征+排序特征
           4.初始特征+筛选的统计特征+筛选的排序特征(后续)
      备用的参与模型融合的模型(注重模型运算上的差异)：
           lgb、xgb、cab
           
    此处过程就能够体现代码之间的耦合性多重要了，我需要把每个处理过程都能够随时控制去除，这样的话，就能减少非常多的代码重复量。
    这样融合下来可能稳定性更强一些，因为能够应对多种突发情况， 因为每一种方案都有适用情况。这样下来综合方案适用更加广阔
'''
#######################################   处理方式一   ##################################
def do_process1():
    # 读取文件
    train_xy = pd.read_csv("data/train_xy.csv",header=0,sep=",")
    train_x = pd.read_csv("data/train_x.csv",header=0,sep=",")
    test_all = pd.read_csv("data/test_all.csv",header=0,sep=",")
    print(train_xy.shape)
    print(train_x.shape)
    print(test_all.shape)
    train = train_xy.copy()
    test = test_all.copy()
    test['y'] = -1
    # 合并一下train 和 test
    data = pd.concat([train,test],axis = 0) # train_xy，test_all索引上连接
    print(train.shape)
    print(test.shape)
    print(data.shape)
    #########   列名划分  ########
    # 对剩下的特征进行分析，分为数值型 、 类别型
    numerical_features = []
    categorical_features = []
    for i in range(157):
        feat = "x_" + str(i+1)
        if i <= 94: # 1-95
            numerical_features.append(feat)
        else:
            categorical_features.append(feat)
    print("有用的数值型特征：",len(numerical_features))
    print("有用的类别型特征：",len(categorical_features))
    ##########
    train = data.loc[data['y']!=-1,:] # train set
    test = data.loc[data['y']==-1,:]  # test set
    print(train.shape)
    print(test.shape)

    # 获取特征列，去除id，group, y
    no_features = ['cust_id','cust_group','y']
    features = [feat for feat in train.columns.values if feat not in no_features]
    print("所有特征的维度：",len(features))
    # 得到输入X ，输出y
    train_id = train['cust_id'].values
    y = train['y'].values
    X = train[features].values
    print("X shape:",X.shape)
    print("y shape:",y.shape)
    test_id = test['cust_id'].values
    test_data = test[features].values

    return y,X,test_data

#######################################   处理方式二   ##################################
def do_process2():
    # 读取文件
    train_xy = pd.read_csv("data/train_xy.csv",header=0,sep=",")
    test_all = pd.read_csv("data/test_all.csv",header=0,sep=",")

    print(train_xy.shape)
    print(test_all.shape)

    train = train_xy.copy()
    test = test_all.copy()
    test['y'] = -1
    # 合并一下train 和 test
    data = pd.concat([train,test],axis = 0) # train_xy，test_all索引上连接
    print(train.shape)
    print(test.shape)
    print(data.shape)
    ################################################   列名划分  #################################################################
    # 对剩下的特征进行分析，分为数值型 、 类别型
    numerical_features = []
    categorical_features = []
    for i in range(157):
        feat = "x_" + str(i+1)
        if i <= 94: # 1-95
            numerical_features.append(feat)
        else:
            categorical_features.append(feat)
    print("有用的数值型特征：",len(numerical_features))
    print("有用的类别型特征：",len(categorical_features))

    #################################################  缺失值统计  ################################################################
    # 统计每个用户缺失值的个数
    def get_nan_count(data):
        df = data.copy()
        df = df.replace(-99,np.nan)
        df['nan_count'] = df.shape[1] - df.count(axis = 1).values  # 列数 - 非nan数
        dummy = pd.get_dummies(pd.cut(df['nan_count'],7),prefix = 'nan') # 对缺失数据进行离散化,划分为7个区间
        print(dummy.shape)
        res = pd.concat([data,dummy],axis = 1) # 合并到原来的数据
        print(res.shape)
        return res
    data = get_nan_count(data)
    #######################################################  缺失填充   ##########################################################
    # 重要性top24
    imp_feat = [ 'x_80', 'x_2', 'x_81', 'x_95', 'x_1',
                 'x_52', 'x_63', 'x_54', 'x_43', 'x_40',
                 'x_93', 'x_42', 'x_157', 'x_62', 'x_29',
                 'x_61', 'x_55', 'x_79', 'x_59', 'x_69',
                 'x_48', 'x_56', 'x_7', 'x_64']
    print("重要的特征个数：",len(imp_feat))
    # 对一些重要的特征进行填充，
    for feat in imp_feat[:10]: # 填充top 10 ,而不是所有
        if feat in numerical_features:   # 数值型用均值
            data[feat] = data[feat].replace(-99,np.nan)
            data[feat] = data[feat].fillna(data[feat].mean()) # 非nan均值
        if feat in categorical_features: # 类别型：不处理、中位数 、众数
            print("这是类别特征：",feat)

    pass
    #################################################################################################################
    train = data.loc[data['y']!=-1,:] # train set
    test = data.loc[data['y']==-1,:]  # test set
    print(train.shape)
    print(test.shape)

    # 获取特征列，去除id，group, y
    no_features = ['cust_id','cust_group','y']
    features = [feat for feat in train.columns.values if feat not in no_features]
    print("所有特征的维度：",len(features))


    # 得到输入X ，输出y
    train_id = train['cust_id'].values
    y = train['y'].values
    X = train[features].values
    print("X shape:",X.shape)
    print("y shape:",y.shape)

    test_id = test['cust_id'].values
    test_data = test[features].values
    print("test shape",test_data.shape)

    #################  下面方式为了半监督学习，保留所属的组信息  ##################
    train.to_csv('../feature_data/suse_all_train.csv')


    return y,X,test_data
#######################################   处理方式三   ##################################

def do_process3():
    # 读取文件
    train_xy = pd.read_csv("data/train_xy.csv", header=0, sep=",")
    test_all = pd.read_csv("data/test_all.csv", header=0, sep=",")
    train = train_xy.copy()
    test = test_all.copy()
    test['y'] = -1
    # 合并一下train 和 test    这样以同样的方式进行处理，更加省事些
    data = pd.concat([train, test], axis=0)  # train_xy，test_all索引上连接
    # 删除一些不必要的特征（噪音、缺失严重、单值、重复等）     删除掉训练集和测试集中同时缺失大于百分之95的
    # train ,test 分开分析
    # 处理一下缺失值严重的特征列，删除
    def get_nan_feature(train, rate=0.95):
        total_num = train.shape[0]
        train_nan_feats = []
        for i in range(157):
            feat = 'x_' + str(i + 1)
            nan_num = train.loc[train[feat] == -99, :].shape[0]
            nan_rate = nan_num / float(total_num)

            if nan_rate == 1.0:  # 只有nan
                train_nan_feats.append(feat)
            if nan_rate > rate:  # 有缺失值 nan,而且缺失严重
                if len(train[feat].unique()) == 2:  # 只有nan + 一个其他值
                    train_nan_feats.append(feat)
        print("一共有 %d 个特征列的缺失值严重，超过%f " % (len(train_nan_feats), rate))
        return train_nan_feats

    train_nan_feats = get_nan_feature(train)
    test_nan_feats = get_nan_feature(test)
    print("缺失严重的特征：train =?= test------", np.all(train_nan_feats == test_nan_feats))

    # 对这些特征取并集:28个
    nan_feats = list(set(train_nan_feats) | set(test_nan_feats))  # 按照train | test的结果,并集，交集，A or B 都一样     并集方式的，其实可以多尝试
    print('严重缺失的特征有 %d 个。' % (len(nan_feats)))

    # 总的删除的特征（发现删重复的5个特征，效果不好，所以只删除28个缺失严重的特征,这是有尝试过得）
    drop_feats = nan_feats
    print('一共删除的特征有 %d 个。' % (len(drop_feats)))
    print(drop_feats)

    # 删除缺失值严重的特征列
    train = train.drop(drop_feats, axis=1)
    test = test.drop(drop_feats, axis=1)
    data = data.drop(drop_feats, axis=1)
    print(data.shape)
    print(train.shape)
    print(test.shape)
    # 删除的特征，全部是特征重要性=0的特征，所以删除 = 原始的特征（不影响精确效果，但是可以减少噪音）
    # 删除了x_92 , x_94 是数值型的，其他 24 个 全部是 类别型
    # 对剩下的特征进行分析，分为数值型 、 类别型        (这里有疑问，难道默认0-94列特征是数值型特征吗，这很不合理啊)
    numerical_features = []
    categorical_features = []
    for i in range(157):
        feat = "x_" + str(i+1)
        if feat not in drop_feats:
            if i <= 94: # 1-95
                numerical_features.append(feat)
            else:
                categorical_features.append(feat)
    # 统计每个样本缺失值的个数                  统计缺失样本数
    def get_nan_count(data, feats, bins=7):
        df = data[feats].copy()
        df = df.replace(-99, np.nan)
        print('总列数:', df.shape[1])  # 这列展示了每一行缺失的特征的数量
        print('每行非空列数:', df.count(axis=1).values)  # 这列展示了每一行缺失的特征的数量
        df['nan_count'] = df.shape[1] - df.count(axis=1).values  # 列数 - 非nan数
        print('每行空列数:', df['nan_count'])  # 这列展示了每一行缺失的特征的数量
        dummy = pd.get_dummies(pd.cut(df['nan_count'], bins),
                               prefix='nan')  # 把每行空列数，做7分离散化再转one-hot编码 对缺失数据进行离散化,划分为7个区间,对于划分区间，这里根据空值情况来造dummies特征
        print(dummy.shape)
        res = pd.concat([data, dummy], axis=1)  # 合并到原来的数据
        print(res.shape)
        return res

    # 在全部特征上面统计缺失值      新加入了对缺失值统计的7列
    data = get_nan_count(data, data.columns.values, 7)
    print('训练集的特征列：', data.columns)

    # 获取缺失很少的数值型的特征         缺失少的数值型用均值
    def get_little_nan_feats(df, numerical_features, rate=0.1):
        total_num = df.shape[0]
        little_nan_feats = []
        for feat in numerical_features:
            nan_num = df.loc[df[feat] == -99, :].shape[0]
            nan_rate = nan_num / float(total_num)
            if nan_rate <= rate:
                little_nan_feats.append(feat)
                # print("feature:",feat,"nan_num = ",nan_num,"nan_rate = ",nan_rate)
        print("一共有 %d 个特征列的缺失值较少，低于%f " % (len(little_nan_feats), rate))
        return little_nan_feats

    little_nan_feats = get_little_nan_feats(data, numerical_features)
    # 对数值型的特征，处理为rank特征（鲁棒性好一点）     数值本身就代表大小的意思，这里构建排序特征并进行归一化，效果会更加鲁棒一些。
    for feat in numerical_features:
        #print('rank前：',data[feat])
        data[feat] = data[feat].rank() / float(data.shape[0]) # 排序，并且进行归一化        这样也行？
        #print('rank后：',data[feat])
    print('训练集的特征列：',data.columns)

    # 对重要的特征进行填充，之后还会想着对着重要的特征利用其进行采样操作
    imp_feat = ['x_80', 'x_2', 'x_81', 'x_95', 'x_1', 'x_52', 'x_63', 'x_54', 'x_43', 'x_40', 'x_93', 'x_42', 'x_157',
                'x_62', 'x_29', 'x_61', 'x_55']
    print('假定的重要特征个数为：', len(imp_feat))
    for feat in imp_feat[:10]:
        if feat in numerical_features:
            print('进行填充吧')
            data[feat] = data[feat].replace(-99, np.nan)
            data[feat] = data[feat].fillna(data[feat].mean())
        if feat in categorical_features:
            print('这是类别特征：', feat)
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

    print('--------  储备五种方式的处理形式  ---------')
    print('第一种形式的文件生成')
    y, X, test_data=do_process1()
    print('第二种形式的文件生成')
    y2, X2, test_data2=do_process2()
    print('第三种形式的文件生成')
    y3, X3, test_data3 = do_process3()
    print('--------  构建不同模型的数据利用  ---------')
    the_score_lg1,r_lg1=lgb_model(X, y, test_data)
    the_score_lg2,r_lg2=lgb_model(X2, y2, test_data2)
    the_score_lg3,r_lg3=lgb_model(X3, y3, test_data3)
    the_score_xg1,r_xg1=xgb_method(X, y, test_data)
    the_score_xg2,r_xg2=xgb_method(X2, y2, test_data2)
    the_score_xg3,r_xg3=xgb_method(X3, y3, test_data3)
    print('--------  融合处理  ---------')
    print('线下的得分依次是：',the_score_lg1,'   ',the_score_lg2,'   ',the_score_lg3,'   ',the_score_xg1,'   ',the_score_xg2,'   ',the_score_xg3)

    the_record_score = (the_score_lg1 + the_score_lg2 + the_score_lg3 + the_score_xg1 + the_score_xg2 + the_score_xg3) / 6
    the_avg_sub=0*r_lg1+0*r_lg2+1*r_lg3+0*r_xg1+0*r_xg2+0*r_xg3


    filepath = '../result/多数据集多特征多模型融合方案' + str(the_record_score) + '.csv'  # 线下平均分数
    # 转为array
    print('result shape:', the_avg_sub.shape)

    sub_sample = pd.read_csv('../result/xgb_nan.csv')
    result = DataFrame()
    result['cust_id'] = sub_sample['cust_id']
    result['pred_prob'] = the_avg_sub
    result.to_csv(filepath, index=False, sep=",")


'''

对6种方式融合的效果：
    第一种：
        0.05*r_lg1+0.15*r_lg2+0.3*r_lg3+0.1*r_xg1+0.3*r_xg2+0.1*r_xg3
        mean auc: 0.8190709923066477
        总的结果： (5, 10000)
        result shape: (10000,)
        --------  融合处理  ---------
        线下的得分依次是： 0.8191922695131673     0.8206291690409669     0.8192571164144196     0.8194091860993773     0.819775776724755     0.8190709923066477
        result shape: (10000,)
        线上：0.75188
    第二种：
        0.1*r_lg1+0.2*r_lg2+0.2*r_lg3+0.1*r_xg1+0.2*r_xg2+0.2*r_xg3
    
'''