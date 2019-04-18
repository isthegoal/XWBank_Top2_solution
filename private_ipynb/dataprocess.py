# coding=gbk
import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas import Series
import time
import h5py
def process_sesu():
    train_x = pd.read_csv("data/train_x.csv",header=0,sep=",")

    #首先获取分开的列名
    numerical_features = []
    categorical_features = []
    for i in range(157):
        feat = "x_" + str(i + 1)
        if i <= 94:  # 1-95
            numerical_features.append(feat)
        else:
            categorical_features.append(feat)
    print("有用的数值型特征：", len(numerical_features))
    print("有用的类别型特征：", len(categorical_features))

    #第二步.对缺失值进行统计
    def get_nan_count(data):
        df = data.copy()
        df = df.replace(-99,np.nan)
        df['nan_count'] = df.shape[1] - df.count(axis = 1).values  # 列数 - 非nan数
        dummy = pd.get_dummies(pd.cut(df['nan_count'],7),prefix = 'nan') # 对缺失数据进行离散化,划分为7个区间
        print(dummy.shape)
        res = pd.concat([data,dummy],axis = 1) # 合并到原来的数据
        print(res.shape)
        return res
    data = get_nan_count(train_x)
    # 第三步.对缺失值进行填充
    # 重要性top24
    imp_feat = ['x_80', 'x_2', 'x_81', 'x_95', 'x_1',
                'x_52', 'x_63', 'x_54', 'x_43', 'x_40',
                'x_93', 'x_42', 'x_157', 'x_62', 'x_29',
                'x_61', 'x_55', 'x_79', 'x_59', 'x_69',
                'x_48', 'x_56', 'x_7', 'x_64']
    print("重要的特征个数：", len(imp_feat))
    # 对一些重要的特征进行填充，
    for feat in imp_feat[:10]:  # 填充top 10 ,而不是所有
        if feat in numerical_features:  # 数值型用均值
            data[feat] = data[feat].replace(-99, np.nan)
            data[feat] = data[feat].fillna(data[feat].mean())  # 非nan均值
        if feat in categorical_features:  # 类别型：不处理、中位数 、众数
            print("这是类别特征：", feat)
    pass

    #第四步.组织并返回
    no_features = ['cust_id','cust_group','y']
    features = [feat for feat in data.columns.values if feat not in no_features]
    print("所有特征的维度：",len(features))
    test_data = data[features].values
    print('这个是测试样本经过筛选之后的结果：',data[features])

    #################  下面方式为了半监督学习，保留所属的组信息  ##################
    no_features = ['cust_id','y']
    features = [feat for feat in data.columns.values if feat not in no_features]
    pd.DataFrame(data[features]).to_csv('../feature_data/suse_test_data.csv')
    return test_data
def process():
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


########################################    第二种预处理方案的结果   ###################################
def  process_for_sel():
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
    #######################################################  排序特征  ##########################################################
    # 对数值型的特征，处理为rank特征（鲁棒性好一点）     数值本身就代表大小的意思，这里构建排序特征并进行归一化，效果会更加鲁棒一些。
    for feat in numerical_features:
        # print('rank前：',data[feat])
        data['rank_'+feat] = data[feat].rank() / float(data.shape[0])  # 排序，并且进行归一化        这样也行？
        # print('rank后：',data[feat])
    print('训练集的特征列：', data.columns)

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

    #对第二种预处理方案的结果进行保存
    train.to_csv('../feature_data/process_have_sort_train.csv')
    test.to_csv('../feature_data/process_have_sort_test.csv')
    pass

if __name__=='__main__':
    print('-------  第一种预处理方案，其实这个方案更加适合lgb,并没有做排序特征处理  -------')
    y, X, test_data=process()
    sesu_pro_data=process_sesu()
    with h5py.File('do_pro_data/pro_data.hdf5','w') as f:
        f['y']=y
        f['X']=X
        f['test_data']=test_data
        f['sesu_pro_data'] = sesu_pro_data

    print('-------  为了特征选择方案中的排序特征筛选做出的处理(第二种预处理方案，保留了排序前和排序后的特征)  -------')
    process_for_sel()