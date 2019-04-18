# coding=gbk
import h5py
import numpy as np
from private_ipynb.lgb import lgb_model
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import time
import numpy as np
import h5py
import pandas as pd
from pandas import *
import pickle
'''
思想（一一尝试）：
   *使用模型加阈值加分组因素做一次   (在 Sesu_learning中)
   *使用knn针对重要的样本、将重要的无缺失的特征进行训练，之后一个个带进去试试这样的无标签预测是否可以
   *使用外加的方式不断跑模型筛选特征看效果【对于交叉特征 和 排序特征】
   *对训练集中的正例样本进行复制扩展，并在扩展出的样本中加入高斯扰动。
   *关于验证方式的思考，如果把分组信息参与到验证的cv中嫩，如果使用使用全量数据直接跑模型提交呢。
'''
def the_valid_model(X,X_sesu,y,y_sesu,test_data):
    print("start：********************************")
    start = time.time()
    # 进行分离，原特征和半监督特征的分离

    N = 5
    skf = StratifiedKFold(n_splits=N, shuffle=True, random_state=2018)

    auc_cv = []
    pred_cv = []

    for k, (train_in, test_in) in enumerate(skf.split(X, y)):
        X_train, X_test, y_train, y_test = X[train_in], X[test_in], \
                                           y[train_in], y[test_in]

        X_train=np.concatenate([X_train, X_sesu], axis=0)
        y_train=np.concatenate([y_train, y_sesu], axis=0)
        # 数据结构
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
        # 设置参数
        params = {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': {'auc'},
            'max_depth': 4,
            'min_child_weight': 6,
            'num_leaves': 16,
            'learning_rate': 0.02,
            'feature_fraction': 0.7,
            'bagging_fraction': 0.7,
            'bagging_freq': 5,
            # 'lambda_l1':0.25,
            # 'lambda_l2':0.5,
            # 'scale_pos_weight':691.0/14309.0, 不能设置
            # 'num_threads':4,
        }

        print('................Start training..........................')
        # train
        gbm = lgb.train(params,
                        lgb_train,
                        num_boost_round=2000,
                        valid_sets=lgb_eval,
                        early_stopping_rounds=100,
                        verbose_eval=100)

        print('................Start predict .........................')
        # 预测
        y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
        # 评估      roc计算评估得分
        tmp_auc = roc_auc_score(y_test, y_pred)
        auc_cv.append(tmp_auc)
        print("valid auc:", tmp_auc)
        # test
        pred = gbm.predict(test_data, num_iteration=gbm.best_iteration)
        pred_cv.append(pred)

    # K交叉验证的平均分数
    print('the cv information:')
    print(auc_cv)
    print('cv mean score', np.mean(auc_cv))

    end = time.time()
    print("......................run with time: ", (end - start) / 60.0)
    print("over:*********************************")

    res = np.array(pred_cv)
    print("总的结果：", res.shape)

    # 最后结果，mean，max，min
    r = res.mean(axis=0)
    print('result shape:', r.shape)

    return np.mean(auc_cv),r

##################################  方案一函数   ##########################################
def gen_sample_use_model_thre():
    print('------------------- 1.对每个样本进行初步预测 -----------------------')
    with h5py.File('do_pro_data/pro_data.hdf5') as f:
        y = f['y'][0:20000]
        X = f['X'][0:20000]
        test_data = f['test_data'][0:20000]
        sesu_pro_data = f['sesu_pro_data'][0:20000]
        print(sesu_pro_data)
    the_train_score, the_pred_label = lgb_model(X, y, sesu_pro_data)
    print(len(the_pred_label))

    print('------------------- 2.探寻阈值 -----------------------')
    #遍历查找合适的threshold    从0.02开始，每次增加0.02
    threshold = 0.118
    the_feature_X=X
    the_label_y=y

    file_context = open('record/sesu_record.txt','w')
    while(threshold<0.8):
        the_feature_X = X
        new_pre_label_list = []
        for i in the_pred_label:
            if i > threshold:
                new_pre_label_list.append(1)
            else:
                new_pre_label_list.append(0)

        #the_feature_X=np.concatenate([the_feature_X, sesu_pro_data], axis=0)
        #new_pre_label_list=np.concatenate([y, new_pre_label_list], axis=0)

        print('合并成的数组the_feature_X的维度为：',the_feature_X.shape)
        #print('合并成的数组y的维度为：', new_pre_label_list.shape)

        #我们只使用正例样本，所以对于半监督结果进行进一步的提取
        the_con_dataset=pd.DataFrame(pd.concat([pd.DataFrame(sesu_pro_data),pd.DataFrame(new_pre_label_list,columns=['label'])],axis=1))
        the_con_dataset.to_csv('../feature_data/sesu_mothod1_concat.csv')
        #print([i for i in the_con_dataset.columns if i not in ['label']])
        the_con_dataset=the_con_dataset[the_con_dataset['label']==1]
        print('the con shape:',the_con_dataset.shape)
        sesu_pro_data=the_con_dataset[[i for i in the_con_dataset.columns if i not in ['label']]]
        new_pre_label_list=the_con_dataset['label'].values

        the_record_score, _ = the_valid_model(the_feature_X,sesu_pro_data, y,new_pre_label_list,test_data)
        file_context = open('record/sesu_record.txt', 'a+')
        file_context.writelines('当前的threshold为：'+str(threshold)+'   得分：'+str(the_record_score)+'\n')
        print('写入的信息为：','当前的threshold为：'+str(threshold)+'   得分：'+str(the_record_score))
        #np.array(list(the_feature_X).extend(sesu_pro_data))

        #np.array(list(y).extend(new_pre_label_list))

        print('***********************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************')
        file_context.close()
        threshold=threshold+0.001
    file_context.close()

def create_sample_use_model_thre():
    print('------------------- 1.对每个样本进行初步预测 -----------------------')
    with h5py.File('do_pro_data/pro_data.hdf5') as f:
        y = f['y'][0:20000]
        X = f['X'][0:20000]
        test_data = f['test_data'][0:20000]
        sesu_pro_data = f['sesu_pro_data'][0:20000]
        print(sesu_pro_data)
    the_train_score, the_pred_label = lgb_model(X, y, sesu_pro_data)
    print(len(the_pred_label))

    print('------------------- 2.阈值预测 -----------------------')
    #遍历查找合适的threshold    从0.02开始，每次增加0.02
    threshold = 0.118
    the_feature_X=X
    the_label_y=y

    the_feature_X = X
    new_pre_label_list = []
    for i in the_pred_label:
        if i > threshold:
            new_pre_label_list.append(1)
        else:
            new_pre_label_list.append(0)

    #the_feature_X=np.concatenate([the_feature_X, sesu_pro_data], axis=0)
    #new_pre_label_list=np.concatenate([y, new_pre_label_list], axis=0)

    print('合并成的数组the_feature_X的维度为：',the_feature_X.shape)
    #print('合并成的数组y的维度为：', new_pre_label_list.shape)
    # 我们只使用正例样本，所以对于半监督结果进行进一步的提取
    the_con_dataset = pd.DataFrame(
        pd.concat([pd.DataFrame(sesu_pro_data), pd.DataFrame(new_pre_label_list, columns=['label'])], axis=1))
    the_con_dataset.to_csv('../feature_data/sesu_mothod1_concat.csv')
    # print([i for i in the_con_dataset.columns if i not in ['label']])
    the_con_dataset = the_con_dataset[the_con_dataset['label'] == 1]
    print('the con shape:', the_con_dataset.shape)
    sesu_pro_data = the_con_dataset[[i for i in the_con_dataset.columns if i not in ['label']]]
    new_pre_label_list = the_con_dataset['label'].values


    #开始做预测
    the_record_score, result_file = the_valid_model(the_feature_X,sesu_pro_data, y,new_pre_label_list,test_data)
    print('是这个score:',the_record_score)

    filepath = '../result/lgb_半监督简单方案1改只正样本_'+ str(the_record_score)+'.csv' # 线下平均分数
    # 转为array
    print('result shape:',result_file.shape)

    sub_sample=pd.read_csv('../result/xgb_nan.csv')
    result = DataFrame()
    result['cust_id'] = sub_sample['cust_id']
    result['pred_prob'] = result_file
    result.to_csv(filepath,index=False,sep=",")


##################################  方案二函数   ##########################################
def use_group_number_se():
    print('---------------------  1.执行初步预测  ------------------------')
    with h5py.File('do_pro_data/pro_data.hdf5') as f:
        y = f['y'][0:20000]
        X = f['X'][0:20000]
        test_data = f['test_data'][0:20000]
        sesu_pro_data = f['sesu_pro_data'][0:20000]
        print(sesu_pro_data)
    the_train_score, the_pred_label = lgb_model(X, y, sesu_pro_data)
    print(len(the_pred_label))
    print('---------------------  2.获取组内正负样本大致数量  ------------------------')
    train_x=pd.read_csv("../feature_data/suse_test_data.csv",header=0,sep=",")
    print(train_x.columns)
    print(train_x['cust_group'].value_counts())
    '''
    利用原train数据中的信息：
        group1 ->  4544:456 = 10:1
        group2 ->  4871:129 = 37:1
        group3 ->  4894:106 = 46:1
    无标签样本中：
        group_1    3500   /10=350
        group_2    3500   /37=95
        group_3    3000   /46=65
    '''
    print('---------------------  3.根据取到的数值进行排序选取  ------------------------')
    train_x['the_pre_y_prob']=the_pred_label

    i=0
    # def group_nei(the_new_data):
    #     print('!!!!!!!!!!!!：',the_new_data)
    #     the_new_data=pd.DataFrame(the_new_data)
    #     the_new_data.sort_values('the_pre_y_prob',inplace=True,ascending=False)
    #     print(the_new_data)
    #     i=1
    #     print('i为：   ',i)
    #     return the_new_data[:50]
    # b=(train_x.groupby(['cust_group'])).apply(group_nei)
    # print(b['group_1'])
    '''
    分组并排序,  自己写的话真麻烦，直接找简洁方式都是可以实现的
    其中这里method是一种如果有重复情况下的方式选择
    '''
    train_x['group_sort'] = train_x['the_pre_y_prob'].groupby(train_x['cust_group']).rank(ascending=0, method='first')
    dataframe1=train_x[(train_x['cust_group']=='group_1') & (train_x['group_sort']<=350)]
    dataframe1['the_pre_y_prob'] = 1
    #print(dataframe1)

    dataframe2=train_x[(train_x['cust_group']=='group_2') & (train_x['group_sort']<=95)]
    dataframe2['the_pre_y_prob'] = 1
    #print(dataframe2)

    dataframe3=train_x[(train_x['cust_group']=='group_3') & (train_x['group_sort']<=65)]
    dataframe3['the_pre_y_prob'] = 1
    #print(dataframe3)

    the_big_frame=pd.concat([dataframe1,dataframe2,dataframe3])
    print(the_big_frame)
    #train_x.to_csv('../feature_data/do_group_sort.csv')

    print('---------------------  4.跑模型得到半监督结果  ------------------------')

    column=[i for i in the_big_frame.columns if i not in ['group_sort','the_pre_y_prob','cust_group','Unnamed: 0']]
    the_feature_X=the_big_frame[column]
    new_pre_label_list=the_big_frame['group_sort']
    the_record_score, result_file = the_valid_model(X, the_feature_X, y, new_pre_label_list, test_data)

    #效果提交
    filepath = '../result/lgb_根据group数量关系方式2_' + str(the_record_score) + '.csv'  # 线下平均分数
    # 转为array
    print('result shape:', result_file.shape)

    sub_sample = pd.read_csv('../result/xgb_nan.csv')
    result = DataFrame()
    result['cust_id'] = sub_sample['cust_id']
    result['pred_prob'] = result_file
    result.to_csv(filepath, index=False, sep=",")
    pass

##################################  方案三函数   ##########################################
def knn_gen_method():
    print('-------  简单 前置工作  --------')
    from sklearn import datasets,neighbors
    with h5py.File('do_pro_data/pro_data.hdf5') as f:
        y = f['y'][0:20000]
        X = f['X'][0:20000]
        test_data = f['test_data'][0:20000]
        sesu_pro_data = f['sesu_pro_data'][0:20000]
        print(sesu_pro_data)
    print('-------  加载数据并做列名处理  --------')
    suse_test=pd.read_csv("../feature_data/suse_test_data.csv",header=0,sep=",")

    suse_train=pd.read_csv("../feature_data/suse_all_train.csv",header=0,sep=",")
    suse_test=suse_test.drop(columns=['Unnamed: 0', 'cust_group'])
    suse_train=suse_train.drop(columns=['Unnamed: 0', 'cust_group', 'cust_id'])
    the_train_y=suse_train.pop('y')
    print(len(suse_test.columns))
    print(len(suse_train.columns))
    suse_train.rename(
        columns={'nan_(35.895, 51.0]':'nan1','nan_(51.0, 66.0]':'nan2','nan_(66.0, 81.0]':'nan3','nan_(81.0, 96.0]':'nan4','nan_(96.0, 111.0]':'nan5','nan_(111.0, 126.0]':'nan6','nan_(126.0, 141.0]':'nan7'},inplace=True)
    suse_test.rename(
        columns={'nan_(33.893, 49.286]': 'nan1', 'nan_(49.286, 64.571]': 'nan2', 'nan_(64.571, 79.857]': 'nan3',
                 'nan_(79.857, 95.143]': 'nan4', 'nan_(95.143, 110.429]': 'nan5', 'nan_(110.429, 125.714]': 'nan6',
                 'nan_(125.714, 141.0]': 'nan7'}, inplace=True)

    print('-------  找到无缺失列，并做归一化处理  --------')
    test_have_nan_columns=[]
    for i in suse_test.columns:
        if -99 not in list(suse_test[i]):
            test_have_nan_columns.append(i)
            print('这列没有缺失：',i)
    train_have_nan_columns=[]
    for i in suse_train.columns:
        if -99 not in list(suse_train[i]):
            train_have_nan_columns.append(i)
            print('这列没有缺失：',i)

    the_jiao_list=list(set(test_have_nan_columns)&set(train_have_nan_columns))
    the_new_train=suse_train[the_jiao_list]
    the_new_test=suse_test[the_jiao_list]

    #归一化处理
    suse_test_norm = the_new_test.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    suse_train_norm = the_new_train.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))

    print('-------  训练模型  --------')
    kn_clf=neighbors.KNeighborsClassifier()
    kn_clf.fit(suse_train_norm,the_train_y)
    the_probility=pd.DataFrame(kn_clf.predict_proba(suse_test_norm))
    print('-------  利用预测概率生成无标签样本结果  --------')
    print('概率展示为：',the_probility)
    the_probility.to_csv('../feature_data/the_probility.csv')
    print('列名是：',the_probility.columns)
    the_probility.rename(
            columns={0:'the_prob_0',1:'the_prob_1'},inplace=True)
    suse_test['sample_prob']=the_probility['the_prob_1']
    suse_test['sample_prob_rank'] = suse_test['sample_prob'].rank(ascending=0, method='first')
    data_frame=suse_test[suse_test['sample_prob_rank']<100]
    data_frame['sample_prob']=1
    data_frame=data_frame.drop(columns='sample_prob_rank')
    data_frame.to_csv('../exp_show/the_test.csv')

    print('-------  融入样本，线下cv结果  --------')
    new_pre_label_list=data_frame.pop('sample_prob')
    the_record_score, result_file = the_valid_model(suse_train.values, data_frame.values, the_train_y, new_pre_label_list, test_data)

    #效果提交
    filepath = '../result/lgb_使用knn做分类生成300正例方式3_' + str(the_record_score) + '.csv'  # 线下平均分数
    # 转为array
    print('result shape:', result_file.shape)

    sub_sample = pd.read_csv('../result/xgb_nan.csv')
    result = DataFrame()
    result['cust_id'] = sub_sample['cust_id']
    result['pred_prob'] = result_file
    result.to_csv(filepath, index=False, sep=",")
    pass
if __name__=='__main__':
    ########################################       方案一.使用纯阈值划分方式（无组考虑）    #################################
    '''
    实验去寻找合适的阈值    对于方案一还是不要使用全加进入样本了吧 ，  本身就是不均衡的，当再加入样本之后，
    不均衡方案会相当严重，所以这种方式的较优策略是只是把阈值划分后，为正例的样本加过去。
    '''
    #gen_sample_use_model_thre()
    #获取阈值下的合适的结果
    #create_sample_use_model_thre()
    ########################################       方案二.使用按数量比例和概率划分方式（使用组考虑）    #################################
    #use_group_number_se()
    ########################################       方案三.使用knn自归类方式（无组考虑，其实空值当-99能够在一些树模型中直接使用，代表这个人极其懒的意义）    #################################
    #knn自回归加次数限制
    knn_gen_method()

    ########################################       方案四.强行造复制正样本加高斯噪声    #################################


#--------------------------------------------------------------------------------------------------------------------------------------------
'''
1.方案一：
*使用简单阈值方式有两个大于0.823
   当前的threshold为：0.11800000000000009   得分：0.8235630178730116
   当前的threshold为：0.05100000000000004   得分：0.8231970317080327
使用这种全量放置全部划分后的样本的方式，效果会变差(0.74+)，
*改进下，我们只把划分好的正例样本放置进去，看看效果。

2.方案二：
   按group数量进行切分的方案：0.8203

3.方案三：
*感觉使用knn的方式真的是好差，效果显示，这样的使用无缺失的特征进行预测，当使用200个作为补全时候效果还行，当
     生成300正例时 0.8189
     生成200正例时 0.8202
'''




