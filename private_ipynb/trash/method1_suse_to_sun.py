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
        放上你自己的参数
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



if __name__=='__main__':
    ########################################       方案一.使用纯阈值划分方式（无组考虑）    #################################
    '''
    实验去寻找合适的阈值    对于方案一还是不要使用全加进入样本了吧 ，  本身就是不均衡的，当再加入样本之后，
    不均衡方案会相当严重，所以这种方式的较优策略是只是把阈值划分后，为正例的样本加过去。
    '''
    #gen_sample_use_model_thre()
    #获取阈值下的合适的结果
    #create_sample_use_model_thre()









#--------------------------------------------------------------------------------------------------------------------------------------------
'''
1.方案一：
*使用简单阈值方式有两个大于0.823
   当前的threshold为：0.11800000000000009   得分：0.8235630178730116
   当前的threshold为：0.05100000000000004   得分：0.8231970317080327
使用这种全量放置全部划分后的样本的方式，效果会变差(0.74+)，
*改进下，我们只把划分好的正例样本放置进去，看看效果。

'''




