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
˼�루һһ���ԣ���
   *ʹ��ģ�ͼ���ֵ�ӷ���������һ��   (�� Sesu_learning��)

'''
def the_valid_model(X,X_sesu,y,y_sesu,test_data):
    print("start��********************************")
    start = time.time()
    # ���з��룬ԭ�����Ͱ�ල�����ķ���

    N = 5
    skf = StratifiedKFold(n_splits=N, shuffle=True, random_state=2018)

    auc_cv = []
    pred_cv = []

    for k, (train_in, test_in) in enumerate(skf.split(X, y)):
        X_train, X_test, y_train, y_test = X[train_in], X[test_in], \
                                           y[train_in], y[test_in]

        X_train=np.concatenate([X_train, X_sesu], axis=0)
        y_train=np.concatenate([y_train, y_sesu], axis=0)
        # ���ݽṹ
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
        # ���ò���
        params = {
        �������Լ��Ĳ���
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
        # Ԥ��
        y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
        # ����      roc���������÷�
        tmp_auc = roc_auc_score(y_test, y_pred)
        auc_cv.append(tmp_auc)
        print("valid auc:", tmp_auc)
        # test
        pred = gbm.predict(test_data, num_iteration=gbm.best_iteration)
        pred_cv.append(pred)

    # K������֤��ƽ������
    print('the cv information:')
    print(auc_cv)
    print('cv mean score', np.mean(auc_cv))

    end = time.time()
    print("......................run with time: ", (end - start) / 60.0)
    print("over:*********************************")

    res = np.array(pred_cv)
    print("�ܵĽ����", res.shape)

    # �������mean��max��min
    r = res.mean(axis=0)
    print('result shape:', r.shape)

    return np.mean(auc_cv),r
def gen_sample_use_model_thre():
    print('------------------- 1.��ÿ���������г���Ԥ�� -----------------------')
    with h5py.File('do_pro_data/pro_data.hdf5') as f:
        y = f['y'][0:20000]
        X = f['X'][0:20000]
        test_data = f['test_data'][0:20000]
        sesu_pro_data = f['sesu_pro_data'][0:20000]
        print(sesu_pro_data)
    the_train_score, the_pred_label = lgb_model(X, y, sesu_pro_data)
    print(len(the_pred_label))

    print('------------------- 2.̽Ѱ��ֵ -----------------------')
    #�������Һ��ʵ�threshold    ��0.02��ʼ��ÿ������0.02
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

        print('�ϲ��ɵ�����the_feature_X��ά��Ϊ��',the_feature_X.shape)
        #print('�ϲ��ɵ�����y��ά��Ϊ��', new_pre_label_list.shape)

        #����ֻʹ���������������Զ��ڰ�ල������н�һ������ȡ
        the_con_dataset=pd.DataFrame(pd.concat([pd.DataFrame(sesu_pro_data),pd.DataFrame(new_pre_label_list,columns=['label'])],axis=1))
        the_con_dataset.to_csv('../feature_data/sesu_mothod1_concat.csv')
        #print([i for i in the_con_dataset.columns if i not in ['label']])
        the_con_dataset=the_con_dataset[the_con_dataset['label']==1]
        print('the con shape:',the_con_dataset.shape)
        sesu_pro_data=the_con_dataset[[i for i in the_con_dataset.columns if i not in ['label']]]
        new_pre_label_list=the_con_dataset['label'].values

        the_record_score, _ = the_valid_model(the_feature_X,sesu_pro_data, y,new_pre_label_list,test_data)
        file_context = open('record/sesu_record.txt', 'a+')
        file_context.writelines('��ǰ��thresholdΪ��'+str(threshold)+'   �÷֣�'+str(the_record_score)+'\n')
        print('д�����ϢΪ��','��ǰ��thresholdΪ��'+str(threshold)+'   �÷֣�'+str(the_record_score))
        #np.array(list(the_feature_X).extend(sesu_pro_data))

        #np.array(list(y).extend(new_pre_label_list))

        print('***********************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************')
        file_context.close()
        threshold=threshold+0.001
    file_context.close()

def create_sample_use_model_thre():
    print('------------------- 1.��ÿ���������г���Ԥ�� -----------------------')
    with h5py.File('do_pro_data/pro_data.hdf5') as f:
        y = f['y'][0:20000]
        X = f['X'][0:20000]
        test_data = f['test_data'][0:20000]
        sesu_pro_data = f['sesu_pro_data'][0:20000]
        print(sesu_pro_data)
    the_train_score, the_pred_label = lgb_model(X, y, sesu_pro_data)
    print(len(the_pred_label))

    print('------------------- 2.��ֵԤ�� -----------------------')
    #�������Һ��ʵ�threshold    ��0.02��ʼ��ÿ������0.02
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

    print('�ϲ��ɵ�����the_feature_X��ά��Ϊ��',the_feature_X.shape)
    #print('�ϲ��ɵ�����y��ά��Ϊ��', new_pre_label_list.shape)
    # ����ֻʹ���������������Զ��ڰ�ල������н�һ������ȡ
    the_con_dataset = pd.DataFrame(
        pd.concat([pd.DataFrame(sesu_pro_data), pd.DataFrame(new_pre_label_list, columns=['label'])], axis=1))
    the_con_dataset.to_csv('../feature_data/sesu_mothod1_concat.csv')
    # print([i for i in the_con_dataset.columns if i not in ['label']])
    the_con_dataset = the_con_dataset[the_con_dataset['label'] == 1]
    print('the con shape:', the_con_dataset.shape)
    sesu_pro_data = the_con_dataset[[i for i in the_con_dataset.columns if i not in ['label']]]
    new_pre_label_list = the_con_dataset['label'].values


    #��ʼ��Ԥ��
    the_record_score, result_file = the_valid_model(the_feature_X,sesu_pro_data, y,new_pre_label_list,test_data)
    print('�����score:',the_record_score)

    filepath = '../result/lgb_��ල�򵥷���1��ֻ������_'+ str(the_record_score)+'.csv' # ����ƽ������
    # תΪarray
    print('result shape:',result_file.shape)

    sub_sample=pd.read_csv('../result/xgb_nan.csv')
    result = DataFrame()
    result['cust_id'] = sub_sample['cust_id']
    result['pred_prob'] = result_file
    result.to_csv(filepath,index=False,sep=",")



if __name__=='__main__':
    ########################################       ����һ.ʹ�ô���ֵ���ַ�ʽ�����鿼�ǣ�    #################################
    '''
    ʵ��ȥѰ�Һ��ʵ���ֵ    ���ڷ���һ���ǲ�Ҫʹ��ȫ�ӽ��������˰� ��  ������ǲ�����ģ����ټ�������֮��
    �����ⷽ�����൱���أ��������ַ�ʽ�Ľ��Ų�����ֻ�ǰ���ֵ���ֺ�Ϊ�����������ӹ�ȥ��
    '''
    #gen_sample_use_model_thre()
    #��ȡ��ֵ�µĺ��ʵĽ��
    #create_sample_use_model_thre()









#--------------------------------------------------------------------------------------------------------------------------------------------
'''
1.����һ��
*ʹ�ü���ֵ��ʽ����������0.823
   ��ǰ��thresholdΪ��0.11800000000000009   �÷֣�0.8235630178730116
   ��ǰ��thresholdΪ��0.05100000000000004   �÷֣�0.8231970317080327
ʹ������ȫ������ȫ�����ֺ�������ķ�ʽ��Ч������(0.74+)��
*�Ľ��£�����ֻ�ѻ��ֺõ������������ý�ȥ������Ч����

'''




