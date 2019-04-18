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
   *ʹ��knn�����Ҫ������������Ҫ����ȱʧ����������ѵ����֮��һ��������ȥ�����������ޱ�ǩԤ���Ƿ����
   *ʹ����ӵķ�ʽ������ģ��ɸѡ������Ч�������ڽ������� �� ����������
   *��ѵ�����е������������и�����չ��������չ���������м����˹�Ŷ���
   *������֤��ʽ��˼��������ѷ�����Ϣ���뵽��֤��cv���ۣ����ʹ��ʹ��ȫ������ֱ����ģ���ύ�ء�
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
            # 'scale_pos_weight':691.0/14309.0, ��������
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

##################################  ����һ����   ##########################################
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


##################################  ����������   ##########################################
def use_group_number_se():
    print('---------------------  1.ִ�г���Ԥ��  ------------------------')
    with h5py.File('do_pro_data/pro_data.hdf5') as f:
        y = f['y'][0:20000]
        X = f['X'][0:20000]
        test_data = f['test_data'][0:20000]
        sesu_pro_data = f['sesu_pro_data'][0:20000]
        print(sesu_pro_data)
    the_train_score, the_pred_label = lgb_model(X, y, sesu_pro_data)
    print(len(the_pred_label))
    print('---------------------  2.��ȡ��������������������  ------------------------')
    train_x=pd.read_csv("../feature_data/suse_test_data.csv",header=0,sep=",")
    print(train_x.columns)
    print(train_x['cust_group'].value_counts())
    '''
    ����ԭtrain�����е���Ϣ��
        group1 ->  4544:456 = 10:1
        group2 ->  4871:129 = 37:1
        group3 ->  4894:106 = 46:1
    �ޱ�ǩ�����У�
        group_1    3500   /10=350
        group_2    3500   /37=95
        group_3    3000   /46=65
    '''
    print('---------------------  3.����ȡ������ֵ��������ѡȡ  ------------------------')
    train_x['the_pre_y_prob']=the_pred_label

    i=0
    # def group_nei(the_new_data):
    #     print('!!!!!!!!!!!!��',the_new_data)
    #     the_new_data=pd.DataFrame(the_new_data)
    #     the_new_data.sort_values('the_pre_y_prob',inplace=True,ascending=False)
    #     print(the_new_data)
    #     i=1
    #     print('iΪ��   ',i)
    #     return the_new_data[:50]
    # b=(train_x.groupby(['cust_group'])).apply(group_nei)
    # print(b['group_1'])
    '''
    ���鲢����,  �Լ�д�Ļ����鷳��ֱ���Ҽ�෽ʽ���ǿ���ʵ�ֵ�
    ��������method��һ��������ظ�����µķ�ʽѡ��
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

    print('---------------------  4.��ģ�͵õ���ල���  ------------------------')

    column=[i for i in the_big_frame.columns if i not in ['group_sort','the_pre_y_prob','cust_group','Unnamed: 0']]
    the_feature_X=the_big_frame[column]
    new_pre_label_list=the_big_frame['group_sort']
    the_record_score, result_file = the_valid_model(X, the_feature_X, y, new_pre_label_list, test_data)

    #Ч���ύ
    filepath = '../result/lgb_����group������ϵ��ʽ2_' + str(the_record_score) + '.csv'  # ����ƽ������
    # תΪarray
    print('result shape:', result_file.shape)

    sub_sample = pd.read_csv('../result/xgb_nan.csv')
    result = DataFrame()
    result['cust_id'] = sub_sample['cust_id']
    result['pred_prob'] = result_file
    result.to_csv(filepath, index=False, sep=",")
    pass

##################################  ����������   ##########################################
def knn_gen_method():
    print('-------  �� ǰ�ù���  --------')
    from sklearn import datasets,neighbors
    with h5py.File('do_pro_data/pro_data.hdf5') as f:
        y = f['y'][0:20000]
        X = f['X'][0:20000]
        test_data = f['test_data'][0:20000]
        sesu_pro_data = f['sesu_pro_data'][0:20000]
        print(sesu_pro_data)
    print('-------  �������ݲ�����������  --------')
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

    print('-------  �ҵ���ȱʧ�У�������һ������  --------')
    test_have_nan_columns=[]
    for i in suse_test.columns:
        if -99 not in list(suse_test[i]):
            test_have_nan_columns.append(i)
            print('����û��ȱʧ��',i)
    train_have_nan_columns=[]
    for i in suse_train.columns:
        if -99 not in list(suse_train[i]):
            train_have_nan_columns.append(i)
            print('����û��ȱʧ��',i)

    the_jiao_list=list(set(test_have_nan_columns)&set(train_have_nan_columns))
    the_new_train=suse_train[the_jiao_list]
    the_new_test=suse_test[the_jiao_list]

    #��һ������
    suse_test_norm = the_new_test.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    suse_train_norm = the_new_train.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))

    print('-------  ѵ��ģ��  --------')
    kn_clf=neighbors.KNeighborsClassifier()
    kn_clf.fit(suse_train_norm,the_train_y)
    the_probility=pd.DataFrame(kn_clf.predict_proba(suse_test_norm))
    print('-------  ����Ԥ����������ޱ�ǩ�������  --------')
    print('����չʾΪ��',the_probility)
    the_probility.to_csv('../feature_data/the_probility.csv')
    print('�����ǣ�',the_probility.columns)
    the_probility.rename(
            columns={0:'the_prob_0',1:'the_prob_1'},inplace=True)
    suse_test['sample_prob']=the_probility['the_prob_1']
    suse_test['sample_prob_rank'] = suse_test['sample_prob'].rank(ascending=0, method='first')
    data_frame=suse_test[suse_test['sample_prob_rank']<100]
    data_frame['sample_prob']=1
    data_frame=data_frame.drop(columns='sample_prob_rank')
    data_frame.to_csv('../exp_show/the_test.csv')

    print('-------  ��������������cv���  --------')
    new_pre_label_list=data_frame.pop('sample_prob')
    the_record_score, result_file = the_valid_model(suse_train.values, data_frame.values, the_train_y, new_pre_label_list, test_data)

    #Ч���ύ
    filepath = '../result/lgb_ʹ��knn����������300������ʽ3_' + str(the_record_score) + '.csv'  # ����ƽ������
    # תΪarray
    print('result shape:', result_file.shape)

    sub_sample = pd.read_csv('../result/xgb_nan.csv')
    result = DataFrame()
    result['cust_id'] = sub_sample['cust_id']
    result['pred_prob'] = result_file
    result.to_csv(filepath, index=False, sep=",")
    pass
if __name__=='__main__':
    ########################################       ����һ.ʹ�ô���ֵ���ַ�ʽ�����鿼�ǣ�    #################################
    '''
    ʵ��ȥѰ�Һ��ʵ���ֵ    ���ڷ���һ���ǲ�Ҫʹ��ȫ�ӽ��������˰� ��  ������ǲ�����ģ����ټ�������֮��
    �����ⷽ�����൱���أ��������ַ�ʽ�Ľ��Ų�����ֻ�ǰ���ֵ���ֺ�Ϊ�����������ӹ�ȥ��
    '''
    #gen_sample_use_model_thre()
    #��ȡ��ֵ�µĺ��ʵĽ��
    #create_sample_use_model_thre()
    ########################################       ������.ʹ�ð����������͸��ʻ��ַ�ʽ��ʹ���鿼�ǣ�    #################################
    #use_group_number_se()
    ########################################       ������.ʹ��knn�Թ��෽ʽ�����鿼�ǣ���ʵ��ֵ��-99�ܹ���һЩ��ģ����ֱ��ʹ�ã���������˼����������壩    #################################
    #knn�Իع�Ӵ�������
    knn_gen_method()

    ########################################       ������.ǿ���츴���������Ӹ�˹����    #################################


#--------------------------------------------------------------------------------------------------------------------------------------------
'''
1.����һ��
*ʹ�ü���ֵ��ʽ����������0.823
   ��ǰ��thresholdΪ��0.11800000000000009   �÷֣�0.8235630178730116
   ��ǰ��thresholdΪ��0.05100000000000004   �÷֣�0.8231970317080327
ʹ������ȫ������ȫ�����ֺ�������ķ�ʽ��Ч������(0.74+)��
*�Ľ��£�����ֻ�ѻ��ֺõ������������ý�ȥ������Ч����

2.��������
   ��group���������зֵķ�����0.8203

3.��������
*�о�ʹ��knn�ķ�ʽ����ǺòЧ����ʾ��������ʹ����ȱʧ����������Ԥ�⣬��ʹ��200����Ϊ��ȫʱ��Ч�����У���
     ����300����ʱ 0.8189
     ����200����ʱ 0.8202
'''




