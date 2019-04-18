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
import xgboost as xgb
import pandas as pd
from pandas import *
import pickle
'''
����׼�����м�������̽���ķ�����
     ����һ.���ڽ�������ʹ������Լ����������ķ�ʽ��ɸѡ������ģ�ͣ���������Բ���Ҫģ�ͣ��������ϵ����
     ������.���ڽ�������ʹ�������𲽼��룬������ģ��������ķ�ʽ��ɸѡ������ģ�ͣ�ʹ������ģ�Ͱɣ�
     ������.������ͬ�ϣ����������������ʹ��ͬ�ϵķ�������ɸѡ��ģ��������Ҫ�ȼ��㣩
     ������.ģ���˻��̰��ɸѡ����������ѡ�������� ���������ԣ�
һ��˼����
    *��������任ʱ��Ҫʹ���������������������û��ʲô�����
'''
#################################################  ����һ����   ##################################################
def get_division_feature(data ,feature_name):
    # ���������֮�� ��������任�������� ÿ��������֮����б仯��  ���ҽ��б任֮��ѱ仯��� ����֤����¼��������֪���˰ɡ�����֮ǰֱ��һ��
    # ����������ȫ��ʧ��
    new_feature = []
    new_feature_name = []
    for i in range(len(data[feature_name].columns ) -1):
        for j in range( i +1 ,len(data[feature_name].columns)):
            # �����´���������ֵ��������
            new_feature_name.append(data[feature_name].columns[i] + '/' + data[feature_name].columns[j])
            new_feature_name.append(data[feature_name].columns[i] + '*' + data[feature_name].columns[j])
            new_feature_name.append(data[feature_name].columns[i] + '+' + data[feature_name].columns[j])
            new_feature_name.append(data[feature_name].columns[i] + '-' + data[feature_name].columns[j])
            new_feature.append(data[data[feature_name].columns[i] ] /data[data[feature_name].columns[j]])
            new_feature.append(data[data[feature_name].columns[i] ] *data[data[feature_name].columns[j]])
            new_feature.append(data[data[feature_name].columns[i] ] +data[data[feature_name].columns[j]])
            new_feature.append(data[data[feature_name].columns[i] ] -data[data[feature_name].columns[j]])

            #��ѡ�ģ�������Ҳд�빹������������ƽ��ֵƫ�������ƽ��ֵ��ƫ��������ĳ��� Ҳ����ģ�Ͷ�����ȫ��ȫ�������У����ǿ����õ�����ģ����
            #��Ҫע�����,Random Forest �� GBDT ��ģ�ͶԵ����ĺ����任������,��������svm������ģ����ʹ�ã�����Ҳ˵���ðɡ�
            # new_feature_name.append(data[feature_name].columns[i] + '-mean' + data[feature_name].columns[j])
            # new_feature_name.append(data[feature_name].columns[i] + '|-mean|' + data[feature_name].columns[j])
            # new_feature.append(data[feature_name].columns[i]-np.mean(data[feature_name].columns[i]))
            # new_feature.append(np.abs(data[feature_name].columns[i] - np.mean(data[feature_name].columns[i])))
    temp_data = pd.DataFrame(pd.concat(new_feature ,axis=1))
    temp_data.columns = new_feature_name
    data = pd.concat([data ,temp_data] ,axis=1).reset_index(drop=True)

    print(data.shape)

    return data.reset_index(drop=True)
def get_jiaocha_feature():
    print('-------  ��������  -------')
    test_data=pd.read_csv("../feature_data/suse_test_data.csv",header=0,sep=",")
    train_data=pd.read_csv("../feature_data/suse_all_train.csv",header=0,sep=",")
    test_data=test_data.drop(columns=['Unnamed: 0', 'cust_group'])
    train_data=train_data.drop(columns=['Unnamed: 0', 'cust_group', 'cust_id'])
    the_train_y=train_data.pop('y')
    train_data.rename(
        columns={'nan_(35.895, 51.0]':'nan1','nan_(51.0, 66.0]':'nan2','nan_(66.0, 81.0]':'nan3','nan_(81.0, 96.0]':'nan4','nan_(96.0, 111.0]':'nan5','nan_(111.0, 126.0]':'nan6','nan_(126.0, 141.0]':'nan7'},inplace=True)

    print(len(train_data.columns))
    print(train_data.columns)
    print('-------  ���ɽ���任����  -------')
    #��ѡ��ʽһ.����������������н���任����ֹ��ȱ©
    #ɸѡ���������ݽ��б任
    the_num_type_feat=[k for k in train_data.columns if k in [i for i in ['x_'+str(j) for j in range(1,96)]]]
    print('���õ���ֵ������:',the_num_type_feat)
    do_div_feature=get_division_feature(train_data,the_num_type_feat)
    do_div_feature.to_csv('../feature_data/division_feature.csv')

#������Ƥ���������ɸѡ
def compute_sing_feat_relat(train_data,target_label):
    print('���õ���������Խ���ɸѡ')
    the_sel_sing=[]
    for i in train_data.columns:
        print('������Ϊ��',i,'    ��Ӧ�����ϵ����',train_data[i].corr(target_label))
        if np.abs((train_data[i].corr(target_label)))>0.02:
            the_sel_sing.append(i)
    return the_sel_sing
#�Խ����������������������Խ���ɸѡ
def compute_corr_relat(div_feature,target_label,the_feature_set_type,feat_type):
    '''
    ��Ҫ˼���Ƿֱ��ÿ������������������  ԭʼ����֮�������ԣ�����Ŀ���ǩ֮�������� ����Ҫ��ÿ����������Ŀ��ֵ֮��������ϵ������0.2
    �����þ�������������˫������Ŀ��ֵ����Դ���0.24����Ҫ��ÿ�����ڵĻ��������ӵ����ϵ����Ҫ���0.05
    ���Ե������ϵ��ֵ��������ͬ��ɸѡЧ��
    '''
    the_sel_div = []
    for i in the_feature_set_type:
        the_div_corr=div_feature[i].corr(target_label)
        the_div_ori_corr_1=div_feature[i.split(feat_type)[0]].corr(target_label)
        the_div_ori_corr_2=div_feature[i.split(feat_type)[1]].corr(target_label)
        #��ú���ɸѡ�ж�
        if((np.abs(the_div_corr)>0.14)&(np.abs(the_div_corr-the_div_ori_corr_1)>0.04)&(np.abs(the_div_corr-the_div_ori_corr_2)>0.04)):
            the_sel_div.append(i)
    return the_sel_div

#����ɸѡ������
def do_method1_feature_select():
    print('-------  ��������  -------')
    train_data=pd.read_csv("../feature_data/suse_all_train.csv",header=0,sep=",")
    train_data=train_data.drop(columns=['Unnamed: 0', 'cust_group', 'cust_id'])
    train_data.rename(
        columns={'nan_(35.895, 51.0]': 'nan1', 'nan_(51.0, 66.0]': 'nan2', 'nan_(66.0, 81.0]': 'nan3',
                 'nan_(81.0, 96.0]': 'nan4', 'nan_(96.0, 111.0]': 'nan5', 'nan_(111.0, 126.0]': 'nan6',
                 'nan_(126.0, 141.0]': 'nan7'}, inplace=True)
    the_need_y=train_data.pop('y')

    div_feature=pd.read_csv('../feature_data/division_feature.csv')
    print(div_feature.shape)
    print('-------  �������ֱ任����������  -------')
    the_feature_set_jia=[]
    the_feature_set_jian = []
    the_feature_set_cheng = []
    the_feature_set_chu = []
    for i in div_feature.columns:
        if '+' in  i:
            the_feature_set_jia.append(i)
        elif '-' in  i:
            the_feature_set_jian.append(i)
        elif '*' in i:
            the_feature_set_cheng.append(i)
        elif '/' in i:
            the_feature_set_chu.append(i)
    print('-------  ���ÿ������ÿ������ɸѡ������������ɸѡ[��ѡ]������任�����ֱ�ɸѡ��  -------')
    sing_fea=compute_sing_feat_relat(train_data,the_need_y)
    the_add_fea_set=compute_corr_relat(div_feature,the_need_y,the_feature_set_jia,'+')
    the_jian_fea_set=compute_corr_relat(div_feature, the_need_y,the_feature_set_jian, '-')
    the_cheng_fea_set=compute_corr_relat(div_feature, the_need_y,the_feature_set_cheng, '*')
    the_chu_fea_set=compute_corr_relat(div_feature, the_need_y,the_feature_set_chu, '/')


    return sing_fea+the_add_fea_set+the_jian_fea_set+the_cheng_fea_set+the_chu_fea_set


def do_pred_by_fea(the_sel_fea_set):
    print('----  ��������  ---')
    train_data=pd.read_csv("../feature_data/suse_all_train.csv",header=0,sep=",")
    the_need_y=train_data.pop('y')

    div_feature=pd.read_csv('../feature_data/division_feature.csv')

    with h5py.File('do_pro_data/pro_data.hdf5') as f:
        y = f['y'][0:20000]
        X = f['X'][0:20000]
        sesu_pro_data = f['sesu_pro_data'][0:20000]

    print('----  ���Լ�����������(ͬ������������ʽ���������)  ---')
    test_data=pd.read_csv("../feature_data/suse_test_data.csv",header=0,sep=",")
    test_data=test_data.drop(columns=['Unnamed: 0', 'cust_group'])
    test_data.rename(
        columns={'nan_(33.893, 49.286]': 'nan1', 'nan_(49.286, 64.571]': 'nan2', 'nan_(64.571, 79.857]': 'nan3',
                 'nan_(79.857, 95.143]': 'nan4', 'nan_(95.143, 110.429]': 'nan5', 'nan_(110.429, 125.714]': 'nan6',
                 'nan_(125.714, 141.0]': 'nan7'}, inplace=True)
    the_num_type_feat=[k for k in train_data.columns if k in [i for i in ['x_'+str(j) for j in range(1,96)]]]
    print('���õ���ֵ������:',the_num_type_feat)
    div_test_data=get_division_feature(test_data,the_num_type_feat)
    sel_div_test_data=div_test_data[the_sel_fea_set]
    sel_div_test_data.to_csv('../feature_data/division_feature_test.csv')


    print('----  ѵ���ó����  ---')
    the_record_score, result_file = lgb_model(div_feature[the_sel_fea_set].values, the_need_y, sel_div_test_data.values)

    # Ч���ύ
    filepath = '../result/lgb_ʹ�������ϵ������ɸѡ_' + str(the_record_score) + '.csv'  # ����ƽ������
    # תΪarray
    print('result shape:', result_file.shape)

    sub_sample = pd.read_csv('../result/xgb_nan.csv')
    result = DataFrame()
    result['cust_id'] = sub_sample['cust_id']
    result['pred_prob'] = result_file
    result.to_csv(filepath, index=False, sep=",")


####################################################   ����������   ############################################
def score_lgb(X,y):
    N = 5
    skf = StratifiedKFold(n_splits=N, shuffle=True, random_state=2018)
    auc_cv = []
    pred_cv = []
    for k, (train_in, test_in) in enumerate(skf.split(X, y)):
        X_train, X_test, y_train, y_test = X[train_in], X[test_in], \
                                           y[train_in], y[test_in]

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
    # K������֤��ƽ������
    print('the cv information:')
    print(auc_cv)
    print('cv mean score', np.mean(auc_cv))
    res = np.array(pred_cv)
    print("�ܵĽ����", res.shape)
    return np.mean(auc_cv)

def score_xgb(X,y):
    print("start��********************************")
    start = time.time()

    auc_list = []
    pred_list = []

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2018)
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # ��������
        params = {'booster': 'gbtree',
                  'objective': 'binary:logistic',
                  'eta': 0.02,
                  'max_depth': 4,
                  'min_child_weight': 6,
                  'colsample_bytree': 0.7,
                  'subsample': 0.7,
                  # 'gamma':1,
                  # 'lambda ':1,
                  # 'alpha ':0��
                  'silent': 1
                  }
        params['eval_metric'] = ['auc']
        # ���ݽṹ
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dvali = xgb.DMatrix(X_test, label=y_test)
        evallist = [(dtrain, 'train'), (dvali, 'valid')]  # 'valid-auc' will be used for early stopping
        # ģ��train
        model = xgb.train(params, dtrain,
                          num_boost_round=2000,
                          evals=evallist,
                          early_stopping_rounds=100,
                          verbose_eval=100)
        # Ԥ����֤
        pred = model.predict(dvali, ntree_limit=model.best_ntree_limit)
        # ����
        auc = roc_auc_score(y_test, pred)
        print('...........................auc value:', auc)
        auc_list.append(auc)
    print('......................validate result mean :', np.mean(auc_list))

    end = time.time()
    print("......................run with time: ", (end - start) / 60.0)

    print("over:*********************************")
    mean_auc = np.mean(auc_list)
    print("mean auc:", mean_auc)
    return mean_auc

def two_model_ave_score(div_feature,the_cur_fea_list,the_need_y):
    lgb_score=score_lgb(div_feature[the_cur_fea_list].values,the_need_y)
    xgb_score=score_xgb(div_feature[the_cur_fea_list].values,the_need_y)
    return lgb_score,xgb_score

def div_feat_model_sel():
    #10000�������������ÿ����ʹ�ñ���̰�ķ�ȥɸѡ�Ļ���̫����ʵ��
    print('------------   ��������  --------------')
    train_data=pd.read_csv("../feature_data/suse_all_train.csv",header=0,sep=",")
    suse_train=train_data.drop(columns=['Unnamed: 0', 'cust_group', 'cust_id'])
    the_need_y = train_data.pop('y')
    div_feature=pd.read_csv('../feature_data/division_feature.csv')

    #��ȡ���н�����������
    the_div_fea_name=[]
    for i in div_feature.columns:
        if '+' in i:
            the_div_fea_name.append(i)
        elif '-' in i:
            the_div_fea_name.append(i)
        elif '*' in i:
            the_div_fea_name.append(i)
        elif '/' in i:
            the_div_fea_name.append(i)
    the_org_no_div_fea_set=[i for i in div_feature.columns if i not in the_div_fea_name]

    print('------------   ��ɸѡ  --------------')
    the_cur_fea_list=the_org_no_div_fea_set

    for fea in the_div_fea_name:
        #δ�����µĲο�����ʱ������ģ�͵��ۺ�cv
        bef_add_lgb_score,bef_add_xgb_score=two_model_ave_score(div_feature,the_cur_fea_list,the_need_y)
        # ������뵥�����������Ͳ�����ʱ�ıȽ�
        the_cur_fea_list.append(fea)
        aft_add_lgb_score,aft_add_xgb_score=two_model_ave_score(div_feature,the_cur_fea_list,the_need_y)

        if((float(bef_add_lgb_score)>float(aft_add_lgb_score))&(float(bef_add_xgb_score)>float(aft_add_xgb_score))):
            print('���������������')
        else:
            the_cur_fea_list.pop(fea)

    div_feature[the_cur_fea_list].to_csv('../feature_data/selt_by_model_tra_feat2.csv')
    print('------------   ɸѡЧ��չʾ��������֤  --------------')
    print('����ɸѡ�������Ϊ��',the_cur_fea_list)

    test_div_data=pd.read_csv('../feature_data/division_feature_test.csv')
    the_train_score, the_pred_label = lgb_model(div_feature[the_cur_fea_list].values, the_need_y, test_div_data[the_cur_fea_list].values)
    print('���ĵ÷�Ϊ��',the_train_score)



####################################################   ����������   ############################################
def sel_sort_feat():
    print('-----   ��������  ------')
    train_all_fea=pd.read_csv("../feature_data/process_have_sort_train.csv",header=0,sep=",")
    test_all_fea=pd.read_csv("../feature_data/process_have_sort_test.csv",header=0,sep=",")
    y = train_all_fea.pop('y')
    print('-----   �ռ�����������ԭʼ�����������룩  ------')
    sort_fea=[i for i in train_all_fea.columns if 'rank_' in i]
    the_ori_fea=list(set((train_all_fea.columns))-set(sort_fea))
    the_ori_fea=[i for i in the_ori_fea if i not in ['Unnamed: 0','cust_group','cust_id']]

    print('����������',sort_fea)
    print('ԭʼ������', the_ori_fea)
    print('-----   ÿ���������������ԣ��˴��趨���ַ�ʽ������ʽ�ͻ������ʽ��  ------')
    #��ʶΪ0��ʾʹ���滻ʽ������ ��ʾΪ1��ʾʹ�ø���ʽ����
    the_mothod_sel=1
    if the_mothod_sel==0:
        print('-----   ��ʼ�滻ʽɸѡ����  ------')
        the_sel_fea_set=the_ori_fea
        for i in sort_fea:
            the_ori_fea_name=i[5:]
            print('ԭʼ��������',the_sel_fea_set)
            print('���滻������', i)
            #ʹ��ԭ����������һ��
            the_before_lgb_score=score_lgb(train_all_fea[the_sel_fea_set].values,y)
            the_before_xgb_score=score_xgb(train_all_fea[the_sel_fea_set].values,y)
            # �������������ԭ��������
            the_sel_fea_set.append(i)
            the_sel_fea_set.remove(the_ori_fea_name)
            the_after_lgb_score=score_lgb(train_all_fea[the_sel_fea_set].values,y)
            the_after_xgb_score=score_xgb(train_all_fea[the_sel_fea_set].values, y)
            #�ж��Ƿ�������  �������Ƿ�����������������
            print('��ֵ��',the_after_lgb_score,the_before_lgb_score)
            if((the_after_lgb_score>the_before_lgb_score)&(the_after_xgb_score>the_before_xgb_score)):
                pass
            else:
                #�������������ķ��������Ӷ� ʵ�ֻ�ԭ��Ч��
                the_sel_fea_set.remove(i)
                the_sel_fea_set.append(the_ori_fea_name)
        print('�������ʽ�����õ�����������Ϊ��',the_sel_fea_set)
    else:
        print('-----   ��ʼ����ʽɸѡ����  ------')
        the_sel_fea_set = the_ori_fea
        for i in sort_fea:
            the_ori_fea_name = i[5:]
            # ʹ��ԭ����������һ��
            the_before_lgb_score = score_lgb(train_all_fea[the_sel_fea_set].values, y)
            the_before_xgb_score = score_xgb(train_all_fea[the_sel_fea_set].values, y)
            # �������������ԭ��������
            the_sel_fea_set.append(i)
            the_after_lgb_score = score_lgb(train_all_fea[the_sel_fea_set].values, y)
            the_after_xgb_score = score_xgb(train_all_fea[the_sel_fea_set].values, y)
            # �ж��Ƿ�������  �������Ƿ�����������������
            if ((the_after_lgb_score > the_before_lgb_score) & (the_after_xgb_score > the_before_xgb_score)):
                pass
            else:
                # �������������ķ��������Ӷ� ʵ�ֻ�ԭ��Ч��
                the_sel_fea_set.remove(i)
        print('��������ʽ�����õ�����������Ϊ��', the_sel_fea_set)
        pass

    the_train_score, the_pred_label = lgb_model(train_all_fea[the_sel_fea_set].values, y, test_all_fea[the_sel_fea_set].values)
    print('���ĵ÷�Ϊ��',the_train_score)

    # Ч���ύ
    filepath = '../result/lgb_ʹ��ģ�ͷ�ʽ��������ɸѡ_' + str(the_train_score) + '.csv'  # ����ƽ������
    # תΪarray
    print('result shape:', the_pred_label.shape)

    sub_sample = pd.read_csv('../result/xgb_nan.csv')
    result = DataFrame()
    result['cust_id'] = sub_sample['cust_id']
    result['pred_prob'] = the_pred_label
    result.to_csv(filepath, index=False, sep=",")
if __name__=='__main__':

    print('---------------------   ����һ.���ϵ��������ԣ���ʽɸѡ��  -----------------------')
    #������������
    #get_jiaocha_feature()
    #��������ɸѡ  ѵ������������ѡ��
    the_sel_fea_set=do_method1_feature_select()

    print('ɸѡ�µ������У�', the_sel_fea_set)
    for i in the_sel_fea_set:
        print(i,end=',')
    #
    #����ɸѡ�õ�������ѵ������֤    ʹ�ú���ȥ��֤ģ��Ч��
    do_pred_by_fea(the_sel_fea_set)
    #pass

    print('---------------------   ������.�𲽼��뷨��̰�ķ�������˫ģ����Ҫ��  -----------------------')
    #div_feat_model_sel()
    print('---------------------   ������.����������ʹ��ģ�ͽ���ɸѡ(�����滻��ʽ��һ����ԭ���������һ���Ǹ���ʽ)  -----------------------')
    #sel_sort_feat()

'''
����һ��
   �����Ļ���ʹ������Լ����һ�ַ����ĵ÷ֽ���� 0.813

'''
