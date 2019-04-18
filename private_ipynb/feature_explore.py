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
这里准备进行几种特征探索的方案：
     方案一.对于交叉特征使用相关性计算增益计算的方式来筛选特征跑模型（计算相关性不需要模型，依靠相关系数）
     方案二.对于交叉特征使用特征逐步加入，计算三模型上增益的方式来筛选特征跑模型（使用三个模型吧）
     方案三.单独的同上，我们针对排序特征使用同上的方法进行筛选（模型特征重要度计算）
     方案四.模拟退火加贪心筛选排序特征、选交叉特征 （后续尝试）
一点思考：
    *在做交叉变换时候不要使用类别特征，这种特征是没有什么意义的
'''
#################################################  方案一函数   ##################################################
def get_division_feature(data ,feature_name):
    # 创造出特征之间 进行四则变换的特征， 每两两特征之间进行变化，  并且进行变换之后把变化后的 新特证名记录下来，（知道了吧。不想之前直接一起，
    # 特征名都完全丢失了
    new_feature = []
    new_feature_name = []
    for i in range(len(data[feature_name].columns ) -1):
        for j in range( i +1 ,len(data[feature_name].columns)):
            # 保存新创建的特征值和特征名
            new_feature_name.append(data[feature_name].columns[i] + '/' + data[feature_name].columns[j])
            new_feature_name.append(data[feature_name].columns[i] + '*' + data[feature_name].columns[j])
            new_feature_name.append(data[feature_name].columns[i] + '+' + data[feature_name].columns[j])
            new_feature_name.append(data[feature_name].columns[i] + '-' + data[feature_name].columns[j])
            new_feature.append(data[data[feature_name].columns[i] ] /data[data[feature_name].columns[j]])
            new_feature.append(data[data[feature_name].columns[i] ] *data[data[feature_name].columns[j]])
            new_feature.append(data[data[feature_name].columns[i] ] +data[data[feature_name].columns[j]])
            new_feature.append(data[data[feature_name].columns[i] ] -data[data[feature_name].columns[j]])

            #可选的，我这里也写入构建样本与组内平均值偏差、样本与平均值的偏差绝对量的程序， 也许树模型对这种全加全减不敏感，但是可以用到其他模型中
            #需要注意的是,Random Forest 和 GBDT 等模型对单调的函数变换不敏感,所以用于svm这样的模型中使用，但是也说不好吧。
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
    print('-------  加载数据  -------')
    test_data=pd.read_csv("../feature_data/suse_test_data.csv",header=0,sep=",")
    train_data=pd.read_csv("../feature_data/suse_all_train.csv",header=0,sep=",")
    test_data=test_data.drop(columns=['Unnamed: 0', 'cust_group'])
    train_data=train_data.drop(columns=['Unnamed: 0', 'cust_group', 'cust_id'])
    the_train_y=train_data.pop('y')
    train_data.rename(
        columns={'nan_(35.895, 51.0]':'nan1','nan_(51.0, 66.0]':'nan2','nan_(66.0, 81.0]':'nan3','nan_(81.0, 96.0]':'nan4','nan_(96.0, 111.0]':'nan5','nan_(111.0, 126.0]':'nan6','nan_(126.0, 141.0]':'nan7'},inplace=True)

    print(len(train_data.columns))
    print(train_data.columns)
    print('-------  生成交叉变换特征  -------')
    #可选方式一.这里对所有特征进行交叉变换，防止有缺漏
    #筛选交叉型数据进行变换
    the_num_type_feat=[k for k in train_data.columns if k in [i for i in ['x_'+str(j) for j in range(1,96)]]]
    print('可用的数值型特征:',the_num_type_feat)
    do_div_feature=get_division_feature(train_data,the_num_type_feat)
    do_div_feature.to_csv('../feature_data/division_feature.csv')

#单特征皮尔顿相关性筛选
def compute_sing_feat_relat(train_data,target_label):
    print('利用单特征相关性进行筛选')
    the_sel_sing=[]
    for i in train_data.columns:
        print('特征名为：',i,'    对应的相关系数：',train_data[i].corr(target_label))
        if np.abs((train_data[i].corr(target_label)))>0.02:
            the_sel_sing.append(i)
    return the_sel_sing
#对交叉特征根据类别利用相关性进行筛选
def compute_corr_relat(div_feature,target_label,the_feature_set_type,feat_type):
    '''
    主要思考是分别对每个交叉特征计算其与  原始特征之间的相关性，其与目标标签之间的相关性 这里要求每个单因子与目标值之间的相关性系数大于0.2
    并且让经过经过交叉后的双因子与目标值相关性大于0.24，且要比每个对于的基础单因子的相关系数都要大出0.05
    可以调整相关系数值来产生不同的筛选效果
    '''
    the_sel_div = []
    for i in the_feature_set_type:
        the_div_corr=div_feature[i].corr(target_label)
        the_div_ori_corr_1=div_feature[i.split(feat_type)[0]].corr(target_label)
        the_div_ori_corr_2=div_feature[i.split(feat_type)[1]].corr(target_label)
        #获得后做筛选判断
        if((np.abs(the_div_corr)>0.14)&(np.abs(the_div_corr-the_div_ori_corr_1)>0.04)&(np.abs(the_div_corr-the_div_ori_corr_2)>0.04)):
            the_sel_div.append(i)
    return the_sel_div

#交叉筛选主函数
def do_method1_feature_select():
    print('-------  加载数据  -------')
    train_data=pd.read_csv("../feature_data/suse_all_train.csv",header=0,sep=",")
    train_data=train_data.drop(columns=['Unnamed: 0', 'cust_group', 'cust_id'])
    train_data.rename(
        columns={'nan_(35.895, 51.0]': 'nan1', 'nan_(51.0, 66.0]': 'nan2', 'nan_(66.0, 81.0]': 'nan3',
                 'nan_(81.0, 96.0]': 'nan4', 'nan_(96.0, 111.0]': 'nan5', 'nan_(111.0, 126.0]': 'nan6',
                 'nan_(126.0, 141.0]': 'nan7'}, inplace=True)
    the_need_y=train_data.pop('y')

    div_feature=pd.read_csv('../feature_data/division_feature.csv')
    print(div_feature.shape)
    print('-------  分离四种变换出来的特征  -------')
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
    print('-------  针对每种类型每个特征筛选（单特征尝试筛选[可选]、四项变换特征分别筛选）  -------')
    sing_fea=compute_sing_feat_relat(train_data,the_need_y)
    the_add_fea_set=compute_corr_relat(div_feature,the_need_y,the_feature_set_jia,'+')
    the_jian_fea_set=compute_corr_relat(div_feature, the_need_y,the_feature_set_jian, '-')
    the_cheng_fea_set=compute_corr_relat(div_feature, the_need_y,the_feature_set_cheng, '*')
    the_chu_fea_set=compute_corr_relat(div_feature, the_need_y,the_feature_set_chu, '/')


    return sing_fea+the_add_fea_set+the_jian_fea_set+the_cheng_fea_set+the_chu_fea_set


def do_pred_by_fea(the_sel_fea_set):
    print('----  加载数据  ---')
    train_data=pd.read_csv("../feature_data/suse_all_train.csv",header=0,sep=",")
    the_need_y=train_data.pop('y')

    div_feature=pd.read_csv('../feature_data/division_feature.csv')

    with h5py.File('do_pro_data/pro_data.hdf5') as f:
        y = f['y'][0:20000]
        X = f['X'][0:20000]
        sesu_pro_data = f['sesu_pro_data'][0:20000]

    print('----  测试集的特征构建(同样的造特征方式来产生结果)  ---')
    test_data=pd.read_csv("../feature_data/suse_test_data.csv",header=0,sep=",")
    test_data=test_data.drop(columns=['Unnamed: 0', 'cust_group'])
    test_data.rename(
        columns={'nan_(33.893, 49.286]': 'nan1', 'nan_(49.286, 64.571]': 'nan2', 'nan_(64.571, 79.857]': 'nan3',
                 'nan_(79.857, 95.143]': 'nan4', 'nan_(95.143, 110.429]': 'nan5', 'nan_(110.429, 125.714]': 'nan6',
                 'nan_(125.714, 141.0]': 'nan7'}, inplace=True)
    the_num_type_feat=[k for k in train_data.columns if k in [i for i in ['x_'+str(j) for j in range(1,96)]]]
    print('可用的数值型特征:',the_num_type_feat)
    div_test_data=get_division_feature(test_data,the_num_type_feat)
    sel_div_test_data=div_test_data[the_sel_fea_set]
    sel_div_test_data.to_csv('../feature_data/division_feature_test.csv')


    print('----  训练得出结果  ---')
    the_record_score, result_file = lgb_model(div_feature[the_sel_fea_set].values, the_need_y, sel_div_test_data.values)

    # 效果提交
    filepath = '../result/lgb_使用相关性系数特征筛选_' + str(the_record_score) + '.csv'  # 线下平均分数
    # 转为array
    print('result shape:', result_file.shape)

    sub_sample = pd.read_csv('../result/xgb_nan.csv')
    result = DataFrame()
    result['cust_id'] = sub_sample['cust_id']
    result['pred_prob'] = result_file
    result.to_csv(filepath, index=False, sep=",")


####################################################   方案二函数   ############################################
def score_lgb(X,y):
    N = 5
    skf = StratifiedKFold(n_splits=N, shuffle=True, random_state=2018)
    auc_cv = []
    pred_cv = []
    for k, (train_in, test_in) in enumerate(skf.split(X, y)):
        X_train, X_test, y_train, y_test = X[train_in], X[test_in], \
                                           y[train_in], y[test_in]

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
    # K交叉验证的平均分数
    print('the cv information:')
    print(auc_cv)
    print('cv mean score', np.mean(auc_cv))
    res = np.array(pred_cv)
    print("总的结果：", res.shape)
    return np.mean(auc_cv)

def score_xgb(X,y):
    print("start：********************************")
    start = time.time()

    auc_list = []
    pred_list = []

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2018)
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # 参数设置
        params = {'booster': 'gbtree',
                  'objective': 'binary:logistic',
                  'eta': 0.02,
                  'max_depth': 4,
                  'min_child_weight': 6,
                  'colsample_bytree': 0.7,
                  'subsample': 0.7,
                  # 'gamma':1,
                  # 'lambda ':1,
                  # 'alpha ':0，
                  'silent': 1
                  }
        params['eval_metric'] = ['auc']
        # 数据结构
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dvali = xgb.DMatrix(X_test, label=y_test)
        evallist = [(dtrain, 'train'), (dvali, 'valid')]  # 'valid-auc' will be used for early stopping
        # 模型train
        model = xgb.train(params, dtrain,
                          num_boost_round=2000,
                          evals=evallist,
                          early_stopping_rounds=100,
                          verbose_eval=100)
        # 预测验证
        pred = model.predict(dvali, ntree_limit=model.best_ntree_limit)
        # 评估
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
    #10000个创造的特征，每个都使用暴力贪心法去筛选的话，太不务实了
    print('------------   载入数据  --------------')
    train_data=pd.read_csv("../feature_data/suse_all_train.csv",header=0,sep=",")
    suse_train=train_data.drop(columns=['Unnamed: 0', 'cust_group', 'cust_id'])
    the_need_y = train_data.pop('y')
    div_feature=pd.read_csv('../feature_data/division_feature.csv')

    #获取所有交叉特征列名
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

    print('------------   逐步筛选  --------------')
    the_cur_fea_list=the_org_no_div_fea_set

    for fea in the_div_fea_name:
        #未加入新的参考特征时的两个模型的综合cv
        bef_add_lgb_score,bef_add_xgb_score=two_model_ave_score(div_feature,the_cur_fea_list,the_need_y)
        # 计算加入单个交叉特征和不加入时的比较
        the_cur_fea_list.append(fea)
        aft_add_lgb_score,aft_add_xgb_score=two_model_ave_score(div_feature,the_cur_fea_list,the_need_y)

        if((float(bef_add_lgb_score)>float(aft_add_lgb_score))&(float(bef_add_xgb_score)>float(aft_add_xgb_score))):
            print('这个特征留下来吧')
        else:
            the_cur_fea_list.pop(fea)

    div_feature[the_cur_fea_list].to_csv('../feature_data/selt_by_model_tra_feat2.csv')
    print('------------   筛选效果展示并线下验证  --------------')
    print('讲过筛选后的特征为：',the_cur_fea_list)

    test_div_data=pd.read_csv('../feature_data/division_feature_test.csv')
    the_train_score, the_pred_label = lgb_model(div_feature[the_cur_fea_list].values, the_need_y, test_div_data[the_cur_fea_list].values)
    print('最后的得分为：',the_train_score)



####################################################   方案三函数   ############################################
def sel_sort_feat():
    print('-----   加载数据  ------')
    train_all_fea=pd.read_csv("../feature_data/process_have_sort_train.csv",header=0,sep=",")
    test_all_fea=pd.read_csv("../feature_data/process_have_sort_test.csv",header=0,sep=",")
    y = train_all_fea.pop('y')
    print('-----   收集排序特征和原始特征名（分离）  ------')
    sort_fea=[i for i in train_all_fea.columns if 'rank_' in i]
    the_ori_fea=list(set((train_all_fea.columns))-set(sort_fea))
    the_ori_fea=[i for i in the_ori_fea if i not in ['Unnamed: 0','cust_group','cust_id']]

    print('排序特征：',sort_fea)
    print('原始特征：', the_ori_fea)
    print('-----   每个排序特征做尝试（此处设定两种方式，附加式和基础替代式）  ------')
    #标识为0表示使用替换式方案、 表示为1表示使用附加式方案
    the_mothod_sel=1
    if the_mothod_sel==0:
        print('-----   开始替换式筛选方案  ------')
        the_sel_fea_set=the_ori_fea
        for i in sort_fea:
            the_ori_fea_name=i[5:]
            print('原始的特征：',the_sel_fea_set)
            print('做替换的特征', i)
            #使用原来的特征跑一次
            the_before_lgb_score=score_lgb(train_all_fea[the_sel_fea_set].values,y)
            the_before_xgb_score=score_xgb(train_all_fea[the_sel_fea_set].values,y)
            # 用排序特征替代原来的特征
            the_sel_fea_set.append(i)
            the_sel_fea_set.remove(the_ori_fea_name)
            the_after_lgb_score=score_lgb(train_all_fea[the_sel_fea_set].values,y)
            the_after_xgb_score=score_xgb(train_all_fea[the_sel_fea_set].values, y)
            #判断是否有增益  来决定是否把这个特征保留下来
            print('分值：',the_after_lgb_score,the_before_lgb_score)
            if((the_after_lgb_score>the_before_lgb_score)&(the_after_xgb_score>the_before_xgb_score)):
                pass
            else:
                #进行特征操作的反操作，从而 实现还原的效果
                the_sel_fea_set.remove(i)
                the_sel_fea_set.append(the_ori_fea_name)
        print('经过替代式方案得到的特征列名为：',the_sel_fea_set)
    else:
        print('-----   开始附加式筛选方案  ------')
        the_sel_fea_set = the_ori_fea
        for i in sort_fea:
            the_ori_fea_name = i[5:]
            # 使用原来的特征跑一次
            the_before_lgb_score = score_lgb(train_all_fea[the_sel_fea_set].values, y)
            the_before_xgb_score = score_xgb(train_all_fea[the_sel_fea_set].values, y)
            # 用排序特征替代原来的特征
            the_sel_fea_set.append(i)
            the_after_lgb_score = score_lgb(train_all_fea[the_sel_fea_set].values, y)
            the_after_xgb_score = score_xgb(train_all_fea[the_sel_fea_set].values, y)
            # 判断是否有增益  来决定是否把这个特征保留下来
            if ((the_after_lgb_score > the_before_lgb_score) & (the_after_xgb_score > the_before_xgb_score)):
                pass
            else:
                # 进行特征操作的反操作，从而 实现还原的效果
                the_sel_fea_set.remove(i)
        print('经过附加式方案得到的特征列名为：', the_sel_fea_set)
        pass

    the_train_score, the_pred_label = lgb_model(train_all_fea[the_sel_fea_set].values, y, test_all_fea[the_sel_fea_set].values)
    print('最后的得分为：',the_train_score)

    # 效果提交
    filepath = '../result/lgb_使用模型方式对排序做筛选_' + str(the_train_score) + '.csv'  # 线下平均分数
    # 转为array
    print('result shape:', the_pred_label.shape)

    sub_sample = pd.read_csv('../result/xgb_nan.csv')
    result = DataFrame()
    result['cust_id'] = sub_sample['cust_id']
    result['pred_prob'] = the_pred_label
    result.to_csv(filepath, index=False, sep=",")
if __name__=='__main__':

    print('---------------------   方案一.相关系数（相关性）方式筛选法  -----------------------')
    #构建交叉特征
    #get_jiaocha_feature()
    #交叉特征筛选  训练集上做特诊选择
    the_sel_fea_set=do_method1_feature_select()

    print('筛选下的特征有：', the_sel_fea_set)
    for i in the_sel_fea_set:
        print(i,end=',')
    #
    #利用筛选得到的特征训练和验证    使用函数去验证模型效果
    do_pred_by_fea(the_sel_fea_set)
    #pass

    print('---------------------   方案二.逐步加入法（贪心法）计算双模型重要度  -----------------------')
    #div_feat_model_sel()
    print('---------------------   方案三.对排序特征使用模型进行筛选(两种替换方式，一种是原型替代，另一种是附加式)  -----------------------')
    #sel_sort_feat()

'''
方案一：
   初步的话，使用相关性计算第一种方案的得分结果是 0.813

'''
