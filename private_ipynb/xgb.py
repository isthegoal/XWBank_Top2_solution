# coding=gbk
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import time
import numpy as np
import h5py
def xgb_method(X,y,test_data):
    print("start：********************************")
    start = time.time()

    auc_list = []
    pred_list = []

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2018)
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        #参数设置
        params = {'booster': 'gbtree',
                  'objective':'binary:logistic',
                  'eta': 0.02,
                  'max_depth': 4,
                  'min_child_weight': 6,
                  'colsample_bytree': 0.7,
                  'subsample': 0.7,
                  #'gamma':1,
                  #'lambda ':1,
                  #'alpha ':0，
                  'silent':1
                  }
        params['eval_metric'] = ['auc']
        # 数据结构
        dtrain = xgb.DMatrix(X_train, label = y_train)
        dvali = xgb.DMatrix(X_test,label = y_test)
        evallist  = [(dtrain,'train'),(dvali,'valid')]  # 'valid-auc' will be used for early stopping
        # 模型train
        model = xgb.train(params, dtrain,
                          num_boost_round=2000,
                          evals = evallist,
                          early_stopping_rounds = 100,
                          verbose_eval=100)
        # 预测验证
        pred = model.predict(dvali, ntree_limit = model.best_ntree_limit)
        # 评估
        auc = roc_auc_score(y_test,pred)
        print('...........................auc value:',auc)
        auc_list.append(auc)
        # 预测
        dtest = xgb.DMatrix(test_data)
        pre = model.predict(dtest,ntree_limit = model.best_ntree_limit)
        pred_list.append(pre)

    print('......................validate result mean :',np.mean(auc_list))

    end = time.time()
    print("......................run with time: ",(end - start) / 60.0)

    print("over:*********************************")
    mean_auc = np.mean(auc_list)
    print("mean auc:",mean_auc)

    res = np.array(pred_list)
    print("总的结果：", res.shape)
    # 最后结果，mean，max，min  有尝试，求mean是最好的
    r = res.mean(axis=0)
    print('result shape:', r.shape)
    return np.mean(auc_list), r

    # filepath = 'result/xgb_'+ str(mean_auc)+'.csv' # 线下平均分数
    # # 转为array
    # res =  np.array(pred_list)
    # print("5折结果：",res.shape)
    #
    # # 最后结果，mean，max，min
    # r = res.mean(axis = 0)
    # print('result shape:',r.shape)
    #
    # result = DataFrame()
    # result['cust_id'] = test_id
    # result['pred_prob'] = r
    # result.to_csv(filepath,index=False,sep=",")
if __name__=='__main__':
    with h5py.File('do_pro_data/pro_data.hdf5') as f:
        y=f['y'][0:20000]
        X=f['X'][0:20000]
        test_data = f['test_data'][0:20000]
    print(len(y))
    print(len(test_data))
    print(len(X))

    #################   启动模型  ##################
    xgb_method(X,y,test_data)