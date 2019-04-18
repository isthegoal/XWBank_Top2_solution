# coding=gbk
# 开始训练     这个效果会更好点，cv效果不可信

# 采取分层采样
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import time
import numpy as np
import h5py
import pickle
def lgb_model(X,y,test_data):
    print("start：********************************")
    start = time.time()

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
if __name__=='__main__':
    with h5py.File('do_pro_data/pro_data.hdf5') as f:
        y=f['y'][0:20000]
        X=f['X'][0:20000]
        test_data = f['test_data'][0:20000]
    print(len(y))
    print(len(test_data))
    print(len(X))

    #################   启动模型  ##################
    lgb_model(X,y,test_data)