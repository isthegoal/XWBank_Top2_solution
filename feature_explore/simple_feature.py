import pandas as pd
import numpy as np
from pandas import DataFrame as DF
import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas import Series
import time
import matplotlib
import matplotlib.pyplot as plt
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


    temp_data = DF(pd.concat(new_feature ,axis=1))
    temp_data.columns = new_feature_name
    data = pd.concat([data ,temp_data] ,axis=1).reset_index(drop=True)

    print(data.shape)

    return data.reset_index(drop=True)


def get_square_feature(data, feature_name):
    # 对特征进行二项变换，开放等方式， 并将处理后的特征名记录
    new_feature = []
    new_feature_name = []
    for i in range(len(data[feature_name].columns)):
        new_feature_name.append(data[feature_name].columns[i] + '**2')
        new_feature_name.append(data[feature_name].columns[i] + '**1/2')
        new_feature.append(data[data[feature_name].columns[i]] ** 2)
        new_feature.append(data[data[feature_name].columns[i]] ** (1 / 2))

    temp_data = DF(pd.concat(new_feature, axis=1))
    temp_data.columns = new_feature_name
    data = pd.concat([data, temp_data], axis=1).reset_index(drop=True)

    print(data.shape)

    return data.reset_index(drop=True)

if __name__=='__main__':
    ###############################   提取交叉特征   #############################
    ########     挑选重要的特征进行交叉变换，这是一种思路，我看别人利用这一方法是有很大效果提升的，但是在我这里使用了之后线上有明显的下降，真是说不好###

    train = pd.read_csv( '../feature_data/train_feature.csv')
    y_target = pd.read_csv( '../feature_data/train_label.csv',header=None)
    # train['x_95+x_93'] = train['x_95'] + train['x_42']
    # train['x_80+x_93'] = train['x_80'] + train['x_42']
    # train['x_63*x_93'] = train['x_95'] - train['x_42']
    # train['x_63*x_93'] = train['x_80'] ** 1 / 2



    print('特征间四项交叉')
    feature_name = ['x_80','x_95','x_91','x_2','x_42','x_48']
    train_data = get_division_feature(train, feature_name)
    #test_data = get_division_feature(test, feature_name)

    print('特征间方关系变换')

    train_data = get_square_feature(train_data, feature_name)

    print(train.columns)

    train_data=train.drop(columns=['Unnamed: 0'],axis=1)
    X=train_data.values
    y=y_target[1].values
    print(train_data.columns)

    ###############################   特征重要度观察   #############################
    features=train_data.columns
    # 采取分层采样
    from sklearn.model_selection import StratifiedKFold
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    import operator
    import time

    print("start：********************************")
    start = time.time()

    print('......................Start train all data .......................')

    # 采取分层采样
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score

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
        # K交叉验证的平均分数
    print('the cv information:')
    print(auc_cv)
    print('cv mean score', np.mean(auc_cv))

    end = time.time()
    print("......................run with time: ", (end - start) / 60.0)
    print("over:*********************************")



    print(gbm.feature_importance())
    print(features)
    #进一步的进行展示
    df = pd.DataFrame({'feature': features, 'importance': gbm.feature_importance()}).sort_values(by='importance',
                                                                                                   ascending=False)  # 降序
    df.loc[df['importance'] >6, :].plot(kind='barh', x='feature', y='importance', legend=False, figsize=(15, 20))
    plt.title('lgb Feature Importance')
    plt.xlabel('relative importance')
    plt.show()