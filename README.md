# 竞赛地址
http://www.dcjingsai.com/common/cmpt/%E8%A5%BF%E5%8D%97%E8%B4%A2%E7%BB%8F%E5%A4%A7%E5%AD%A6%E2%80%9C%E6%96%B0%E7%BD%91%E9%93%B6%E8%A1%8C%E6%9D%AF%E2%80%9D%E6%95%B0%E6%8D%AE%E7%A7%91%E5%AD%A6%E7%AB%9E%E8%B5%9B_%E7%AB%9E%E8%B5%9B%E4%BF%A1%E6%81%AF.html
# 说明
   (1).具体方案请详见 《新网银行杯竞赛报告.pdf》
   
   (2).针对本赛题的数据分析过程在 analyzeData.ipynb 中这里不做具体介绍
   
   (3).比赛中主要的构建特征和处理过程在 train_lgb.ipynb中
   
   (4).针对无标签数据使用了如下的方案，具体代码请详见 private_ipynb/SeSu_learning.py，内部有大量的注释。
       
       *使用模型加阈值加分组因素   
       *使用knn针对重要的样本、将重要的无缺失的特征进行训练，之后一个个带进去尝试无标签预测。
       *使用外加的方式不断跑模型筛选特征【对于交叉特征 和 排序特征】
       *对训练集中的正例样本进行复制扩展，并在扩展出的样本中加入高斯扰动。
       
   (5).此外也尝试了多特征群、多模型融合的方案，尝试了如何组合方案，代码请详见，单后期因为炸榜，如下方案训练的模型并没有做B榜提交方案
      
       *特征群构建：
           1.初始特征+补缺失+统计特征
           2.初始特征+补缺失+统计特征+排序特征
           3.初始特征+筛选的统计特征+筛选的排序特征
       *备用的参与模型融合的模型(注重模型运算上的差异)：
           lgb、xgb、cab
