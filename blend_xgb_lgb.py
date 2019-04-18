# coding=gbk
import pandas as pd

SSE = pd.read_csv('./result/lgb_根据group数量关系方式2_0.8203307010567176.csv')
SSE=SSE[['cust_id','pred_prob']]
TX = pd.read_csv('./result/lgb_使用knn做分类生成300正例方式3_0.8202148008953563.csv')
TX=TX[['cust_id','pred_prob']]


df = pd.merge(SSE,TX,on= 'cust_id')
df.columns = ['cust_id','T1','T2']
print(df.head())
pred_prob = 0.5 * df['T1'] + 0.5 * df['T2']
print(pred_prob.head())
sub = pd.DataFrame()
sub['cust_id'] = df['cust_id']
sub['pred_prob']= pred_prob
#pd.read_csv('../midern_data/blend_file/final_blended.csv')
sub.to_csv('./result/blended.csv', index=False)
