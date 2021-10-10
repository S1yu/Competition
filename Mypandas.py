import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
pwd=os.getcwd()
train = pd.read_csv(pwd+'/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv(pwd+'/house-prices-advanced-regression-techniques/train.csv')

#train.describe() 统计数据
#print(train.describe())

train5=train.head()
 # .sort_values 排序
# print(train5.sort_values(by="SalePrice"))
print(train5.describe())
#pd.isnull 找出缺少的页
#print( pd.isnull(train5['MSZoning']))

# sns.lineplot(data=fifa_data) 折线图
coor= train5[train5.columns].corr()['SalePrice'].sort_values().to_frame()
print(coor['SalePrice'])

#
fig ,ax = plt.subplots(figsize=(9,9))
fig.patch.set_facecolor('white')
ax.patch.set_facecolor('white')

ax.barh(coor.index,coor['SalePrice'],align='center')

ax.tick_params(axis='x', colors='black')
ax.tick_params(axis='y', colors='black')
plt.xlim((-1,1))


plt.show()


#-------------------------------
train_id=train5.Id
del train5['Id']

X=train5.drop("SalePrice",axis=1)
print(X)
Y=train5["SalePrice"].to_frame()

print("Missing value \n")
print(X.isnull().sum)
# skew  求偏度
skew_vals = X.skew()
skew_cols = (skew_vals
             .sort_values(ascending=False)
             .to_frame()
             .rename(columns={0:'Skew'})
             .query('abs(Skew) > {0}'.format(0.5)))
print(skew_cols)

# 将偏移的变量 转换正常分布  使用boxcox变换 具有线性关系
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax

for col in skew_cols.index :
    X[col]=boxcox1p(X[col], boxcox_normmax(X[col] + 1))

fig,ax=plt.subplots(figsize=(5,5))
fig.patch.set_facecolor('white')
ax.patch.set_facecolor('white')
# 绘制 Y的密度图
sns.histplot(Y['SalePrice'], stat='density', linewidth=0, color = '#ff7f50', kde=True, alpha=0.3);

X['train'] = 1
test['train'] = 0
# 将测试和训练样本一起预处理
df= pd.concat([test,X])
print("\n",type(df))


# 转换 y = log(1+y)
Y['SalePrice']=np.log1p(Y["SalePrice"])

categ_cols=df.dtypes[df.dtypes==np.dtype]
categ_cols = categ_cols.index.tolist()
df_enc = pd.get_dummies(df, columns=categ_cols, drop_first=True)


X = df_enc[df_enc['train']==1]
test = df_enc[df_enc['train']==0]
X.drop(['train'], axis=1, inplace=True)
test.drop(['train'], axis=1, inplace=True)


from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error



X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=12345)










