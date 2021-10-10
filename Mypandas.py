import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
pwd=os.getcwd()
train = pd.read_csv(pwd+'/house-prices-advanced-regression-techniques/train.csv')

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
y=train5["SalePrice"].to_frame()

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

# 将偏移的变量 转换正常分布
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax

for col in skew_cols.index :
    X[col]=boxcox1p(X[col], boxcox_normmax(X[col] + 1))


























