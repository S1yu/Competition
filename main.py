# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import pandas as pd
import  os


pwd=os.getcwd()
train = pd.read_csv(pwd+'/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv(pwd+'/house-prices-advanced-regression-techniques/test.csv')

# .coor 求相关系数 默认为标准相关系数
corr = train[train.columns].corr()['SalePrice'][:].sort_values(ascending=True).to_frame()
corr=corr.drop(corr[corr.SalePrice > 0.99].index)
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import proplot as pplt

# 此函数的两个整数参数指定子图网格的行数和列数。该函数返回一个图形对象和一个包含等于nrows * ncols的轴对象的元组
fig, ax=plt.subplots(figsize=(9,9))
fig.patch.set_facecolor('black')
ax.patch.set_facecolor('black')
ax.barh(corr.index,corr.SalePrice,align="center",color=np.where(corr["SalePrice"]<0,"crimson","#89CFF0"))

ax.tick_params(axis="both",which="major",labelsize=8)
ax.yaxis.set_label_coords(0, 0)
ax.grid(color='white', linewidth=2)
ax.xaxis.set_ticks_position('none')
ax.yaxis.set_ticks_position('none')

for i in ['top', 'bottom', 'left', 'right']:
    ax.spines[i].set_visible(False)

ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')

mpl.rcParams['font.family'] = 'Source Sans Pro'
plt.text(-0.12, 39, "Correlation", size=24, color="grey", fontweight="bold");
plt.text(0.135, 39, "of", size=24, color="grey");
plt.text(0.185, 39, "SalePrice", size=24, color="#89CFF0", fontweight="bold");
plt.text(0.4, 39, "to", size=24, color="grey");
plt.text(0.452, 39, "Other Features", size=24, color="grey", fontweight="bold");
plt.figure(figsize=(10,10))


print('Training Shape:', train.shape)
print('Test Shape:', test.shape)
train_id = train['Id']
test_id = test['Id']
del train['Id']
del test['Id']


top_corr = corr['SalePrice'].sort_values(ascending=False).head(10).index
top_corr = top_corr.union(['SalePrice'])
# 所以变量 两两之间的关系，线性、非线性、相关
sns.pairplot(train[top_corr][:10])

plt.show()











