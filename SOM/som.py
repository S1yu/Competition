import math
from itertools import cycle

from minisom import MiniSom
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, auc, accuracy_score, precision_score
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn import datasets
MAX_ITER = 400

class SOM:
    def get_data(self):
        iris = datasets.load_iris()
        feature_names = iris.feature_names
        class_names = iris.target_names

        X = iris.data
        y = iris.target
        #7：3划分
        return train_test_split(X,y,test_size=0.3,random_state=0)
    def get_som(self):
        train_data, test_data, train_labels, test_labels = self.get_data()
        N = train_data.shape[0]  #样本数量
        M = train_data.shape[1]  #维度/特征数量

        size = math.ceil(np.sqrt(5 * np.sqrt(N)))  # 经验公式：决定输出层尺寸

        som = MiniSom(size, size, M, sigma=3, learning_rate=0.5,neighborhood_function='gaussian')
        som.random_weights_init(train_data) #随机初始化

        som.train_batch(train_data,MAX_ITER,verbose=False)

        #labels_map利用标签信息，标注训练好的Som网络：
        winmap=som.labels_map(train_data,train_labels)
        # [x1,x2]:counter{1:2,2:1} x1,x1出现第一类 第二类的次数
        return som,winmap
#根据训练时 激活神经元的总类的比例 决定这个神经元属于哪一类
    def classify(self,som,data,winmap):
        from numpy import sum as npsum
        default_class = npsum(list(winmap.values())).most_common()[0][0]
        result = []
        for d in data:
            win_position = som.winner(d)
            if win_position in winmap:
                result.append(winmap[win_position].most_common()[0][0])
            else:
                 result.append(default_class)
        return result



#前行的类别在测试数据中的样本总量

    def print_map(self):
        heatmap = som.distance_map()
        plt.imshow(heatmap, cmap='bone_r')
        plt.colorbar()
        plt.show()
        # 在矩阵中标记输入的值
        plt.figure(figsize=(9, 9))
        plt.pcolor(heatmap, cmap='bone_r')  # 设置背景
        markers = ['o', 's', 'D']
        colors = ['C0', 'C1', 'C2']
        category_color = {'setosa': 'C0',
                          'versicolor': 'C1',
                          'virginica': 'C2'}
        for cnt, xx in enumerate(self.train_data):
            w=som.winner(xx)
            plt.plot(w[0]+.5, w[1]+.5, markers[self.train_labels[cnt]], markerfacecolor='None',
                     markeredgecolor=colors[self.train_labels[cnt]], markersize=12, markeredgewidth=2)
        plt.axis([0, self.size, 0,  self.size,])
        ax = plt.gca()
        ax.invert_yaxis()
        legend_elements = [Patch(facecolor=clr,
                                 edgecolor='w',
                                 label=l) for l, clr in category_color.items()]
        plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, .95))
        plt.show()

def print_Roc(pred_labels,test_labels):
    pred_labels=np.eye(3)[pred_labels]
    test_labels=np.eye(3)[test_labels]
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(3):
        fpr[i],tpr[i],_=roc_curve(test_labels[:,i],pred_labels[:,i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(test_labels.ravel(), pred_labels.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    plt.figure()

    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(3), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=3,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))


    plt.plot([0, 1], [0, 1], 'k--', lw=3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('多类ROC')
    plt.legend(loc="lower right")
    plt.show()



if __name__ == '__main__':
    som_modle=SOM()
    som_train_data, som_test_data, som_train_labels,som_test_labels = som_modle.get_data()
    som,winmap=som_modle.get_som()
    som_pred_labels = som_modle.classify(som,som_test_data,winmap) #(45*3)
    print(classification_report(som_test_labels, np.array(som_pred_labels)))

    test_labels=np.eye(3)[som_test_labels]
    pred_labels=np.eye(3)[som_pred_labels]
    print(accuracy_score(test_labels,pred_labels)) #预测正确占总比例
    print(accuracy_score(test_labels[:,0],pred_labels[:,0])) #预测正确占总比例

    print(precision_score(test_labels,pred_labels,average='micro')) #预测正样本中 正确的比例