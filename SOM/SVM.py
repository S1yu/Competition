from itertools import cycle

import matplotlib.pyplot as plt
import numpy
from sklearn import svm, datasets
from sklearn.metrics import classification_report, roc_curve, auc, accuracy_score, precision_score,recall_score
from sklearn.model_selection import train_test_split
import numpy as np

#macroaverage是每个类有相同的权重 precision、recall 求和除以类别个数
# micro-average     每个样本相同权重  正确个数/总个数 与样本类的分布多少无关
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize

from SOM import som

def get_data():
    iris = datasets.load_iris()


    X = iris.data
    y = iris.target
    y = label_binarize(y, classes=[0, 1, 2])
    #7：3划分
    # random_state = np.random.RandomState(0)
    # n_samples, n_features = X.shape
    # X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]
    return train_test_split(X,y,test_size=0.3,random_state=0)
X_train, X_test, y_train, y_test  = get_data()
classNunber=3
svm=classifier = OneVsRestClassifier(
    svm.SVC(C=1,kernel="rbf")
)
# 对超平面的距离
print(X_train.size,y_train.size,X_test.size)
y_score = svm.fit(X_train, y_train).decision_function(X_test)

print("SVM : \n "+classification_report(y_test,classifier.predict(X_test))) #打印

def print_Roc():
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(classNunber):
        fpr[i],tpr[i],_=roc_curve(y_test[:,i],y_score[:,i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    #ravel： 降维
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
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


som_modle=som.SOM()
som_train_data, som_test_data, som_train_labels,som_test_labels = som_modle.get_data()
som,winmap=som_modle.get_som()
som_pred_labels = som_modle.classify(som,som_test_data,winmap) #(45*3)
print("SOM \n"+classification_report(som_test_labels, np.array(som_pred_labels)))

test_labels=np.eye(3)[som_test_labels]
pred_labels=np.eye(3)[som_pred_labels]

som_acc=accuracy_score(test_labels,pred_labels)
svm_acc=accuracy_score(y_test,classifier.predict(X_test))
som_precision_mac=[]
som_recall=[]
svm_precision_mac=[]
svm_recall=[]
for i in range(3):
    som_precision_mac.append(precision_score(test_labels[:,i],pred_labels[:,i],average='macro'))
    som_recall.append(recall_score(test_labels[:,i],pred_labels[:,i],average='macro'))
    svm_precision_mac.append(precision_score(y_test[:,i],classifier.predict(X_test)[:,i],average='macro'))
    svm_recall.append(recall_score(y_test[:,i],classifier.predict(X_test)[:,i],average='macro'))
c=np.c_[np.asarray(som_precision_mac),np.asarray(som_recall)]

tab=plt.table(cellText=c,colWidths=[0.3]*6,loc='center',cellLoc='center',rowLoc='center')
tab.scale(1,2)
plt.axis("off")
plt.show()




print(accuracy_score(y_test,classifier.predict(X_test))) #预测正确占总比例
print(accuracy_score(y_test[:,0],classifier.predict(X_test)[:,0])) #预测正确占总比例

print(precision_score(y_test,classifier.predict(X_test),average='micro')) #预测正样本中 正确的比例













