

from sklearn.metrics import roc_curve, auc, precision_recall_curve, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
### draw roc
def RocBySelf(predict_score,y_test):
    '''
    ROC  曲线图片
    :param model:
    :param x_test:
    :param y_test:
    :return:
    '''
    fpr, tpr, t = roc_curve(y_test, predict_score)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.plot(fpr, tpr, lw=2, alpha=0.3)
    plt.legend()
    plt.title("ROC ")
    plt.show()
    precision, recall, thresholds = precision_recall_curve(y_test, predict_score)
    plt.figure(1)
    plt.plot(precision, recall)
    plt.title("Precision-Recall")
    plt.legend()
    plt.show()
    plt.savefig('Precision-Recall.png')
def AUROC(correct_label, predict_score):
  fpr, tpr, _ = roc_curve(correct_label, predict_score)
  auroc = auc(fpr, tpr)
  RocBySelf(predict_score, correct_label)
  return auroc

def AUPR(correct_label, predict_score):
  precision, recall, _ = precision_recall_curve(correct_label, predict_score)
  aupr = auc(recall, precision)
  return aupr
