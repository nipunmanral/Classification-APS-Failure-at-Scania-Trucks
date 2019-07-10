from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
import scikitplot as skplt
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import gc

def classify_naive_bayes(features, labels, x_test, y_test):
    print("***************************")
    print("Using Naive Bayes for Classification")
    gnb = GaussianNB()
    gnb.fit(features, labels)
    pred_labels = gnb.predict(x_test)
    y_probas = gnb.predict_proba(x_test)
    accuracy_value = accuracy_score(y_test, pred_labels)
    f1_value = f1_score(y_test, pred_labels)
    confusion_nb = confusion_matrix(y_test, pred_labels)
    print("***************************")
    print("The Accuracy on test set using Naive Bayes is ", accuracy_value)
    print("The F1 Score on test set using Naive Bayes is ", f1_value)
    print("The Confusion Matrix for Naive Bayes Classifier is ", confusion_nb)
    false_positive = confusion_nb[0][1]
    false_negative = confusion_nb[1][0]
    total_cost = 10 * false_positive + 500 * false_negative
    print("Total Cost using Naive Bayes Classifier on test set is ", total_cost)
    roc_score = roc_auc_score(y_test, y_probas[:, 1])
    print("The ROC AUC Score using Naive Bayes Classifier is {}".format
          (roc_score))
    skplt.metrics.plot_confusion_matrix(y_test, pred_labels, normalize=True)
    plt.show()
    skplt.metrics.plot_roc(y_test, y_probas, plot_micro=False, plot_macro=False)
    plt.show()
    skplt.metrics.plot_precision_recall(y_test, y_probas, plot_micro=False)
    plt.show()
    print("***************************")
    gc.collect()
