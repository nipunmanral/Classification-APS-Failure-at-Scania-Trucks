import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


def classify_baseline_model(y_test):
    print("Majority class labels in the training data-set are negative i.e. 0 labels")
    test_pred_labels = np.zeros(len(y_test))
    accuracy_value = accuracy_score(y_test, test_pred_labels)
    f1_value = f1_score(y_test, test_pred_labels)
    confusion_base = confusion_matrix(y_test, test_pred_labels)
    print("***************************")
    print("The Accuracy on test set using baseline model is ", accuracy_value)
    print("The F1 Score on test set using baseline model is ", f1_value)
    print("The Confusion Matrix for baseline model is ", confusion_base)
    false_positive = confusion_base[0][1]
    false_negative = confusion_base[1][0]
    total_cost = 10 * false_positive + 500 * false_negative
    print("Total Cost using baseline model is ", total_cost)

