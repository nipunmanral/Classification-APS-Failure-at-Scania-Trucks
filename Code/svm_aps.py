import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
import scikitplot as skplt
import gc


def get_AccuracyRate(features, labels, no_splits, gamma_value, C_value):
    list_val_accuracy = []
    list_val_f1score = []
    list_val_cost = []
    skf = StratifiedKFold(n_splits=no_splits, shuffle=True)
    for train_index, val_index in skf.split(features, labels):
        x_train = features[train_index]
        x_val = features[val_index]
        y_train = labels[train_index]
        y_val = labels[val_index]
        svm_rbf = SVC(kernel='rbf', C=C_value, gamma=gamma_value)
        svm_rbf.fit(x_train, y_train)
        pred_labels = svm_rbf.predict(x_val)
        accuracy_value = accuracy_score(y_val, pred_labels)
        f1_value = f1_score(y_val, pred_labels)
        list_val_accuracy.append(accuracy_value)
        list_val_f1score.append(f1_value)
        val_confusion = confusion_matrix(y_val, pred_labels)
        cost_confusion = 10 * val_confusion[0][1] + 500 * val_confusion[1][0]
        list_val_cost.append(cost_confusion)
    avg_val_accuracy = np.mean(list_val_accuracy)
    avg_val_accuracy_std = np.std(list_val_accuracy)
    avg_val_f1_score = np.mean(list_val_f1score)
    avg_val_f1_std = np.std(list_val_f1score)
    avg_val_cost = np.mean(list_val_cost)
    return avg_val_accuracy, avg_val_accuracy_std, avg_val_f1_score, avg_val_f1_std, avg_val_cost

def get_all_acc_dev(features, labels, list_gamma, list_C):
    ACC = np.zeros((len(list_gamma), len(list_C)))
    DEV = np.zeros((len(list_gamma), len(list_C)))
    F1 = np.zeros((len(list_gamma), len(list_C)))
    F1_DEV = np.zeros((len(list_gamma), len(list_C)))
    CONF = np.zeros((len(list_gamma), len(list_C)))
    cnt = 1
    for gamma_index in range(len(list_gamma)):
        for C_index in range(len(list_C)):
            print("Calculating parameter combination number ", cnt)
            cnt += 1
            avg_acc, avg_acc_std, avg_f1, avg_f1_std, avg_cost = get_AccuracyRate(features,
                labels, no_splits=5, gamma_value=list_gamma[gamma_index], C_value=list_C[C_index])
            ACC[gamma_index][C_index], DEV[gamma_index][C_index] = avg_acc, avg_acc_std
            F1[gamma_index][C_index], F1_DEV[gamma_index][C_index] = avg_f1, avg_f1_std
            CONF[gamma_index][C_index] = avg_cost
    return ACC, DEV, F1, F1_DEV, CONF

def get_best_parameters_index(ACC, DEV):
    max_accuracy = np.max(ACC)
    list_index_max_accuracy = [np.unravel_index(idx, ACC.shape)for idx, val in enumerate(ACC.flatten())
                               if val == max_accuracy]
    best_index_max_accuracy = []
    if len(list_index_max_accuracy) > 1:
        print("{} pairs of C and gamma have the same Accuracy/F1 Score. Choosing best pair....".format
              (len(list_index_max_accuracy)))
        low_value_std = DEV[list_index_max_accuracy[0][0], list_index_max_accuracy[0][1]]
        low_value_idx = 0
        for x_idx, x_val in enumerate(list_index_max_accuracy):
            if DEV[x_val[0], x_val[1]] < low_value_std:
                low_value_std = DEV[x_val[0], x_val[1]]
                low_value_idx = x_idx
        best_index_max_accuracy.append(list_index_max_accuracy[low_value_idx])
        best_index_max_accuracy = np.asarray(best_index_max_accuracy)
    else:
        best_index_max_accuracy = list_index_max_accuracy
    return best_index_max_accuracy

def classify_SVM(features, labels, x_test, y_test):
    print("***************************")
    print("Using Support Vector Machines for Classification")
    list_gamma = np.logspace(start=-2, stop=3, num=10, endpoint=True, base=10)
    list_C = np.logspace(start=-2, stop=3, num=10, endpoint=True, base=10)
    ACC, DEV, F1, F1_DEV, CONF = get_all_acc_dev(features, labels, list_gamma, list_C)

    # Show statistics regarding the accuracy metric
    best_index_max_accuracy = get_best_parameters_index(ACC, DEV)
    best_C_acc = list_C[best_index_max_accuracy[0][1]]
    best_gamma_acc = list_gamma[best_index_max_accuracy[0][0]]
    print("Index of best Accuracy on validation set is ", best_index_max_accuracy)
    print("Best value of C for the accuracy metric on validation set is ", best_C_acc)
    print("Best value of gamma for the accuracy metric on validation set is ", best_gamma_acc)
    print("Best Accuracy Score on validation set is {} when C = {} and Gamma = {}".format(
          ACC[best_index_max_accuracy[0][0], best_index_max_accuracy[0][1]], best_C_acc, best_gamma_acc))
    print("Standard Deviation of Best Accuracy Score is ",
          DEV[best_index_max_accuracy[0][0], best_index_max_accuracy[0][1]])
    print("***************************")

    # Show statistics regarding the F1 Score metric
    best_index_max_f1 = get_best_parameters_index(F1, F1_DEV)
    best_C_f1 = list_C[best_index_max_f1[0][1]]
    best_gamma_f1 = list_gamma[best_index_max_f1[0][0]]
    print("Index of best F1 Score on validation set is ", best_index_max_f1)
    print("Best value of C for the F1 Score metrics on validation set is ", best_C_f1)
    print("Best value of gamma for the F1 Score metrics on validation set is ", best_gamma_f1)
    print("Best F1 Score on validation set is {} when C = {} and Gamma = {}".format(
        F1[best_index_max_f1[0][0], best_index_max_f1[0][1]], best_C_f1, best_gamma_f1))
    print("Standard Deviation of Best F1 Score is ", F1_DEV[best_index_max_f1[0][0], best_index_max_f1[0][1]])
    print("***************************")

    # Show statistics regarding Confusion Matrix and Total Cost
    min_cost = np.min(CONF)
    index_min_cost = [np.unravel_index(idx, CONF.shape)for idx, val in enumerate(CONF.flatten())
                               if val == min_cost][0]
    best_C_cost = list_C[index_min_cost[1]]
    best_gamma_cost = list_gamma[index_min_cost[0]]
    print("Index of minimum cost on validation set is ", index_min_cost)
    print("Best value of C for the cost metrics on validation set is ", best_C_cost)
    print("Best value of gamma for the cost metrics on validation set is ", best_gamma_cost)
    print("Least cost on validation set is {} when C = {} and Gamma = {}".format(
        CONF[index_min_cost[0], index_min_cost[1]], best_C_cost, best_gamma_cost))
    print("***************************")

    np.save("array_SVM_Acc.npy", ACC)
    np.save("array_SVM_Dev_Acc.npy", DEV)
    np.save("array_SVM_F1_Array.npy", F1)
    np.save("array_SVM_Std_F1_Array.npy", F1_DEV)
    np.save("array_SVM_Conf_Array.npy", CONF)

    plt.imshow(ACC)
    plt.xlabel("C Index")
    plt.ylabel("Gamma Index")
    plt.title("Accuracy Metrics Visualisation")
    plt.colorbar()
    plt.show()

    plt.imshow(DEV)
    plt.xlabel("C Index")
    plt.ylabel("Gamma Index")
    plt.title("Standard Deviation Visualisation for Accuracy Metrics")
    plt.colorbar()
    plt.show()

    plt.imshow(F1)
    plt.xlabel("C Index")
    plt.ylabel("Gamma Index")
    plt.title("F1 Score Metrics Visualisation")
    plt.colorbar()
    plt.show()

    plt.imshow(F1_DEV)
    plt.xlabel("C Index")
    plt.ylabel("Gamma Index")
    plt.title("Standard Deviation Visualisation for F1 Score Metrics")
    plt.colorbar()
    plt.show()

    plt.imshow(CONF)
    plt.xlabel("C Index")
    plt.ylabel("Gamma Index")
    plt.title("Total Cost Visualisation")
    plt.colorbar()
    plt.show()

    # best_C_cost = 1000
    # best_gamma_cost = 1
    svm_rbf = SVC(kernel='rbf', C=best_C_cost, gamma=best_gamma_cost, probability=True)
    svm_rbf.fit(features, labels)
    test_pred_labels = svm_rbf.predict(x_test)
    y_probas = svm_rbf.predict_proba(x_test)
    test_accuracy_value = accuracy_score(y_test, test_pred_labels)
    test_f1_value = f1_score(y_test, test_pred_labels)
    confusion_svm = confusion_matrix(y_test, test_pred_labels)
    print("***************************")
    print("The Accuracy on test set using SVM Classifier with C = {} and gamma = {} is {}".format
          (best_C_cost, best_gamma_cost, test_accuracy_value))
    print("The F1 Score  on test set using SVM Classifier with C = {} and gamma = {} is {}".format
          (best_C_cost, best_gamma_cost, test_f1_value))
    print("The Confusion Matrix for SVM classifier with C = {} and gamma = {} is:".format(best_C_cost, best_gamma_cost))
    print(confusion_svm)
    false_positive = confusion_svm[0][1]
    false_negative = confusion_svm[1][0]
    total_cost = 10 * false_positive + 500 * false_negative
    print("Total Cost using SVM Classifier with C = {} and gamma = {} on test set is {}".format
          (best_C_cost, best_gamma_cost, total_cost))
    roc_score = roc_auc_score(y_test, y_probas[:, 1])
    print("The ROC AUC Score using SVM Classifier with C ={} and gamma ={} is {}".format
          (best_C_cost, best_gamma_cost, roc_score))
    skplt.metrics.plot_confusion_matrix(y_test, test_pred_labels, normalize=True)
    plt.show()
    skplt.metrics.plot_roc(y_test, y_probas, plot_micro=False, plot_macro=False)
    plt.show()
    skplt.metrics.plot_precision_recall(y_test, y_probas, plot_micro=False)
    plt.show()
    print("***************************")
    gc.collect()
