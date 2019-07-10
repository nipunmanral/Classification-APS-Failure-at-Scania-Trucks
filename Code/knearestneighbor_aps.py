from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import scikitplot as skplt
import gc

def get_best_parameter(features, labels, list_k_neighbor):
    ACC = np.zeros(len(list_k_neighbor))
    F1 = np.zeros(len(list_k_neighbor))
    CONF = np.zeros(len(list_k_neighbor))
    list_val_accuracy = []
    list_val_f1score = []
    list_val_cost = []
    for k_index in range(len(list_k_neighbor)):
        print("Training configuration {}....".format(k_index + 1))
        current_k = list_k_neighbor[k_index]
        skf = StratifiedKFold(n_splits=5, shuffle=True)
        for train_index, val_index in skf.split(features, labels):
            x_train = features[train_index]
            x_val = features[val_index]
            y_train = labels[train_index]
            y_val = labels[val_index]
            kneigh = KNeighborsClassifier(n_neighbors=current_k)
            kneigh.fit(x_train, y_train)
            pred_labels = kneigh.predict(x_val)
            accuracy_value = accuracy_score(y_val, pred_labels)
            f1_value = f1_score(y_val, pred_labels)
            list_val_accuracy.append(accuracy_value)
            list_val_f1score.append(f1_value)
            val_confusion = confusion_matrix(y_val, pred_labels)
            cost_confusion = 10 * val_confusion[0][1] + 500 * val_confusion[1][0]
            list_val_cost.append(cost_confusion)
        ACC[k_index] = np.mean(list_val_accuracy)
        F1[k_index] = np.mean(list_val_f1score)
        CONF[k_index] = np.mean(list_val_cost)
    return ACC, F1, CONF


def classify_k_nearest_neigbor(features, labels, x_test, y_test):
    print("***************************")
    print("Using K Nearest Neighbor for Classification")
    list_k_neighbor = np.arange(3, 15, 2)
    ACC, F1, CONF = get_best_parameter(features, labels, list_k_neighbor)
    best_acc_index = np.argmax(ACC)
    best_f1_index = np.argmax(F1)
    best_conf_index = np.argmin(CONF)
    best_k = list_k_neighbor[best_conf_index]
    print("***************************")
    print("Best accuracy on validation set is {} when k = {}".format
          (ACC[best_acc_index], list_k_neighbor[best_acc_index]))
    print("Best F1 Score on validation set is {} when k = {}".format(F1[best_f1_index], list_k_neighbor[best_f1_index]))
    print("Lowest Total Cost on validation set is {} when k = {}".format
          (CONF[best_conf_index], list_k_neighbor[best_conf_index]))
    print("The value of K for the lowest total cost on validation set is ", best_k)

    np.save("array_KNN_Acc.npy", ACC)
    np.save("array_KNN_F1.npy", F1)
    np.save("array_KNN_Conf.npy", CONF)

    # Show statistics on test sat for lowest cost parameters
    kneigh = KNeighborsClassifier(n_neighbors=best_k)
    kneigh.fit(features, labels)
    test_pred_labels = kneigh.predict(x_test)
    y_probas = kneigh.predict_proba(x_test)
    test_accuracy_value = accuracy_score(y_test, test_pred_labels)
    test_f1_value = f1_score(y_test, test_pred_labels)
    confusion_k_neigbor = confusion_matrix(y_test, test_pred_labels)
    print("***************************")
    print("The Accuracy on test set using K Nearest Neighbor with K = {} is {}".format(best_k, test_accuracy_value))
    print("The F1 Score on test set using K Nearest Neighbor with K = {} is {}".format(best_k, test_f1_value))
    print("The Confusion Matrix for K Nearest Neighbor classifier with K = {} is:".format(best_k))
    print(confusion_k_neigbor)
    false_positive = confusion_k_neigbor[0][1]
    false_negative = confusion_k_neigbor[1][0]
    total_cost = 10 * false_positive + 500 * false_negative
    print("Total Cost using K Nearest Neighbor Classifier with K = {} on test set is {}".format(best_k, total_cost))
    roc_score = roc_auc_score(y_test, y_probas[:, 1])
    print("The ROC AUC Score using K Nearest Neighbor Classifier with K = {} is {}".format
          (best_k, roc_score))
    skplt.metrics.plot_confusion_matrix(y_test, test_pred_labels, normalize=True)
    plt.show()
    skplt.metrics.plot_roc(y_test, y_probas, plot_micro=False, plot_macro=False)
    plt.show()
    skplt.metrics.plot_precision_recall(y_test, y_probas, plot_micro=False)
    plt.show()
    print("***************************")

    ACC_graph = np.reshape(ACC, (1, len(ACC)))
    plt.imshow(ACC_graph)
    plt.xlabel("K Value")
    plt.title("K Nearest Neighbor - Accuracy Metrics Visualisation")
    plt.gca().axes.get_yaxis().set_ticks([])
    plt.xticks(np.arange(len(list_k_neighbor)), list_k_neighbor)
    plt.colorbar()
    plt.show()

    F1_graph = np.reshape(F1, (1, len(F1)))
    plt.imshow(F1_graph)
    plt.xlabel("K Value")
    plt.title("K Nearest Neighbor - F1 Score Metrics Visualisation")
    plt.gca().axes.get_yaxis().set_ticks([])
    plt.xticks(np.arange(len(list_k_neighbor)), list_k_neighbor)
    plt.colorbar()
    plt.show()

    CONF_graph = np.reshape(CONF, (1, len(CONF)))
    plt.imshow(CONF_graph)
    plt.xlabel("K Value")
    plt.title("K Nearest Neighbor - Total Cost Visualisation")
    plt.gca().axes.get_yaxis().set_ticks([])
    plt.xticks(np.arange(len(list_k_neighbor)), list_k_neighbor)
    plt.colorbar()
    plt.show()

    gc.collect()
