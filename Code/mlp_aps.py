from sklearn.neural_network import MLPClassifier
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

def get_best_parameter(features, labels, list_hidden_layer_size):
    ACC = np.zeros(len(list_hidden_layer_size))
    F1 = np.zeros(len(list_hidden_layer_size))
    CONF = np.zeros(len(list_hidden_layer_size))
    list_val_accuracy = []
    list_val_f1score = []
    list_val_cost = []
    for idx in range(len(list_hidden_layer_size)):
        print("Training configuration {}....".format(idx+1))
        current_layer_size = list_hidden_layer_size[idx]
        skf = StratifiedKFold(n_splits=5, shuffle=True)
        for train_index, val_index in skf.split(features, labels):
            x_train = features[train_index]
            x_val = features[val_index]
            y_train = labels[train_index]
            y_val = labels[val_index]
            clf = MLPClassifier(hidden_layer_sizes=current_layer_size, max_iter=700, alpha=0.0001,
                                solver='sgd', verbose=0)
            clf.fit(x_train, y_train)
            pred_labels = clf.predict(x_val)
            accuracy_value = accuracy_score(y_val, pred_labels)
            f1_value = f1_score(y_val, pred_labels)
            list_val_accuracy.append(accuracy_value)
            list_val_f1score.append(f1_value)
            val_confusion = confusion_matrix(y_val, pred_labels)
            cost_confusion = 10 * val_confusion[0][1] + 500 * val_confusion[1][0]
            list_val_cost.append(cost_confusion)
        ACC[idx] = np.mean(list_val_accuracy)
        F1[idx] = np.mean(list_val_f1score)
        CONF[idx] = np.mean(list_val_cost)
    return ACC, F1, CONF

def classify_MLP(features, labels, x_test, y_test):
    print("***************************")
    print("Using Multi Layer Perceptron for Classification")
    list_hidden_layer_size = [(100), (200), (200, 100), (200, 200)]
    ACC, F1, CONF = get_best_parameter(features, labels, list_hidden_layer_size)
    best_acc_index = np.argmax(ACC)
    best_f1_index = np.argmax(F1)
    best_conf_index = np.argmin(CONF)
    best_layer_size = list_hidden_layer_size[best_conf_index]
    print("***************************")
    print("Best accuracy on validation set is {} when hidden layer configuration = {}".format
          (ACC[best_acc_index], list_hidden_layer_size[best_acc_index]))
    print("Best F1 Score on validation set is {} when hidden layer configuration = {}".format
          (F1[best_f1_index], list_hidden_layer_size[best_f1_index]))
    print("Lowest Total Cost on validation set is {} when k = {}".format
          (CONF[best_conf_index], list_hidden_layer_size[best_conf_index]))
    print("The hidden layer size for the lowest total cost on validation set is ", best_layer_size)

    np.save("array_MLP_Acc.npy", ACC)
    np.save("array_MLP_F1.npy", F1)
    np.save("array_MLP_Conf.npy", CONF)

    # Compute F1 Score and Accuracy metric on test data-set
    clf = MLPClassifier(hidden_layer_sizes=best_layer_size, max_iter=500, alpha=0.0001,
                        solver='sgd', verbose=0, tol=0.000000001)
    clf.fit(features, labels)
    test_pred_labels = clf.predict(x_test)
    y_probas = clf.predict_proba(x_test)
    test_accuracy_value = accuracy_score(y_test, test_pred_labels)
    test_f1_value = f1_score(y_test, test_pred_labels)
    confusion_mlp = confusion_matrix(y_test, test_pred_labels)
    print("***************************")
    print("The Accuracy on test set using Multilayer Perceptron with hidden layer configuration = {} is {}".format
          (best_layer_size, test_accuracy_value))
    print("The F1 Score on test set using Multilayer Perceptron with hidden layer configuration = {} is {}".format
          (best_layer_size, test_f1_value))
    print("The Confusion Matrix using Multilayer Perceptron with hidden layer configuration = {} is:".format
          (best_layer_size))
    print(confusion_mlp)
    false_positive = confusion_mlp[0][1]
    false_negative = confusion_mlp[1][0]
    total_cost = 10 * false_positive + 500 * false_negative
    print("Total Cost using Multilayer Perceptron with hidden layer configuration = {} on test set is {}".format
          (best_layer_size, total_cost))
    roc_score = roc_auc_score(y_test, y_probas[:, 1])
    print("The ROC AUC Score using Multilayer Perceptron Classifier with hidden layer configuration = {} is {}".format
          (best_layer_size, roc_score))
    skplt.metrics.plot_confusion_matrix(y_test, test_pred_labels, normalize=True)
    plt.show()
    skplt.metrics.plot_roc(y_test, y_probas, plot_micro=False, plot_macro=False)
    plt.show()
    skplt.metrics.plot_precision_recall(y_test, y_probas, plot_micro=False)
    plt.show()
    print("***************************")

    ACC_graph = np.reshape(ACC, (1, len(ACC)))
    plt.imshow(ACC_graph)
    plt.xlabel("Hidden Layers")
    plt.title("Multilayer Perceptron - Accuracy Metrics Visualisation")
    plt.gca().axes.get_yaxis().set_ticks([])
    plt.colorbar()
    plt.show()

    F1_graph = np.reshape(F1, (1, len(F1)))
    plt.imshow(F1_graph)
    plt.xlabel("Hidden Layers")
    plt.title("Multilayer Perceptron- F1 Score Metrics Visualisation")
    plt.gca().axes.get_yaxis().set_ticks([])
    plt.colorbar()
    plt.show()

    CONF_graph = np.reshape(CONF, (1, len(CONF)))
    plt.imshow(CONF_graph)
    plt.xlabel("Hidden Layers")
    plt.title("Multilayer Perceptron - Total Cost Visualisation")
    plt.gca().axes.get_yaxis().set_ticks([])
    plt.colorbar()
    plt.show()

    gc.collect()
