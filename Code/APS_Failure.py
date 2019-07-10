import numpy as np
import os
import svm_aps
import naivebayes_aps
import knearestneighbor_aps
import mlp_aps
import baselinemodel_aps
import gc


gc.collect()
data_path = os.path.dirname(__file__)
train_data_set = np.genfromtxt(os.path.join(data_path, "train_processed_aps.csv"), delimiter=',')
test_data_set = np.genfromtxt(os.path.join(data_path, "test_processed_aps.csv"), delimiter=',')
features = train_data_set[:, 1:]
labels = train_data_set[:, 0]
x_test = test_data_set[:, 1:]
y_test = test_data_set[:, 0]
print("Number of features = ", features.shape[1])

svm_aps.classify_SVM(features, labels, x_test, y_test)
naivebayes_aps.classify_naive_bayes(features, labels, x_test, y_test)
knearestneighbor_aps.classify_k_nearest_neigbor(features, labels, x_test, y_test)
mlp_aps.classify_MLP(features, labels, x_test, y_test)
baselinemodel_aps.classify_baseline_model(y_test)