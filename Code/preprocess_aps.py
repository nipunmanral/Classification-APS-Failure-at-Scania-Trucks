import pandas as pd
import os
import gc
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import seaborn as sns

# np.set_printoptions(suppress=True)
pd.options.display.max_columns = 200
pd.options.display.max_rows = 200

# List to denote how missing values are indicated in the data-set
missing_values_label = ["na"]

# Read the training and test data from the csv files
train_data_path = os.path.join(os.path.dirname(__file__), "Dataset/aps_failure_training_set_SMALLER.csv")
# train_data_path = os.path.join(os.path.dirname(__file__), "Dataset/aps_failure_training_set.csv")
test_data_path = os.path.join(os.path.dirname(__file__), "Dataset/aps_failure_test_set.csv")
df_train_data = pd.read_csv(train_data_path, sep=',', header=0, na_values=missing_values_label)
df_test_data = pd.read_csv(test_data_path, sep=',', header=0, na_values=missing_values_label)

# Display the number of rows and columns
total_num_data = len(df_train_data.index)
print("Number of attributes = ", len(df_train_data.columns))
print("Number of data = ", total_num_data)
print("*******************")

# Display all column headers
print("----- Column Headers -----")
print(df_train_data.columns.values)

# Display the first n rows
print("----- Display top rows -----")
print(df_train_data.head(n=5))
#
# Describe the statistics of the data-set
print("----- Data-Set Statistics -----")
print(df_train_data.describe(include="all"))

# Print number of positive classes and number of negative classes in the training data-set
print("Number of positive classes = ", sum(df_train_data['class'] == 'pos'))
print("Number of negative classes = ", sum(df_train_data['class'] == 'neg'))
print("*******************")

# Replace class labels with integer values (neg = 0, pos = 1) in training and test data-set
df_train_data['class'].replace({
    'neg': 0,
    'pos': 1
}, inplace=True)
df_test_data['class'].replace({
    'neg': 0,
    'pos': 1
}, inplace=True)

# Compute the percentage of missing data for each attribute in the training data set
missing_percent_threshold = 0.50
missing_data_count = pd.DataFrame(df_train_data.isnull().sum().sort_values(ascending=False), columns=['Number'])
missing_data_percent = pd.DataFrame(df_train_data.isnull().sum().sort_values(ascending=False)/total_num_data, columns=['Percent'])
missing_data = pd.concat([missing_data_count, missing_data_percent], axis=1)
# print(missing_data)
missing_column_headers = missing_data[missing_data['Percent'] > missing_percent_threshold].index
print(missing_column_headers)

# Drop the features with high amount of missing data in both train and test data-set
df_train_data = df_train_data.drop(columns=missing_column_headers)
print("Training data-set shape after dropping features is ", df_train_data.shape)
df_test_data = df_test_data.drop(columns=missing_column_headers)
print("Test data-set shape after dropping features is ", df_test_data.shape)
# print(df_data.describe())

# Extract features and labels from the training and test data-set
y_train = df_train_data.loc[:, 'class']
x_train = df_train_data.drop('class', axis=1)
y_test = df_test_data.loc[:, 'class']
x_test = df_test_data.drop('class', axis=1)

# corrmat = x_train.corr()
# sns.heatmap(corrmat, vmax=.8, square=True);
# plt.show()

# Fill missing data in training and test data-set
imputer_median = SimpleImputer(strategy='median')
imputer_median.fit(x_train.values)
x_train = imputer_median.transform(x_train.values)
x_test = imputer_median.transform(x_test.values)

# Standardize the training and test data-set
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#  Synthetic Minority Oversampling Technique to balance the training data-set
sm = SMOTE()
x_train, y_train = sm.fit_sample(x_train, y_train)

# Principal Component Analysis
pca = PCA(n_components=0.75)
pca.fit(x_train)
x_train = pca.transform(x_train)
x_test = pca.transform(x_test)
print("Number of features after PCA = ", x_test.shape[1])

# corrmat_pca = pd.DataFrame(x_train).corr()
# sns.heatmap(corrmat_pca, vmax=.8, square=True);
# plt.show()


# Save the data-sets to a csv file
x_train = pd.DataFrame(x_train)
y_train = pd.DataFrame(y_train)
df_train_data = pd.concat([y_train, x_train], axis=1)
df_train_data.to_csv('train_processed_aps.csv', sep=',', index=False, header=False)

x_test = pd.DataFrame(x_test)
y_test = pd.DataFrame(y_test)
df_test_data = pd.concat([y_test, x_test], axis=1)
df_test_data.to_csv('test_processed_aps.csv', sep=',', index=False, header=False)

gc.collect()
print("Pre processing completed")
