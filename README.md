# APS Failure at Scania Trucks Dataset

## Summary
This project aims to develop a pattern recognition system for the “APS Failure at Scania Trucks” dataset in order to minimize the overall maintenance costs of the air pressure system in Scania trucks. Four different types of classification techniques (Support Vector Machines, Naïve Bayes, K Nearest Neighbor and Multilayer perceptron) were used to evaluate, compare and optimize the total cost. The Support Vector Machines classifier with Radial basis Function kernel, penalty parameter (C) = 1000 and kernel parameter (gamma) = 1 gave the best minimum total cost of 14,090$ with a ROC AUC score of 0.9805 using the down-sampled training dataset and 12,260$ with a ROC AUC score of 0.9865 using the complete training dataset.

## Environment Setup
Download the codebase and open up a terminal in the root directory. Make sure python 3.6 is installed in the current environment. Then execute

    pip install -r requirements.txt

This should install all the necessary packages for the code to run.

## Dataset
The dataset consists of data collected from heavy Scania trucks in everyday usage. The system in focus is the “Air Pressure system (APS) which generates pressurized air that are utilized in various functions in a truck, such as braking and gear changes.” This is a 2-class problem, and the goal is to predict the failure of components in the APS system, given various inputs.

The training set labeled “SMALLER” is present in the **Code/Dataset/** folder and has been down-sampled by factor of 3 (stratified), from the complete training set. The test set is also present in the same folder. The dataset has 171 attributes.

For more information on the dataset and to download the entire training data, visit the link:
https://archive.ics.uci.edu/ml/datasets/APS+Failure+at+Scania+Trucks

## Evaluation of Performance
For APS dataset, the unnormalized weighted-error measure that is defined in the aps_failure_description.txt file on the UCI website is used which provides a score for the classification system (lower scores are better). Additionally, the confusion matrix, and the F1 score are also used.
Note: The “positive” class is the APS failure class; “type 1” error means a false positive, and “type 2” error means a false negative.

The positive class in the dataset corresponds to component failures for a specific component of the APS system. The negative class corresponds to failures for components not related to the APS

The cost-metric of mis-classification is Cost_1 = 10 and Cost_2 = 500 where Cost_1 refers to the cost that an unnecessary checks needs to be done by a mechanic at a workshop (Type 1 - false positive) and Cost_2 refers to the cost of missing a faulty truck, which may cause a breakdown (Type 2 - false negative). Hence, the total cost is calculated as:

Total Cost = Cost_1 * Type 1 failure instances + Cost_2 * Type 2 Failure instances

The goal of this project is to create a prediction model using classification techniques learnt in class that minimizes the total cost, which in turn would minimize the maintenance costs of Scania trucks.

## Baseline Model
Baseline for APS dataset: A classifier that always decides majority class. Note that this will give a very high accuracy, but a sub-par weighted-error score and a high number of false negatives (Type 2 errors).

## Pre-Processing
In order to pre-process the data, execute the code file available in the `preprocess_aps.py` file. This code performs pre-processing operations such as imputation, standardization, PCA and SMOTE on the datasets in the **Code/Dataset/** and saves the processed files by the name `train_processed_aps.csv` and `test_processed_aps.csv` in the same folder as the script file (**Code/**)

The `train_processed_aps.csv` and `test_processed_aps.csv` are already uploaded along with the code for use.

## Code
Execute the `APS_Failure.py` file which runs the different classifiers:
- Support Vector Machines (*svm_aps.py*)
- Naïve Bayes (*naivebayes_aps.py*)
- K Nearest Neighbor (*knearestneighbor_aps.py*)
- Multilayer perceptron (*mlp_aps.py*)
- Baseline Model (*baselinemodel_aps.py*)

## Conclusion
The APS Failure at Scania Trucks dataset was evaluated using different classifiers on the test dataset and among all the classifiers, support vector machines with radial basis function kernel achieved the lowest total cost of 14,090$ using the down-sampled training dataset and 12,260$ using the complete training dataset.

We also observed that all our four classifiers performed better than the baseline model. For further follow-on work, we can investigate the impact of increasing the number of hidden layers and the number of neurons in Multilayer Perceptron as that could improve the performance of our classifier. It would also be interesting to perform feature engineering on the existing features and evaluate the performance on the test set using other classifiers such as Random Forest Classifier.

For more information on the project, refer the project report.
