########################################################################################################################
# Author:      Antoni Adamczyk
# MatNr:       12306508
# Description: ... short description of the file ...
# Comments:    ... comments for the tutors ...
#              ... can be multiline ...
########################################################################################################################

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# TODO Import anything from sklearn that you need for your calculations
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.metrics import r2_score

#------------------------------------------------------- Tasks --------------------------------------------------------#
# IMPORTANT: In contrast to the other assignments, in this assignment you can define as many helper functions as you 
# want. Just implement the main logic in the respective task sections.


def read_data(
    red_wine: str, 
    white_wine: str
    ) -> pd.DataFrame:
    """
    Args:
        red_wine (str): The path to the red wine dataset.
        white_wine (str): The path to the white wine dataset.

    Returns:
        merged dataset (pd.DataFrame): The merged dataset of red and white wine with the additional type collumn.
    """
    pass

    red_wine_data = pd.read_csv(red_wine)
    white_wine_data = pd.read_csv(white_wine)

    red_wine_data['type'] = 0
    white_wine_data['type'] = 1

    merged_data = pd.concat([red_wine_data, white_wine_data], ignore_index=True)

    return merged_data

print("Linear Regression Model for Wine Quality Prediction".center(80, "-"))
# TODO Implement the first task

def Exploratory_data_analysis(merged_data: pd.DataFrame):

    for column in merged_data.columns[:-1]:
        plt.figure(figsize=(8, 5))
        plt.hist(merged_data[column], bins=20)
        plt.title(column)
        plt.xlabel(column)
        plt.ylabel('Count')
        plt.grid(True)
        plt.show()
    
def Cross_Validation(merged_data: pd.DataFrame):
    if 'quality' in merged_data.columns:
        X = merged_data.drop(['quality'], axis=1)
    if 'type' in merged_data.columns:
        X = merged_data.drop(['type'], axis=1)
    
    y = merged_data['quality']

    X_standardized = preprocessing.scale(X)

    r2_scores = []
    cv = 5
    kfold = KFold(n_splits=cv, shuffle=True, random_state=None)

    model = LinearRegression()

    for train_index, test_index in kfold.split(X_standardized):
        X_train, X_test = X_standardized[train_index], X_standardized[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        r2_scores.append(r2)

    for i, r2 in enumerate(r2_scores):
        print(f'Fold {i+1}: R^2 = {r2:.4f}')

    average_r2 = np.mean(r2_scores)
    print(f'Average R^2 across folds: {average_r2:.4f}')

    return average_r2


Exploratory_data_analysis(read_data("winequality-red.csv", "winequality-white.csv"))
Cross_Validation(read_data("winequality-red.csv", "winequality-white.csv"))

print("Model Selection with Greedy Search".center(80, "-"))
# TODO Implement the second task

def model_selection(data, features):
    combined_columns = features + ['quality']
    data_subset = data[combined_columns]
    full_score = Cross_Validation(data_subset)

    feature_scores = {}
    for feature in features:
        new_data = data_subset
        new_data = new_data.drop([feature], axis=1)
        feature_scores[feature] = Cross_Validation(new_data)

    weakest_feature = max(feature_scores, key=feature_scores.get)
    return full_score, feature_scores[weakest_feature], [f for f in features if f != weakest_feature]


def gready_search(data: pd.DataFrame):
    data = data.drop(['type'], axis=1)
    all_features = list(data.columns[:-1])
    y = data.iloc[:, -1]

    score = 0
    features = list(all_features)

    
    zero_score = Cross_Validation(data)
    best_score = zero_score
    while True:
        full_score, score, new_features = model_selection(data, features)
        print('{}\nscore={}'.format(features, full_score))
        print('{}\nnew score={}'.format(new_features, score))
        print('********')
        if score >= best_score:  
            best_score = score          
            features = new_features
        else:
            break
    print('Best model:\n{}\nscore={}'.format(features, best_score))
    print('Full model:\n{}\nscore={}'.format(all_features, zero_score))

    return features

gready_search(read_data("winequality-red.csv", "winequality-white.csv"))

print("Logistic Regression Model for Wine Type".center(80, "-"))
# TODO Implement the third task

