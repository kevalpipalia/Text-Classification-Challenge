import os
import pandas as pd
from sklearn.model_selection import train_test_split

def load_modeling_data():
    train_data = pd.read_csv('./data/train_data.csv', index_col=0)
    train_labels = pd.read_csv('./data/train_results.csv', index_col=0)
    return train_data, train_labels

def load_testing_data():
    test_data = pd.read_csv('./data/test_data.csv', index_col=0)
    return test_data

def prepare_kaggle_submission(y_predicted, filename):
    y_pred = pd.DataFrame(y_predicted).reset_index()
    y_pred.columns = ['id', 'target']
    y_pred.to_csv(os.path.join('kaggle-submissions', filename) ,index=False)