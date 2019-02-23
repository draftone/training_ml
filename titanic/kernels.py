import numpy as np
import pandas as pd
import scipy as sp
import IPython
from IPython import display
import sklearn
import random
import time
import warnings
warnings.filterwarnings('ignore')

from subprocess import check_output

from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
# from xgboost import XGBClassifier

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics

#Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
#from pandas.tools.plotting import scatter_matrix

import analyze_dataset as ad



if __name__ == "__main__":
    ana = ad.DataSet('data/train.csv', 'data/test.csv', 0 , 0)

    data_raw = pd.read_csv('data/train.csv')
    data_val = pd.read_csv('data/test.csv')
    data1 = data_raw.copy(deep = True)
    data_cleaner = [data1, data_val]

    for dataset in data_cleaner:
        #dataset['Age'].fillna(dataset['Age'].median(), inplace = True)
        dataset['Age'].fillna(dataset['Age'].median(), inplace = True)
        dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace = True)
    
    ana.print_describe(data_cleaner[0])