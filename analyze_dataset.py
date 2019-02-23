import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_iris, load_linnerud, load_boston

class DataSet(object):
    def __init__(self, file, x_index, y_index):
        self.file = file
        self.x_index = x_index
        self.y_index = y_index

    def read_data__(self):
        data = load_boston()
        X = data["data"]
        y = data["target"]
        print(data.keys())
        print(data['target'])
        print(data['feature_names'])
        feature_names = data["feature_names"]
        boston_df = pd.DataFrame(data=X, columns=feature_names)
        print(boston_df.head())
        plt.rcParams["font.size"] = 14

        use_df = boston_df.copy()
        use_df["MEDV"] = y
        use_cols = ["NOX", "AGE", "TAX","LSTAT","MEDV"]
        plt.rcParams["font.size"] = 14

        pg = sns.pairplot(data = use_df, vars=use_cols)
        pg.savefig('./test_data2.png')

#    def read_data_linnerud(self):
    def read_data(self):
        data = load_linnerud()
        X = data["data"]
        y = data["target"]
        print(data.keys())
        print(data['target'])
        print(data['target_names'])
        print(data['feature_names'])
        feature_names = data["feature_names"]
        linnerud_df = pd.DataFrame(data=X, columns=feature_names)
        print(linnerud_df.head())
        plt.rcParams["font.size"] = 14

        print(y[0:, 0])
        use_df = linnerud_df.copy()
        use_df['Weight'] = y[0:, 0]
        use_df['Waist'] = y[0:, 1]
        use_df['Pulse'] = y[0:, 2]
        use_cols = ['Chins', 'Situps', 'Jumps', 'Weight', 'Waist', 'Pulse']
        pg = sns.pairplot(data = use_df, vars=use_cols)

        pg.savefig('./test_data3.png')

    def read_data_(self):
#    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
        iris_dataset = load_iris()
        print(iris_dataset.keys())
        print(iris_dataset['target_names'])
        print(iris_dataset['feature_names'])
        df = pd.DataFrame(iris_dataset['data'], columns=iris_dataset['feature_names'])
        df['target'] = iris_dataset['target']
        for i, v in enumerate(iris_dataset['target_names']):
            df.loc[df['target'] == i, 'target'] = v
        print(df.describe())
        print(df['target'])
        pg = sns.pairplot(df, hue="target")

        pg.savefig('./test_data.png')

def get_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-f', '--filepath', help='read file path' ,type=str)
    argparser.add_argument('-d', '--dataset_name', help='read dataset name', type=str)
    argparser.add_argument('-l', '--dataset_list', help='print dataset list',action='store_true')
    argparser.add_argument('-x', '--x_index', help='x', nargs=2,type=int)
    argparser.add_argument('-y', '--y_index', help='y',type=int)

    args = argparser.parse_args()
    return args

def print_dataset_list():
    print('test')

if __name__ == "__main__":
    args = get_args()

    if args.filepath:
        print(args.filepath)
    if args.dataset_list:
        print_dataset_list()
    
    data = DataSet("aaa", args.x_index, args.y_index)
    data.read_data()

