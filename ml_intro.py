#%%
import pandas as pd
import matplotlib
import numpy as np
import scipy as sp
import IPython
import sklearn

#%%
from sklearn.datasets import load_iris
iris_dataset = load_iris()

#%%
print("Keys of iris_dataset:\n", iris_dataset.keys())

#%%
print(iris_dataset['DESCR'][:193] + "\n...")

