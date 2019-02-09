#%%
import pandas as pd
from IPython.display import display

# create a simple dataset of people
data = {'Name': ["John", "Anna", "Peter", "Linda"],
        'Location' : ["New York", "Paris", "Berlin", "London"],
        'Age' : [24, 13, 53, 33]
       }

data_pandas = pd.DataFrame(data)
# IPython. display allows "pretty printing" of dataframes
# in the Jupyter notebook
display(data_pandas)

#%%

# Select all rows that have an age column greater than 30
display(data_pandas[data_pandas.Age > 30])
