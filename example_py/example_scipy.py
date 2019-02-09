#%%

import numpy as np
from scipy import sparse

# Create a 2D NumPy array with a diagonal of ones, and zeros wverywheare else
eye = np.eye(4)
print("NumPy array:\n", eye)

#%%

# Convert the NumPy array to a SciPy sparse matrix in CSR format
# Only the nonzero entries are stored
sparse_matrix = sparse.csr_matrix(eye)
print("\nSciPy sparse CSR matrix:\n", sparse_matrix)

#%%
data = np.ones(4)
row_indices = np.arange(4)
col_indices = np.arange(4)
eye_coo = sparse.coo_matrix((data, (row_indices, col_indices)))
print("COO representation:\n", eye_coo)
