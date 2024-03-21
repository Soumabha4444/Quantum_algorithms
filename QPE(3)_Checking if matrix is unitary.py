#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np

# Define your matrix
matrix = np.array([[0.707, 0.707], [0.707, -0.707]])
print(matrix)

# Verify if the matrix is square
if matrix.shape[0] != matrix.shape[1]:
    print("The matrix is not square, therefore not unitary.")
else:
    # Compute the conjugate transpose
    conjugate_transpose = np.conjugate(np.transpose(matrix))

    # Multiply the original matrix with its conjugate transpose
    product = np.matmul(matrix, conjugate_transpose)

    # Check if the resulting product is equal to the identity matrix
    identity_matrix = np.identity(matrix.shape[0])
    
    # Check if the product is close to the identity matrix
    if np.allclose(product, identity_matrix):
        print("The matrix is unitary.")
    else:
        print("The matrix is not unitary.")


# In[8]:


import numpy as np

# Define your matrix
matrix = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])
print(matrix)

# Verify if the matrix is square
if matrix.shape[0] != matrix.shape[1]:
    print("The matrix is not square, therefore not unitary.")
else:
    # Compute the conjugate transpose
    conjugate_transpose = np.conjugate(np.transpose(matrix))

    # Multiply the original matrix with its conjugate transpose
    product = np.matmul(matrix, conjugate_transpose)

    # Check if the resulting product is equal to the identity matrix
    identity_matrix = np.identity(matrix.shape[0])
    
    # Check if the product is close to the identity matrix
    if np.allclose(product, identity_matrix):
        print("The matrix is unitary.")
    else:
        print("The matrix is not unitary.")


# In[9]:


import numpy as np

# Define your matrix
matrix = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]])
print(matrix)

# Verify if the matrix is square
if matrix.shape[0] != matrix.shape[1]:
    print("The matrix is not square, therefore not unitary.")
else:
    # Compute the conjugate transpose
    conjugate_transpose = np.conjugate(np.transpose(matrix))

    # Multiply the original matrix with its conjugate transpose
    product = np.matmul(matrix, conjugate_transpose)

    # Check if the resulting product is equal to the identity matrix
    identity_matrix = np.identity(matrix.shape[0])
    
    # Check if the product is close to the identity matrix
    if np.allclose(product, identity_matrix):
        print("The matrix is unitary.")
    else:
        print("The matrix is not unitary.")


# In[10]:


import numpy as np

# Define your matrix
matrix = (1 / np.sqrt(2)) * np.array([[1, 1j], [1j, 1]])
print(matrix)

# Verify if the matrix is square
if matrix.shape[0] != matrix.shape[1]:
    print("The matrix is not square, therefore not unitary.")
else:
    # Compute the conjugate transpose
    conjugate_transpose = np.conjugate(np.transpose(matrix))

    # Multiply the original matrix with its conjugate transpose
    product = np.matmul(matrix, conjugate_transpose)

    # Check if the resulting product is equal to the identity matrix
    identity_matrix = np.identity(matrix.shape[0])
    
    # Check if the product is close to the identity matrix
    if np.allclose(product, identity_matrix):
        print("The matrix is unitary.")
    else:
        print("The matrix is not unitary.")


# In[11]:


import numpy as np

# Define your matrix
matrix = np.array([[np.exp(1j * np.pi / 4), 0], [0, -np.exp(1j * np.pi / 4)]])
print(matrix)

# Verify if the matrix is square
if matrix.shape[0] != matrix.shape[1]:
    print("The matrix is not square, therefore not unitary.")
else:
    # Compute the conjugate transpose
    conjugate_transpose = np.conjugate(np.transpose(matrix))

    # Multiply the original matrix with its conjugate transpose
    product = np.matmul(matrix, conjugate_transpose)

    # Check if the resulting product is equal to the identity matrix
    identity_matrix = np.identity(matrix.shape[0])
    
    # Check if the product is close to the identity matrix
    if np.allclose(product, identity_matrix):
        print("The matrix is unitary.")
    else:
        print("The matrix is not unitary.")


# In[18]:


import numpy as np

# Define your matrix
matrix = (1 / np.sqrt(3)) * np.array([[1, 1 + 1j], [1 - 1j, -1]])
print(matrix)

# Verify if the matrix is square
if matrix.shape[0] != matrix.shape[1]:
    print("The matrix is not square, therefore not unitary.")
else:
    # Compute the conjugate transpose
    conjugate_transpose = np.conjugate(np.transpose(matrix))

    # Multiply the original matrix with its conjugate transpose
    product = np.matmul(matrix, conjugate_transpose)

    # Check if the resulting product is equal to the identity matrix
    identity_matrix = np.identity(matrix.shape[0])
    
    # Check if the product is close to the identity matrix
    if np.allclose(product, identity_matrix):
        print("The matrix is unitary.")
    else:
        print("The matrix is not unitary.")


# In[20]:


import numpy as np

# Define your matrix
x = (4 * np.pi)/7
matrix = np.array([[np.cos(x), -np.sin(x)], [np.sin(x), np.cos(x)]])
print(matrix)

# Verify if the matrix is square
if matrix.shape[0] != matrix.shape[1]:
    print("The matrix is not square, therefore not unitary.")
else:
    # Compute the conjugate transpose
    conjugate_transpose = np.conjugate(np.transpose(matrix))

    # Multiply the original matrix with its conjugate transpose
    product = np.matmul(matrix, conjugate_transpose)

    # Check if the resulting product is equal to the identity matrix
    identity_matrix = np.identity(matrix.shape[0])
    
    # Check if the product is close to the identity matrix
    if np.allclose(product, identity_matrix):
        print("The matrix is unitary.")
    else:
        print("The matrix is not unitary.")


# In[ ]:




