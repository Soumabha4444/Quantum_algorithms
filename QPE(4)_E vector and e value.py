#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np

A = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]]) # 4x4 NumPy array
print(A) # Print matrix

Eval, Evec = np.linalg.eig(A) # Calculate eigenvalues and eigenvectors
print(Eval) # Print eigenvalues
print(Evec) # Print eigenvectors


# In[10]:


print(Evec[:, 1])  # Print the second eigenvector
print(Eval[1])  # Print the second eigenvector


# In[11]:


# Initializing a 2-qubit system in the obtained eigenvector


# In[3]:


from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit import IBMQ, Aer, transpile, assemble
from qiskit.visualization import plot_histogram

# Define the statevector
desired_state = Statevector([0, 0.70710678, 0, -0.70710678])

# Create a quantum circuit
qc = QuantumCircuit(3,2)

# Initialize the quantum circuit to the statevector
qc.initialize(desired_state, [0, 1])  # Apply the statevector to qubits 0 and 1

# Draw the circuit (optional)
print(qc)


# In[4]:


# In this case, our statevector is not normalized. So we need to normalize it


# In[39]:


from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit import IBMQ, Aer, transpile, assemble
from qiskit.visualization import plot_histogram

# Define the statevector
desired_state = [0, 0.70710678, 0, -0.70710678]  # Unnormalized statevector

# Calculate the normalization factor
norm_factor = sum(abs(c)**2 for c in desired_state) ** 0.5

# Normalize the statevector
normalized_state = [c / norm_factor for c in desired_state]

# Create a quantum circuit
qc = QuantumCircuit(3,2)

# Initialize the quantum circuit to the normalized statevector
qc.initialize(Statevector(normalized_state), [0, 1])  # Apply the normalized state to qubits 0 and 1

# Draw the circuit (optional)
print(qc)


# In[40]:


# Perform some operations on the initialized qubits
# The initialized state is |-1>
qc.x(0)
qc.h(1)
qc.x(1)

# The output should be |00> when measured
for n in range(2):
    qc.measure(n,n)
qc.draw()


# In[41]:


# Let's see the results!
aer_sim = Aer.get_backend('aer_simulator')
t_qc = transpile(qc, aer_sim)
results = aer_sim.run(t_qc, shots=1024).result()
answer = results.get_counts()

plot_histogram(answer)


# In[1]:


# Let us find eigenvalues and eigenvectors of some unitaries


# In[5]:


import numpy as np

A = (1 / np.sqrt(2)) * np.array([[1, 1j], [1j, 1]])
print(A) # Print matrix

Eval, Evec = np.linalg.eig(A) # Calculate eigenvalues and eigenvectors
print(Eval) # Print eigenvalues
print(Evec) # Print eigenvectors


# In[1]:


import numpy as np

A = (1 / np.sqrt(3)) * np.array([[1, 1 + 1j], [1 - 1j, -1]])
print(A) # Print matrix

Eval, Evec = np.linalg.eig(A) # Calculate eigenvalues and eigenvectors
print(Eval) # Print eigenvalues
print(Evec) # Print eigenvectors


# In[ ]:




