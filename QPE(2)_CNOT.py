#!/usr/bin/env python
# coding: utf-8

# In[10]:


#initialization
import matplotlib.pyplot as plt
import numpy as np
import math

# importing Qiskit
from qiskit import IBMQ, Aer, transpile, assemble
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister

# import basic plot tools and circuits
from qiskit.visualization import plot_histogram
from qiskit.circuit.library import QFT

# Create and set up circuit
qpe = QuantumCircuit(6, 4)

# Apply H-Gates to counting qubits:
for qubit in range(4):
    qpe.h(qubit)

# Prepare our eigenstate |psi> = |-1>:
qpe.x(4)
qpe.x(5)
qpe.h(5)

# Do the controlled-U operations:
repetitions = 1
for counting_qubit in range(4):
    for i in range(repetitions):
        qpe.ccx(counting_qubit, 4, 5);
    repetitions *= 2

# Do the inverse QFT:
qpe = qpe.compose(QFT(4, inverse=True), range(4))

# Measure of course!
for n in range(4):
    qpe.measure(n,n)

qpe.draw()


# In[11]:


# Let's see the results!
aer_sim = Aer.get_backend('aer_simulator')
shots = 4096
t_qpe = transpile(qpe, aer_sim)
results = aer_sim.run(t_qpe, shots=shots).result()
answer = results.get_counts()

plot_histogram(answer)


# In[14]:


#initialization
import matplotlib.pyplot as plt
import numpy as np
import math

# importing Qiskit
from qiskit import IBMQ, Aer, transpile, assemble
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister

# import basic plot tools and circuits
from qiskit.visualization import plot_histogram
from qiskit.circuit.library import QFT

# Create and set up circuit
qpe = QuantumCircuit(6, 4)

# Apply H-Gates to counting qubits:
for qubit in range(4):
    qpe.h(qubit)

# Prepare our eigenstate |psi> = |10>:
qpe.x(5)

# Do the controlled-U operations:
repetitions = 1
for counting_qubit in range(4):
    for i in range(repetitions):
        qpe.ccx(counting_qubit, 4, 5);
    repetitions *= 2

# Do the inverse QFT:
qpe = qpe.compose(QFT(4, inverse=True), range(4))

# Measure of course!
for n in range(4):
    qpe.measure(n,n)

qpe.draw()


# In[15]:


# Let's see the results!
aer_sim = Aer.get_backend('aer_simulator')
shots = 4096
t_qpe = transpile(qpe, aer_sim)
results = aer_sim.run(t_qpe, shots=shots).result()
answer = results.get_counts()

plot_histogram(answer)


# In[17]:


#initialization
import matplotlib.pyplot as plt
import numpy as np
import math

# importing Qiskit
from qiskit import IBMQ, Aer, transpile, assemble
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister

# import basic plot tools and circuits
from qiskit.visualization import plot_histogram
from qiskit.circuit.library import QFT

# Create and set up circuit
qpe = QuantumCircuit(6, 4)

# Apply H-Gates to counting qubits:
for qubit in range(4):
    qpe.h(qubit)

# Prepare our eigenstate |psi> = |01>:
qpe.x(4)

# Do the controlled-U operations:
repetitions = 1
for counting_qubit in range(4):
    for i in range(repetitions):
        qpe.ccx(counting_qubit, 4, 5);
    repetitions *= 2

# Do the inverse QFT:
qpe = qpe.compose(QFT(4, inverse=True), range(4))

# Measure of course!
for n in range(4):
    qpe.measure(n,n)

qpe.draw()


# In[18]:


# Let's see the results!
aer_sim = Aer.get_backend('aer_simulator')
shots = 4096
t_qpe = transpile(qpe, aer_sim)
results = aer_sim.run(t_qpe, shots=shots).result()
answer = results.get_counts()

plot_histogram(answer)


# In[ ]:




