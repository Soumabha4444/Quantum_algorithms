#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# First we implement for T-gate unitary


# In[7]:


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
qpe = QuantumCircuit(4, 3)

# Apply H-Gates to counting qubits:
for qubit in range(3):
    qpe.h(qubit)

# Prepare our eigenstate |psi> = |1>:
qpe.x(3)

# Do the controlled-U operations:
angle = math.pi/4 # T-gate
repetitions = 1
for counting_qubit in range(3):
    for i in range(repetitions):
        qpe.cp(angle, counting_qubit, 3)
    repetitions *= 2

# Do the inverse QFT:
qpe = qpe.compose(QFT(3, inverse=True), range(3))

# Measure of course!
for n in range(3):
    qpe.measure(n,n)

qpe.draw()


# In[2]:


# Let's see the results!
aer_sim = Aer.get_backend('aer_simulator')
shots = 4096
t_qpe = transpile(qpe, aer_sim)
results = aer_sim.run(t_qpe, shots=shots).result()
answer = results.get_counts()

plot_histogram(answer)


# In[ ]:


# Next, we implement for another unitary


# In[9]:


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
qpe = QuantumCircuit(4, 3)

# Apply H-Gates to counting qubits:
for qubit in range(3):
    qpe.h(qubit)

# Prepare our eigenstate |psi> = |1>:
qpe.x(3)

# Do the controlled-U operations:
angle = 2*math.pi/3
repetitions = 1
for counting_qubit in range(3):
    for i in range(repetitions):
        qpe.cp(angle, counting_qubit, 3)
    repetitions *= 2

# Do the inverse QFT:
qpe = qpe.compose(QFT(3, inverse=True), range(3))

# Measure of course!
for n in range(3):
    qpe.measure(n,n)

qpe.draw()


# In[10]:


# Let's see the results!
aer_sim = Aer.get_backend('aer_simulator')
shots = 4096
t_qpe = transpile(qpe, aer_sim)
results = aer_sim.run(t_qpe, shots=shots).result()
answer = results.get_counts()

plot_histogram(answer)


# In[11]:


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
qpe = QuantumCircuit(4, 3)

# Apply H-Gates to counting qubits:
for qubit in range(3):
    qpe.h(qubit)

# Prepare our eigenstate |psi> = |0>:

# Do the controlled-U operations:
angle = 2*math.pi/3
repetitions = 1
for counting_qubit in range(3):
    for i in range(repetitions):
        qpe.cp(angle, counting_qubit, 3)
    repetitions *= 2

# Do the inverse QFT:
qpe = qpe.compose(QFT(3, inverse=True), range(3))

# Measure of course!
for n in range(3):
    qpe.measure(n,n)

qpe.draw()


# In[12]:


# Let's see the results!
aer_sim = Aer.get_backend('aer_simulator')
shots = 4096
t_qpe = transpile(qpe, aer_sim)
results = aer_sim.run(t_qpe, shots=shots).result()
answer = results.get_counts()

plot_histogram(answer)


# In[13]:


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
qpe = QuantumCircuit(4, 3)

# Apply H-Gates to counting qubits:
for qubit in range(3):
    qpe.h(qubit)

# Prepare our eigenstate |psi> = |+>:
qpe.h(3)

# Do the controlled-U operations:
angle = 2*math.pi/3
repetitions = 1
for counting_qubit in range(3):
    for i in range(repetitions):
        qpe.cp(angle, counting_qubit, 3)
    repetitions *= 2

# Do the inverse QFT:
qpe = qpe.compose(QFT(3, inverse=True), range(3))

# Measure of course!
for n in range(3):
    qpe.measure(n,n)

qpe.draw()


# In[14]:


# Let's see the results!
aer_sim = Aer.get_backend('aer_simulator')
shots = 4096
t_qpe = transpile(qpe, aer_sim)
results = aer_sim.run(t_qpe, shots=shots).result()
answer = results.get_counts()

plot_histogram(answer)


# In[ ]:




