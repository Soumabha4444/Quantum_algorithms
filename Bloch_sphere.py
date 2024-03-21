#!/usr/bin/env python
# coding: utf-8

# In[9]:


from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_bloch_multivector


# In[10]:


qc = QuantumCircuit(1)
qc.h(0)
qc.draw()


# In[11]:


sim = AerSimulator()
qc.save_statevector()
transpiled_circuit = transpile(qc, sim)
state = sim.run(transpiled_circuit).result().get_statevector()
plot_bloch_multivector(state)


# In[12]:


qc2 = QuantumCircuit(1)
qc2.x(0)
qc2.h(0)
qc2.draw()


# In[13]:


sim = AerSimulator()
qc2.save_statevector()
transpiled_circuit2 = transpile(qc2, sim)
state2 = sim.run(transpiled_circuit2).result().get_statevector()
plot_bloch_multivector(state2)


# In[ ]:




