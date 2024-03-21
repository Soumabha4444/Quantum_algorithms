#!/usr/bin/env python
# coding: utf-8

# In[32]:


# IMPLEMENTING SHOR'S ALGORITHM CLASSICALLY


# In[2]:


N = 15 # The number we want to factor
a = 13 # taking a number co-prime with N


# In[3]:


import math
math.gcd(a, N) # Checking if a, N are co-prime


# In[6]:


# Plot the function a^x (mod N)

import matplotlib.pyplot as plt
X = list(range(N)) # List of numbers from 0 to N-1
Y = [a**n % N for n in X] # List of numbers a^x (mod N)

plt.plot(X, Y)
plt.xlabel ('X')
plt.ylabel(f'{a}^X (mod{N})')
plt.show()


# In[9]:


r = X[Y[1:].index(1)+1]
# Y[1:] creates a new list with element of Y starting from index 1, i.e. without Y[0]
# .index(1) returns the index of new list Y[1:] where we first see the value "1"
# We do .index(1)+1 to match the index of X with Y[1:], i.e. ignoring the value X[0]
print(f'r = {r}') # r is the period


# In[13]:


# We write the full algorithm of Shor

if r % 2 == 0: # i.e. if r is even
    x = (a**(r/2.)) % N # "." after 2 ensures floating values
    print(f'x = {x}')
    if ((x+1) % N) != 0:
        print(f'Factors of {N} are {math.gcd((int(x)+1), N)} and {math.gcd((int(x)-1), N)}')
    else:
        print("x+1 is 0 (mod N)")
else:
    print(f'r = {r} is odd')


# In[14]:


# Let us try for another number


# In[27]:


N = 91 # The number we want to factor
a = 66 # taking a number co-prime with N


# In[28]:


import math
math.gcd(a, N) # Checking if a, N are co-prime


# In[29]:


# Plot the function a^x (mod N)

import matplotlib.pyplot as plt
X = list(range(N)) # List of numbers from 0 to N-1
Y = [a**n % N for n in X] # List of numbers a^x (mod N)

plt.plot(X, Y)
plt.xlabel ('X')
plt.ylabel(f'{a}^X (mod{N})')
plt.show()


# In[30]:


r = X[Y[1:].index(1)+1]
print(f'r = {r}') # r is the period


# In[31]:


# We write the full algorithm of Shor

if r % 2 == 0: # i.e. if r is even
    x = (a**(r/2.)) % N # "." after 2 ensures floating values
    print(f'x = {x}')
    if ((x+1) % N) != 0:
        print(f'Factors of {N} are {math.gcd((int(x)+1), N)} and {math.gcd((int(x)-1), N)}')
    else:
        print("x+1 is 0 (mod N)")
else:
    print(f'r = {r} is odd')


# In[1]:


#######################################


# In[4]:


# Calculating the coefficients in |x> register after step-4 (see theory)

import numpy as np
pi = np.pi
for y in range(16):
    coeff = np.exp(-1j*y*3*pi/8) + np.exp(-1j*y*7*pi/8) + np.exp(-1j*y*11*pi/8) + np.exp(-1j*y*15*pi/8)
    if abs(coeff) < 1e-10:
        coeff = 0 # if coeff is very small, i.e. less than e^-10, then we presume it to be 0
    print (f'Coeff of |{y}> : {coeff}')


# In[5]:


# We get: 
# coeff of |0> is 4
# coeff of |4> is 4i
# coeff of |8> is -4
# coeff of |12> is -4i


# In[ ]:


# Calculation for N=105


# In[1]:


import numpy as np
pi = np.pi
for y in range(128):
    coeff = np.exp(-1j*y*pi/64) + np.exp(-1j*y*5*pi/64) + np.exp(-1j*y*9*pi/64) + np.exp(-1j*y*13*pi/64) + np.exp(-1j*y*17*pi/64) + np.exp(-1j*y*21*pi/64) + np.exp(-1j*y*25*pi/64) + np.exp(-1j*y*29*pi/64) + np.exp(-1j*y*33*pi/64) + np.exp(-1j*y*37*pi/64) + np.exp(-1j*y*41*pi/64) + np.exp(-1j*y*45*pi/64) + np.exp(-1j*y*49*pi/64) + np.exp(-1j*y*53*pi/64) + np.exp(-1j*y*57*pi/64) + np.exp(-1j*y*61*pi/64) + np.exp(-1j*y*65*pi/64) + np.exp(-1j*y*69*pi/64) + np.exp(-1j*y*73*pi/64) + np.exp(-1j*y*77*pi/64) + np.exp(-1j*y*81*pi/64) + np.exp(-1j*y*85*pi/64) + np.exp(-1j*y*89*pi/64) + np.exp(-1j*y*93*pi/64) + np.exp(-1j*y*97*pi/64) + np.exp(-1j*y*101*pi/64) + np.exp(-1j*y*105*pi/64) + np.exp(-1j*y*109*pi/64) + np.exp(-1j*y*113*pi/64) + np.exp(-1j*y*117*pi/64) + np.exp(-1j*y*121*pi/64) + np.exp(-1j*y*125*pi/64)
    if abs(coeff) < 1e-10:
        coeff = 0 # if coeff is very small, i.e. less than e^-10, then we presume it to be 0
    print (f'Coeff of |{y}> : {coeff}')


# In[ ]:




