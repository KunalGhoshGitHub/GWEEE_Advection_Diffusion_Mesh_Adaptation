#!/usr/bin/env python
# coding: utf-8

# # Importing necessary libraries:

# *************************************************

# In[1]:


import numpy as np


# In[2]:


from Solver_Adapt import *


# *********************************************

# # Physical Variables:

# *******************************************************

# In[3]:


rho = 1


# In[4]:


pts = 250
# Analytical_Solution_Contour_Plot(pts,u,v,Gamma_phi)


# In[5]:


# Gamma_phi_range = np.linspace(0.1,1,5)
# u_range = np.linspace(1,10,5)
# v_range = np.linspace(1,10,5)

Gamma_phi_range = [0.4]
u_range = [10]
v_range = [10]


# *******************************************

# # Mesh Controls

# **********************************************

# In[6]:


Target_Dof = 1024


# In[7]:


p = 0
D = 2
q = 1


# In[8]:


Mesh_Adaptation_Cycles = 10


# **********************************************************

# # Solve

# ****************************************************************

# In[9]:


for Gamma_phi in Gamma_phi_range:
    for u in u_range:
        for v in v_range:
            print(f"Gamma_Phi: {Gamma_phi}, u: {u}, v: {v}")
            V = np.array([u,v])
            Mesh_Adapt(Mesh_Adaptation_Cycles,rho,Gamma_phi,V,pts,Target_Dof,p,D,q)


# In[ ]:




