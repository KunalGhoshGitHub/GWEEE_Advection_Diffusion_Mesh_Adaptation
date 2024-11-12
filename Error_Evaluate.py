#!/usr/bin/env python
# coding: utf-8

# # Importing necessary libraries:

# ******************************************************

# In[1]:


import numpy as np


# ****************************************************

# In[2]:


def Cen_Error_Data_Writer(Error_Discrete,Element_Area,Anal_sol_over_Area,Num_sol_over_Area,Num_Triangles,u,v,Gamma_phi):
    L1 = Error_Discrete.sum()
    L2 = (np.sum(Error_Discrete**2))**0.5
    Linf = Error_Discrete.max()
    E_A = np.sum(Error_Discrete*Element_Area[:,1])
    Error_File = open("Error_Data.csv","a+")
    Error_Data = f"{Anal_sol_over_Area - Num_sol_over_Area},{Anal_sol_over_Area},{Num_sol_over_Area},{Num_Triangles},{E_A},{L1},{L2},{Linf},{u},{v},{Gamma_phi}\n"
    Error_File.write(Error_Data)
    Error_File.close()


# In[3]:


def Vertex_Based_Error_Data_Writer(Element_Vertex_Avg_sol,phi_0,Num_Triangles,Element_Area,u,v,Gamma_phi):
    Vertex_Error = np.abs(Element_Vertex_Avg_sol[:,1] - phi_0)
    Error_File = open("Vertex_Based_Error_Data.csv","a+")
    Error_Data = f"{Vertex_Error.sum()},{(np.sum(Vertex_Error**2))**0.5},{Vertex_Error.max()},{Num_Triangles},{np.sum(Vertex_Error*Element_Area[:,1])},{u},{v},{Gamma_phi}\n"
    Error_File.write(Error_Data)
    Error_File.close()

