#!/usr/bin/env python
# coding: utf-8

# # Importing necessary libraries:

# ******************************************************

# In[1]:


import numpy as np


# ****************************************

# In[2]:


def Sol_Over_Area(u,v,Gamma_phi,phi_0,Element_Area):
    Anal_sol_over_Area = (0.5 - (1/(1 - (np.exp(u/Gamma_phi)))) - (Gamma_phi/u))*((0.5 - (1/(1 - (np.exp(v/Gamma_phi)))) - (Gamma_phi/v)))
    Num_sol_over_Area = phi_0@Element_Area[:,1]
    return Anal_sol_over_Area, Num_sol_over_Area


# In[3]:


def Analytical_Solution(Element_cen,V,Gamma_phi):
    """
    Input:
    Element_cen: Centroid of the element
    V: np.array([u,v])
    Gamma_phi: Diffusion coefficients
    
    Output:
    Element_cen_phi_Actual: Analytical solution at the cell centroid
    """
    u = V[0]
    v = V[1]
    x = Element_cen[:,1]
    y = Element_cen[:,2]
    term_x = (x + (((np.exp(u*x/Gamma_phi) - 1))/(1-(np.exp(u/Gamma_phi)))))
    term_y = (y + (((np.exp(v*y/Gamma_phi) - 1))/(1-(np.exp(v/Gamma_phi)))))
    Element_cen_phi_Actual = term_x*term_y
    
    return Element_cen_phi_Actual


# In[4]:


def Element_Vertex_Avg_Anal_Sol(Num_Triangles,Element_Node_Connectivity_new,Node_Coordinates,V,Gamma_phi):
    """
    Calculates the average of the analytical solution at the vertices of the element
    """
    Element_Vertex_Avg_sol = np.zeros((Num_Triangles,2))
    for i in range(Element_Node_Connectivity_new.shape[0]):
        Element = Element_Node_Connectivity_new[i,0]
        Nodes = Element_Node_Connectivity_new[i,1:]
        sol = 0
        for j in range(Nodes.shape[0]):
            node = Nodes[j]
            Coordinates = Node_Coordinates[int(node)]
            Coordinates = Coordinates[0:2]
            u = V[0]
            v = V[1]
            x = Coordinates[0]
            y = Coordinates[1]
            term_x = (x + (((np.exp(u*x/Gamma_phi) - 1))/(1-(np.exp(u/Gamma_phi)))))
            term_y = (y + (((np.exp(v*y/Gamma_phi) - 1))/(1-(np.exp(v/Gamma_phi)))))
            sol =  sol + term_x*term_y
        sol = sol/3
        Element_Vertex_Avg_sol[i,0] = Element
        Element_Vertex_Avg_sol[i,1] = sol
        
    return Element_Vertex_Avg_sol

