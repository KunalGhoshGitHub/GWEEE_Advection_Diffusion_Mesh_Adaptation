#!/usr/bin/env python
# coding: utf-8

# # Importing necessary libraries:

# ******************************************************

# In[1]:


import numpy as np


# In[2]:


from numba import jit


# In[3]:


from LS_Solver import *


# ****************************************************

# # Error

# ************************************************

# In[4]:


@jit(nopython=True)
def Error_Estimate_CDAS(Element_cen_phi_Actual,phi_0,Element_Area):
    """
    Calculates the exact error using the solution at the centroid of the element
    """
    # Centroid Difference with Analytical Solution (CDAS)
    error = (Element_cen_phi_Actual - phi_0)*Element_Area[:,1]
    return error


# In[5]:


@jit(nopython=True)
def Error_Estimate_CDEVAAS(Element_Vertex_Avg_sol,phi_0,Element_Area):
    """
    Calculates the exact error using the average of the solution at the vertices of the element
    """
    # Centroid Difference with Element Vertex Averaged Analytical Solution (CDEVAAS)
    error = (Element_Vertex_Avg_sol[:,1] - phi_0)*Element_Area[:,1]
    return error


# In[6]:


@jit(nopython=True)
def Error_Estimate_EMGA(phi_0,phi_0_u,Element_Element_Connectivity_new,Element_cen,Element_Area):
    """
    Calculates the error estimate using:
    (phi_0 - phi_0_u)*(|Gradient(phi_0)|)*(Element_Area)
    """
    error = np.abs(phi_0 - phi_0_u)
    Element_grad_phi_LS = Grad_Phi_LS(Element_Element_Connectivity_new,Element_cen,phi_0)
    mod_grad = (np.sum(Element_grad_phi_LS[:,1:]**2,axis=1))**0.5
    error = error*mod_grad*Element_Area[:,1]
    return error,mod_grad

