#!/usr/bin/env python
# coding: utf-8

# # Importing necessary libraries:

# ******************************************************

# In[1]:


import numpy as np


# In[2]:


from numba import jit


# *********************************************

# # Calculating the source term:

# *************************************

# ## Assume the solution is $$\phi = \left(x+\frac{e^{\frac{ux}{\Gamma^{\phi}}}-1}{1-e^{\frac{u}{\Gamma^{\phi}}}}\right)\left(y+\frac{e^{\frac{vy}{\Gamma^{\phi}}}-1}{1-e^{\frac{v}{\Gamma^{\phi}}}}\right)$$ 

# ##  Then put it in the equation to get the source term

# In[3]:


@jit(nopython=True)
def Source_Cal_diff(x,y,Gamma_phi,u,v):
    """
    Input:
    x: x-coordinate of the centroid of the element
    y: y-coordinate of the centroid of the element
    Gamma_phi: Diffusion coefficient
    
    Output:
    source: soruce evaluated at the centroid of the element
    """
    term_x = (x + (((np.exp(u*x/Gamma_phi) - 1))/(1-(np.exp(u/Gamma_phi)))))
    term_y = (y + (((np.exp(v*y/Gamma_phi) - 1))/(1-(np.exp(v/Gamma_phi)))))
    term_x_derivative = (1 + (((u/Gamma_phi)*(np.exp(u*x/Gamma_phi)))/(1-(np.exp(u/Gamma_phi)))))
    term_x_d_derivative = (((((u/Gamma_phi)**2)*(np.exp(u*x/Gamma_phi)))/(1-(np.exp(u/Gamma_phi)))))
    term_y_derivative = (1 + (((v/Gamma_phi)*(np.exp(v*y/Gamma_phi)))/(1-(np.exp(v/Gamma_phi)))))
    term_y_d_derivative = (((((v/Gamma_phi)**2)*(np.exp(v*y/Gamma_phi)))/(1-(np.exp(v/Gamma_phi)))))
    source = Gamma_phi*((term_x_d_derivative*term_y) + (term_x*term_y_d_derivative))
    return source


# In[4]:


@jit(nopython=True)
def Source_Cal_diff_Elements(Num_Triangles,Element_cen,Gamma_phi,u,v):
    """
    Input:
    Num_Triangles: Number of triangles
    Element_cen: Centroid of the elements
    Gamma_phi: Diffusion coefficient
    
    Output:
    Element_Source: Format:  [Element,Source]
    """
    Element_Source = np.zeros((Num_Triangles,2))
    
    for i in range(Num_Triangles):
        Element = Element_cen[i,0]
        Cen = Element_cen[i,1:]
        x,y = Cen
        Source = Source_Cal_diff(x,y,Gamma_phi,u,v)
        Element_Source[i,0] = Element
        Element_Source[i,1] = Source
        
    return Element_Source


# In[5]:


@jit(nopython=True)
def Source_Cal_advec(x,y,rho,Gamma_phi,u,v):
    """
    Input:
    x: x-coordinate of the centroid of the element
    y: y-coordinate of the centroid of the element
    Gamma_phi: Diffusion coefficient
    
    Output:
    source: soruce evaluated at the centroid of the element
    """
    term_x = (x + (((np.exp(u*x/Gamma_phi) - 1))/(1-(np.exp(u/Gamma_phi)))))
    term_y = (y + (((np.exp(v*y/Gamma_phi) - 1))/(1-(np.exp(v/Gamma_phi)))))
    term_x_derivative = (1 + (((u/Gamma_phi)*(np.exp(u*x/Gamma_phi)))/(1-(np.exp(u/Gamma_phi)))))
    term_x_d_derivative = (((((u/Gamma_phi)**2)*(np.exp(u*x/Gamma_phi)))/(1-(np.exp(u/Gamma_phi)))))
    term_y_derivative = (1 + (((v/Gamma_phi)*(np.exp(v*y/Gamma_phi)))/(1-(np.exp(v/Gamma_phi)))))
    term_y_d_derivative = (((((v/Gamma_phi)**2)*(np.exp(v*y/Gamma_phi)))/(1-(np.exp(v/Gamma_phi)))))
    source = (rho*u*term_x_derivative*term_y) + (rho*v*term_x*term_y_derivative)
    return source


# In[6]:


@jit(nopython=True)
def Source_Cal_advec_Elements(Num_Triangles,Element_cen,rho,Gamma_phi,u,v):
    """
    Input:
    Num_Triangles: Number of triangles
    Element_cen: Centroid of the elements
    Gamma_phi: Diffusion coefficient
    
    Output:
    Element_Source: Format:  [Element,Source]
    """
    Element_Source = np.zeros((Num_Triangles,2))
    
    for i in range(Num_Triangles):
        Element = Element_cen[i,0]
        Cen = Element_cen[i,1:]
        x,y = Cen
        Source = Source_Cal_advec(x,y,rho,Gamma_phi,u,v)
        Element_Source[i,0] = Element
        Element_Source[i,1] = Source
        
    return Element_Source


# In[7]:


@jit(nopython=True)
def Elements_Mass_Calculate(rho,Element_Area):
    """
    Input: 
    rho: Density
    Element_Area: Area of the element
    
    Output:
    Element_Mass: Format: [Element, Mass]
    """
    Element_Mass = Element_Area.copy()
    Element_Mass[:,1] = Element_Area[:,1]*rho

    return Element_Mass


# In[8]:


@jit(nopython=True)
def Source_Term_Elements(rho,Element_Area,Element_Source):
    """
    Input: 
    rho: Density
    Element_Area: Area of the element
    Element_Source Source at the centroid of the element
    
    Output:
    Source_Term: 1D array
    """
    
    Source_Term = (Element_Source[:,1]*Element_Area[:,1])
    return Source_Term

