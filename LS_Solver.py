#!/usr/bin/env python
# coding: utf-8

# # Importing necessary libraries:

# ******************************************************

# In[1]:


import numpy as np


# In[2]:


import scipy


# In[3]:


from numba import jit


# # Least Square Solver:

# *************************************

# $$\begin{pmatrix}
# \Delta x_1 & \Delta y_1 \\
# \Delta x_2 & \Delta y_2 \\
# \Delta x_3 & \Delta y_3 \\
# \vdots & \vdots\\
# \Delta x_m & \Delta y_m \\
# \end{pmatrix} 
# \begin{pmatrix}
# \left(\frac{\partial \phi}{\partial x}\right)_P\\
# \left(\frac{\partial \phi}{\partial y}\right)_P\\
# \end{pmatrix}
# =
# \begin{pmatrix}
# \Delta \phi_{N_{k1}} - \phi_P\\
# \Delta \phi_{N_{k2}} - \phi_P\\
# \Delta \phi_{N_{k3}} - \phi_P\\
# \vdots\\
# \Delta \phi_{N_{km}} - \phi_P\\
# \end{pmatrix} $$

# # This can be written as AX = b

# # This can be solved as a Least square problem

# In[4]:


@jit(nopython=True)
def Grad_Phi_LS(Element_Element_Connectivity_new,Element_cen,phi_0):
    """
    Input:
    Element_Element_Connectivity_new: Element Element Connectivity new (Previously Calculated)
    Element_cen: Element cen (Previousy Calculated)
    phi_0: Solution of phi at the last iteration or time step
    
    Output:
    Element_grad_phi: grad phi at the centroid of the element
    """
    dim = 2
    Num_Triangles = Element_Element_Connectivity_new.shape[0]
    Element_grad_phi = np.zeros((Num_Triangles,3))
    
    # Debug script:
    # Element_with_two_boundary_edges = np.zeros((Num_Triangles,2)) - 1
    
    for i in range(Num_Triangles):
        Element = Element_Element_Connectivity_new[i,0]
        Nb_Elements = Element_Element_Connectivity_new[i,1:]
        cen = Element_cen[int(Element),1:]
        num_nb = Nb_Elements.shape[0]
        temp = np.zeros((num_nb,dim))
        temp_rhs = np.zeros((num_nb))
        
        Element_grad_phi[i,0] = Element
        
        # Debug script:
        # print(Element)
        
        for j in range(Nb_Elements.shape[0]):
            
            # Debug script:
            # if j == Element:
                # reflection_ctr = reflection_ctr + 1
            
            cen_nb = Element_cen[int(Nb_Elements[j]),1:]
            p_Nb_vec = cen_nb - cen
            temp[j,:] = p_Nb_vec
            temp_rhs[j] = phi_0[Nb_Elements[j]] - phi_0[Element]
            
            grad = np.linalg.lstsq(temp,temp_rhs,rcond=-1)[0]
            
            # Debug script:
            # print(grad)
            
            Element_grad_phi[i,1:] = grad

    return Element_grad_phi

