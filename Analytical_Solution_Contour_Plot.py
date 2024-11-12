#!/usr/bin/env python
# coding: utf-8

# # Importing necessary libraries:

# ******************************************************

# In[1]:


import numpy as np


# In[2]:


import matplotlib.pyplot as plt


# In[3]:


import matplotlib


# In[4]:


from matplotlib.colors import Normalize


# ****************************************

# # Analytical Solution Contour Plot

# *************************************************

# In[5]:


def Analytical_Solution_Contour_Plot(pts,u,v,Gamma_phi,Img_File):
    x = np.linspace(0,1,pts)
    y = np.linspace(0,1,pts)
    xv, yv = np.meshgrid(x, y, indexing='ij')
    term_x = (xv + (((np.exp(u*xv/Gamma_phi) - 1))/(1-(np.exp(u/Gamma_phi)))))
    term_y = (yv + (((np.exp(v*yv/Gamma_phi) - 1))/(1-(np.exp(v/Gamma_phi)))))
    Ans = term_x*term_y
    plt.contourf(x,y,Ans, cmap = "turbo")
    plt.axis("scaled")
    plt.title("Analytical Solution")
    plt.colorbar()
    plt.savefig(Img_File)
    # plt.show()
    plt.close()

