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


# # Post Processing:

# **************************************************

# In[5]:


def Post_Process(phi,string,Element_Node_Connectivity_new,Node_Coordinates,Img_file):
    """
    Input:
    phi: Scalar
    string: String for the title
    Element_Node_Connectivity_new:
    Node_Coordinates:
    
    Output:
    A contour plot showing the value of phi in different elements
    """
    
    temp = np.zeros((4,2))
    x = Node_Coordinates[:,0]
    y = Node_Coordinates[:,1]
    # cmap = "YlGn"
    # cmap = "viridis"
    # cmap = "rainbow"
    # cmap = "jet"
    cmap = "turbo"

    cmap = matplotlib.colormaps.get_cmap(cmap)
    
    norm = Normalize(vmin=min(phi), vmax=max(phi))  # Normalize the scalar values

    fig, ax = plt.subplots()
    
    for i in range(Element_Node_Connectivity_new.shape[0]):
        Nodes = Element_Node_Connectivity_new[i,1:]
        Nodes = np.array(Nodes,dtype = int)
        temp[:3,0] = x[Nodes]
        temp[:3,1] = y[Nodes]
        temp[-1,0] = x[Nodes[0]]
        temp[-1,1] = y[Nodes[0]]
        X = temp[:,0]
        Y = temp[:,1]
        ax.plot(X,Y,"k")
        # color = plt.cm.get_cmap(cmap)((phi[i] - min(phi)) / (max(phi) - min(phi)))
        color = cmap(norm(phi[i]))
        
        ax.fill(X,Y,color=color,linewidth = 1)
        
        # Debug Script:
        # plt.plot(x[Nodes[0]],y[Nodes[0]],"-b*")
        
    scalar_map = plt.cm.ScalarMappable(cmap=cmap,norm=plt.Normalize(vmin=min(phi), vmax=max(phi)))
    scalar_map.set_array(phi)
    plt.colorbar(scalar_map,ax = ax)
    # cbar = plt.colorbar(scalar_map)
    # cbar.set_label(r'$\Phi$: '+string)
    plt.title(string)
    plt.axis("scaled")
    plt.savefig(Img_file)
    # plt.show()
    plt.close()


# In[6]:


def Post_Process_Without_Grid(phi,string,Element_Node_Connectivity_new,Node_Coordinates,Img_file):
    """
    Input:
    phi: Scalar
    string: String for the title
    Element_Node_Connectivity_new:
    Node_Coordinates:
    
    Output:
    A contour plot showing the value of phi in different elements
    """
    
    temp = np.zeros((4,2))
    x = Node_Coordinates[:,0]
    y = Node_Coordinates[:,1]
    # cmap = "YlGn"
    # cmap = "viridis"
    # cmap = "rainbow"
    # cmap = "jet"
    cmap = "turbo"

    cmap = matplotlib.colormaps.get_cmap(cmap)
    
    norm = Normalize(vmin=min(phi), vmax=max(phi))  # Normalize the scalar values

    fig, ax = plt.subplots()
    
    for i in range(Element_Node_Connectivity_new.shape[0]):
        Nodes = Element_Node_Connectivity_new[i,1:]
        Nodes = np.array(Nodes,dtype = int)
        temp[:3,0] = x[Nodes]
        temp[:3,1] = y[Nodes]
        temp[-1,0] = x[Nodes[0]]
        temp[-1,1] = y[Nodes[0]]
        X = temp[:,0]
        Y = temp[:,1]
        # ax.plot(X,Y,"k")
        # color = plt.cm.get_cmap(cmap)((phi[i] - min(phi)) / (max(phi) - min(phi)))
        color = cmap(norm(phi[i]))
        
        ax.fill(X,Y,color=color)
        
        # Debug Script:
        # plt.plot(x[Nodes[0]],y[Nodes[0]],"-b*")
        
    scalar_map = plt.cm.ScalarMappable(cmap=cmap,norm=plt.Normalize(vmin=min(phi), vmax=max(phi)))
    scalar_map.set_array(phi)
    plt.colorbar(scalar_map,ax = ax)
    # cbar = plt.colorbar(scalar_map)
    # cbar.set_label(r'$\Phi$: '+string)
    plt.title(string)
    plt.axis("scaled")
    plt.savefig(Img_file)
    # plt.show()
    plt.close()

