#!/usr/bin/env python
# coding: utf-8

# # Importing necessary libraries:

# ******************************************************

# In[1]:


import numpy as np


# In[2]:


from numba import jit


# ****************************************************

# In[3]:


@jit(nopython=True)
def Calculate_Optimal_Mesh_Density(Error_Discrete,Element_Area,Target_Dof,p,D,q):
    """
    Calculates the optimal mesh density (Rangarajan's Thesis: d*)
    """
    I_c = Element_Area[:,1]
    var_0 = ((p+1)+D)/D
    var_1 = D*q/((q*(p+1))+D)
    n_h = Target_Dof
    kappa = I_c
    alpha = (3**0.5)/4
    N = alpha*n_h
    A_bar = Error_Discrete*(var_0)*(alpha**((p+1)/D))*(kappa**(-var_0))
    d = N*(A_bar**(var_1))/(np.sum((A_bar**(var_1))*kappa))

    return d


# ## We need to scale the metric

# In[4]:


@jit(nopython=True)
def Metric_Scale_Calculate(Target_Dof,Num_Triangles,metric_term_a_element,Element_Area):
    """
    Input:
    Target_Dof: Target number of the cells
    Num_Triangles: Number of the triangles
    metric_term_a_element: a term of the metric
    Element_Area: Area of the element
    
    Output:
    metric_scale: The scale to ensure Target_Dof
    """
    dof_met = 0

    for i in range(Num_Triangles):
        det = metric_term_a_element[i]**2
        dof_met = dof_met + ((det**0.5)*Element_Area[i,1])
    dof_met = dof_met*4/(3**0.5)
    # dof_met = dof_met/0.46

    dof_t = Target_Dof

    metric_scale = dof_t/dof_met
    
    return metric_scale


# In[5]:


@jit(nopython=True)
def Node_Wise_Metric(metric_term_a_element,Node_Coordinates,Element_Node_Connectivity_new,Element_Area):
    Num_Triangles = metric_term_a_element.shape[0]
    Node_metric = np.zeros(Node_Coordinates.shape[0])
    for i in range(Node_Coordinates.shape[0]):
        Element_Area_sum = 0
        for j in range(Num_Triangles):
            Nodes = Element_Node_Connectivity_new[j,1:]
            for k in range(Nodes.shape[0]):
                if Nodes[k] == i:
                    Node_metric[i] = Node_metric[i] + (metric_term_a_element[j]*Element_Area[j,1])
                    Element_Area_sum = Element_Area_sum + Element_Area[j,1]
        Node_metric[i] = Node_metric[i]/Element_Area_sum
    return Node_metric


# In[6]:


def metric_file_writer(adj_metric_file,Node_Coordinates,Node_metric):
    Adj_metric_string = "" + str(Node_Coordinates.shape[0])+ " 3\n"
    
    for i in range(Node_Coordinates.shape[0]):
        Adj_metric_string = Adj_metric_string +str(Node_metric[i]) + "  0  " + str(Node_metric[i])  +"\n"

    metric_file = open(adj_metric_file,"w")
    metric_file.write(Adj_metric_string)
    status = metric_file.close()
    
    return status


# In[7]:


def Adapted_mesh_file_writer(Adapted_mesh_file_name):
    Adapted_mesh_file = open(f"{Adapted_mesh_file_name}","r")
    Adapted_mesh_file_content = Adapted_mesh_file.read()
    Adapted_mesh_file.close()
    Adapted_mesh_file_content = Adapted_mesh_file_content.replace("MeshVersionFormatted 0","MeshVersionFormatted 1")
    Adapted_mesh_file = open(f"{Adapted_mesh_file_name}","w")
    Adapted_mesh_file.write(Adapted_mesh_file_content)
    Adapted_mesh_file.close()


# In[8]:


@jit(nopython=True)
def Plot_Edge_Number_Cell_Number(Node_Coordinates,Element_Node_Connectivity,Face_Centroid,Element_cen):
    """
    Input:
    Node_Coordinates: Node Coordinates (Previously Calculated)
    Element_Node_Connectivity: Element Node Connectivity (Previously Calculated)
    Face_Centroid: Face Centroid (Previously Calculated)
    
    Output:
    This function plots mesh with edge number and cell number
    """
    temp = np.zeros((4,2))
    x = Node_Coordinates[:,0]
    y = Node_Coordinates[:,1]
    for i in range(Element_Node_Connectivity.shape[0]):
        Nodes = Element_Node_Connectivity[i,1:]
        Nodes = np.array(Nodes,dtype = int)
        temp[:3,0] = x[Nodes]
        temp[:3,1] = y[Nodes]
        temp[-1,0] = x[Nodes[0]]
        temp[-1,1] = y[Nodes[0]]
        X = temp[:,0]
        Y = temp[:,1]
        plt.plot(X,Y)
        # plt.plot(x[Nodes[0]],y[Nodes[0]],"-b*")
    # plt.plot(Face_Centroid[:,1],Face_Centroid[:,2],"gd")
    # for i in range(Num_Edges):
        # plt.text(Face_Centroid[i,1],Face_Centroid[i,2],int(Face_Centroid[i,0]))
    for i in range(Num_Triangles):
        plt.text(Element_cen[i,1],Element_cen[i,2],int(Element_cen[i,0]))
    plt.axis('scaled')
    plt.show()

