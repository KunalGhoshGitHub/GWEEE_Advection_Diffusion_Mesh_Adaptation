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


# In[5]:


import pyamg


# In[6]:


import scipy


# In[7]:


from LS_Solver import *


# In[8]:


from Mesh_Preprocess import *


# In[9]:


from Source_Term import *


# ****************

# # Solver:

# ********************

# ## Advection-Diffusion Equation:

# # $$\frac{\partial (\rho \phi)}{\partial t} + \nabla \cdot (\rho \mathbf{V} \phi)= \nabla \cdot (\Gamma^{\phi} \nabla \phi) + S$$

# Where:
# 
# $\rho$ is the density of the quantity undergoing diffusion.
# 
# $\phi$ is the scalar field representing the quantity undergoing diffusion.
# 
# $\frac{\partial (\rho \phi)}{\partial t}$ is the rate of change of the product $\rho\phi$ with respect to time.
# 
# $\nabla \cdot$ is the divergence operator.
# 
# $\Gamma^{\phi}$ is the diffusion coefficient tensor.
# 
# $\nabla \phi$ is the gradient of $\phi$, representing the spatial variation of $\phi$.
# 
# $S$ is the source term scalar, representing the rate at which the quantity is being added or removed from the system.

# ## Integrating of both sides of the equation with respect to volume ($dV$) and time ($dt$).

# ### \begin{equation}
# \int_{0}^{t} \int_{V} \frac{\partial (\rho \phi)}{\partial t} \, dV \, dt + \int_{0}^{t} \int_{V}\nabla \cdot (\rho \mathbf{V} \phi)  \, dV \, dt = \int_{0}^{t} \int_{V} \nabla \cdot (\Gamma^{\phi} \nabla \phi) \, dV \, dt + \int_{0}^{t} \int_{V} S \, dV \, dt
# \end{equation}

# ## Applying Gauss Divergence theorem to the divergence term $(\int_{V} \nabla \cdot (\Gamma^{\phi} \nabla \phi) \, dV)$ on the LHS.

# # \begin{equation}
# \int_{V}\nabla \cdot (\rho \mathbf{V} \phi)  \, dV = \oint_{S} (\rho \mathbf{V} \phi) \cdot d\mathbf{S}
# \end{equation}

# ## Applying Gauss Divergence theorem to the divergence term $(\int_{V} \nabla \cdot (\Gamma^{\phi} \nabla \phi) \, dV)$ on the RHS.

# # \begin{equation}
# \int_{V} \nabla \cdot (\Gamma^{\phi} \nabla \phi) \, dV = \oint_{S} (\Gamma^{\phi} \nabla \phi) \cdot d\mathbf{S}
# \end{equation}
# 

# ### \begin{equation}
# \int_{0}^{t} \int_{V} \frac{\partial (\rho \phi)}{\partial t} \, dV \, dt + \int_{0}^{t} \oint_{S} (\rho \mathbf{V} \phi) \cdot d\mathbf{S} \, dt = \int_{0}^{t} \oint_{S} (\Gamma^{\phi} \nabla \phi) \cdot d\mathbf{S} \, dt + \int_{0}^{t} \int_{V} S \, dV \, dt
# \end{equation}

# ## $\mathbf{dS} \text{ is the differential surface area vector.} \\
# \mathbf{n} \text{ is the outward unit normal vector to the surface } S. \\
# dS \text{ is the differential scalar surface area.}
# $

# ### \begin{equation}
# \int_{0}^{t} \int_{V} \frac{\partial (\rho \phi)}{\partial t} \, dV \, dt + \int_{0}^{t} \oint_{S} (\rho \mathbf{V} \phi) \cdot (\mathbf{n} \, dS) \, dt = \int_{0}^{t} \oint_{S} (\Gamma^{\phi} \nabla \phi) \cdot (\mathbf{n} \, dS) \, dt + \int_{0}^{t} \int_{V} S \, dV \, dt
# \end{equation}
# 

# ### \begin{equation}
# \int_{V} \frac{(\rho \phi) - (\rho \phi)_0}{\Delta t} \, dV \, \Delta t + \oint_{S} (\rho \mathbf{V} \phi) \cdot (\mathbf{n} \, dS) \, \Delta t = \oint_{S} (\Gamma^{\phi} \nabla \phi) \cdot (\mathbf{n} \, dS) \, \Delta t + \int_{V} S \, dV \, \Delta t
# \end{equation}
# 

# # \begin{equation}
# \int_{V} \frac{(\rho \phi) - (\rho \phi)_0}{\Delta t} \, dV \, + \oint_{S} (\rho \mathbf{V} \phi) \cdot (\mathbf{n} \, dS) \, = \oint_{S} (\Gamma^{\phi} \nabla \phi) \cdot (\mathbf{n} \, dS) \, + \int_{V} S \, dV
# \end{equation}

# ## Now consider only 1 cell of the mesh

# ## \begin{equation}
# \frac{(\rho \phi) - (\rho \phi)_0}{\Delta t} \, \Delta V \, + \left(\sum_{S} (\rho \mathbf{V} \phi) \cdot (\mathbf{n} \, \Delta S) \right) \, = \left(\sum_{S} (\Gamma^{\phi} \nabla \phi) \cdot (\mathbf{n} \, \Delta S) \right) \, + S \, \Delta V
# \end{equation}

# ## \begin{equation}
# \frac{(\rho \phi) - (\rho \phi)_0}{\Delta t} \, \Delta V \, + \sum_{ip} \left( (\rho \mathbf{V} \phi) \cdot (\mathbf{n} \, \Delta S) \right)_{ip} \, = \sum_{ip} \left((\Gamma^{\phi} \nabla \phi) \cdot (\mathbf{n} \, \Delta S) \right)_{ip} \, + S \, \Delta V
# \end{equation}

# ## Here, ip is the integration point over the surface of the control volume or cell. In this 2D case it is the mid point of the edges.

# ## As, $\rho \Delta V = M$

# ## \begin{equation}
# \frac{(M \phi) - (M \phi)_0}{\Delta t} \,  + \sum_{ip} \left( (\rho \mathbf{V} \phi) \cdot (\mathbf{n} \, \Delta S) \right)_{ip} \, = \sum_{ip} \left((\Gamma^{\phi} \nabla \phi) \cdot (\mathbf{n} \, \Delta S) \right)_{ip} \, + S \, \Delta V
# \end{equation}

# ### See the below diagram
# ### Our control volume of interest is the $\color{blue}{\text{blue P}}$ one and $\color{green}{\text{green N_k}}$ is one of the neighbours

# # If the below diagram is not rendering properly then restart kernel and clear output and run the below block at first

# %reload_ext itikz

# %%itikz --file-prefix implicit-demo- --implicit-pic --scale=1
# \usetikzlibrary{calc}
# 
# % Calculate centroid for the first polygon
# \coordinate (A) at (0,0);
# \coordinate (B) at (1,4);
# \coordinate (C) at (-1,6);
# \coordinate (D) at (-4,5);
# \coordinate (E) at (-3,1);
# \coordinate (centroid1) at (barycentric cs:A=1,B=1,C=1,D=1,E=1);
# 
# % Calculate centroid for the second polygon
# \coordinate (F) at (3,-2);
# \coordinate (G) at (6,-1);
# \coordinate (H) at (7,3);
# \coordinate (I) at (4,6);
# \coordinate (centroid2) at (barycentric cs:A=1,F=1,G=1,H=1,I=1,B=1);
# 
# \draw[fill=green!30] (A) -- (F) -- (G) -- (H) -- (I) -- (B) -- cycle;
# \draw[fill=blue!20] (A) -- (B) -- (C) -- (D) -- (E) -- cycle;
# 
# % Draw centroids
# \filldraw[black] (centroid1) circle (2pt) node[left] {$P$};
# \filldraw[black] (centroid2) circle (2pt) node[right] {$N_k$};
# 
# % Draw midpoint of the common edge
# \coordinate (midpoint) at (barycentric cs:A=1,B=1);
# 
# % Draw centroids
# \filldraw[black] (midpoint) circle (2pt) node[below] {$k$};
# 
# % Draw line between centroids
# \draw[black, thick] (centroid1) -- (centroid2);
# 
# % Draw line between centroid1 and midpoint
# \draw[black, thick,-latex,line width=2.5pt] (centroid1) -- (midpoint);
# 
# % Draw line between centroid2 and midpoint
# \draw[black, thick,-latex,line width=2.5pt] (midpoint) -- (centroid2);
# 
# % Calculate vector AB
# \coordinate (vecAB) at ($(B)-(A)$);
# 
# % Calculate perpendicular line passing through the midpoint
# \coordinate (perpendicular) at ($(midpoint)!1!90:(vecAB)$);
# 
# % Draw perpendicular line
# % \draw[black, thick] (midpoint) -- (perpendicular);
# 
# 
# % Calculate perpendicular line passing through the midpoint
# \coordinate (perpendicular1) at ($(midpoint)!1!-90:(vecAB)$);
# 
# % Draw perpendicular line
# % \draw[black, thick] (midpoint) -- (perpendicular1);
# 
# % Extend the perpendicular lines
# \coordinate (extPerpendicular) at ($(perpendicular)!-1.5cm!(midpoint)$);
# \coordinate (extPerpendicular1) at ($(perpendicular1)!-2.1cm!(midpoint)$);
# 
# % Draw perpendicular lines
# \draw[black, thick] (midpoint) -- (extPerpendicular);
# \draw[black, thick] (midpoint) -- (extPerpendicular1);
# 
# \coordinate (pointOnPerpendicular1) at ($(midpoint)!1cm!(perpendicular1)$);
# \fill [black] ($(midpoint)!(centroid1)!(pointOnPerpendicular1)$) circle [radius=2pt];
# 
# \coordinate (pointOnPerpendicular) at ($(midpoint)!1cm!(perpendicular)$);
# \fill [black] ($(midpoint)!(centroid2)!(pointOnPerpendicular)$) circle [radius=2pt];
# 
# \coordinate (p_prime) at ($(midpoint)!(centroid1)!(pointOnPerpendicular1)$);
# \coordinate (Nk_prime) at ($(midpoint)!(centroid2)!(pointOnPerpendicular)$);
# 
# \draw[black, dashed] (centroid1) -- (p_prime);
# 
# \draw[black, dashed] (centroid2) -- (Nk_prime);
# 
# % Calculate perpendicular line passing through the midpoint
# \coordinate (Nk_prime) at ($(midpoint)!1!180:(p_prime)$);
# 
# \filldraw[black] (Nk_prime) circle (2pt) node[below] [below] {${N_k}^{\prime}$};
# 
# \draw[black, dashed] (Nk_prime) -- (centroid2);
# 
# \draw[black, thick,-latex,line width=2.5pt] (midpoint) -- (Nk_prime);
# 
# \draw[black, thick,-latex,line width=2.5pt] (p_prime) -- (midpoint);
# 
# \node at (p_prime) [below] {$P^{\prime}$};
# \node at (midpoint) [above] {$\vec{n}$};

# ## This diagram is motivated by 
# ## [Ferziger JH, Perić M, Street RL. Computational methods for fluid dynamics. springer; 2019 Aug 16.]
# ## [Chapter: 9,Fig. 9.19: On the approximation of diffusion fluxes for arbitrary polyhedral CVs]
# ## [Page: 312]    

# ## \begin{equation}
# \frac{(M \phi) - (M \phi)_0}{\Delta t} \,  + \sum_{ip} \left( (\rho \mathbf{V} \phi) \cdot (\mathbf{n} \, \Delta S) \right)_{ip} \, = \sum_{ip} \left((\Gamma^{\phi} \nabla \phi) \cdot (\mathbf{n} \, \Delta S) \right)_{ip} \, + S \, \Delta V
# \end{equation}

# ## Consider the second term in the LHS

# ## We assume that $\mathbf{V}$ is known everywhere in the computational domain. But $\phi$ at the ip is not known. So, we did the following (Central Differencing Scheme):
# ## NOTE: ip is k in the below figure

# ## \begin{equation}
# \frac{(M \phi) - (M \phi)_0}{\Delta t} \,  + \sum_{ip} \left( (\rho \mathbf{V}) \cdot (\mathbf{n} \, \Delta S) \right)_{ip} \phi_{ip}\, = \sum_{ip} \left((\Gamma^{\phi} \nabla \phi) \cdot (\mathbf{n} \, \Delta S) \right)_{ip} \, + S \, \Delta V
# \end{equation}

# ## $$a = \min((\mathbf{r}_k - \mathbf{r}_P) \cdot \mathbf{n}, (\mathbf{r}_{Nk} - \mathbf{r}_k) \cdot \mathbf{n})$$

# ## $$\mathbf{r}_{P^{\prime}} = \mathbf{r}_k - a\mathbf{n}, \quad \mathbf{r}_{Nk^{\prime}} = \mathbf{r}_k + a\mathbf{n}$$

# ## $$\phi_{P^{\prime}} = \phi_P + (\nabla \phi)_P \cdot (\mathbf{r}_P' - \mathbf{r}_P), \quad \phi_{{Nk}^{\prime}} = \phi_{Nk} + (\nabla \phi)_{Nk} \cdot (\mathbf{r}_{Nk}' - \mathbf{r}_{Nk})$$

# ## $$\phi_k = \frac{\phi_P' + \phi_{Nk}'}{2}$$

# ## So, 
# ## $$\phi_{ip} = \frac{\phi_P' + \phi_{Nk}'}{2}$$

# ## $$\phi_{ip} = \frac{1}{2}(\phi_P + (\nabla \phi)_P \cdot (\mathbf{r}_P' - \mathbf{r}_P) + \phi_{Nk} + (\nabla \phi)_{Nk} \cdot (\mathbf{r}_{Nk}' - \mathbf{r}_{Nk})) $$

# ## $$\phi_{ip} = \frac{1}{2}(\phi_P + \phi_{Nk} + (\nabla \phi)_P \cdot (\mathbf{r}_P' - \mathbf{r}_P) + (\nabla \phi)_{Nk} \cdot (\mathbf{r}_{Nk}' - \mathbf{r}_{Nk})) $$

# ## $$\phi_{ip} = \frac{1}{2}(\phi_P + \phi_{Nk}) + \frac{1}{2}((\nabla \phi)_P \cdot (\mathbf{r}_P' - \mathbf{r}_P) + (\nabla \phi)_{Nk} \cdot (\mathbf{r}_{Nk}' - \mathbf{r}_{Nk})) $$

# ## We will treat the first term in the RHS implicitly and second term explicitly using the vaules of $\phi$ from the previous iteration.

# ## $$\phi_{ip} = \frac{1}{2}(\phi_P + \phi_{Nk}) + \frac{1}{2}\bigg((\nabla \phi)_P \cdot (\mathbf{r}_P' - \mathbf{r}_P) + (\nabla \phi)_{Nk} \cdot (\mathbf{r}_{Nk}' - \mathbf{r}_{Nk})\bigg)_{0} $$

# ## Again,

# ## \begin{equation}
# \frac{(M \phi) - (M \phi)_0}{\Delta t} \,  + \sum_{ip} \left( (\rho \mathbf{V}) \cdot (\mathbf{n} \, \Delta S) \right)_{ip} \phi_{ip}\, = \sum_{ip} \left((\Gamma^{\phi} \nabla \phi) \cdot (\mathbf{n} \, \Delta S) \right)_{ip} \, + S \, \Delta V
# \end{equation}

# ## Consider the first term in the RHS

# ## $$(\nabla \phi) \cdot (\mathbf{n}) = \frac{\partial \phi}{\partial \mathbf{n}}$$

# ## This is the derivative of $\phi$ along the $\mathbf{n}$ direction

# ## So,

# # \begin{equation}
# \frac{(M \phi) - (M \phi)_0}{\Delta t} \, + \sum_{ip} \left( (\rho \mathbf{V}) \cdot (\mathbf{n} \, \Delta S) \right)_{ip} \phi_{ip}\, = \sum_{ip} \left( \Gamma^{\phi} \,\frac{\partial \phi}{\partial \mathbf{n}}\, \Delta S \right)_{ip}\, + S \, \Delta V
# \end{equation}

# # $$\left(\frac{\partial \phi}{\partial n}\right)_{k} \approx \frac{\phi_{N_k} - \phi_{P}}{|\mathbf{r}_{N_k'} - \mathbf{r}_{P'}|} +  \frac{(\nabla \phi)_{N_k} \cdot (\mathbf{r}_{N_k'} - \mathbf{r}_{N_k}) - (\nabla \phi)_{P} \cdot (\mathbf{r}_{P'} - \mathbf{r}_{P})}{|\mathbf{r}_{N_k'} - \mathbf{r}_{P'}|} $$

# ### Here the first term in the RHS is treated implicitly and the second term (also known as deferred correction term) is treated explicitly using the values of the previous iteration.

# ### So,
# 
# # $$\left(\frac{\partial \phi}{\partial n}\right)_{k} \approx \frac{\phi_{N_k} - \phi_{P}}{|\mathbf{r}_{N_k'} - \mathbf{r}_{P'}|} +  \left(\frac{(\nabla \phi)_{N_k} \cdot (\mathbf{r}_{N_k'} - \mathbf{r}_{N_k}) - (\nabla \phi)_{P} \cdot (\mathbf{r}_{P'} - \mathbf{r}_{P})}{|\mathbf{r}_{N_k'} - \mathbf{r}_{P'}|} \right)_{0}$$

# ## So,

# ### \begin{equation}
# \frac{(M \phi_P) - (M \phi_P)_0}{\Delta t} \, + \sum_{ip} \left( (\rho \mathbf{V}) \cdot (\mathbf{n} \, \Delta S) \right)_{ip} \bigg(\frac{1}{2}(\phi_P + \phi_{Nk}) + \frac{1}{2}\bigg((\nabla \phi)_P \cdot (\mathbf{r}_P' - \mathbf{r}_P) + (\nabla \phi)_{Nk} \cdot (\mathbf{r}_{Nk}' - \mathbf{r}_{Nk})\bigg)_{0}\bigg)\, 
# \newline
# = \sum_{ip} \left(\Gamma^{\phi} \left(\frac{\phi_{N_k} - \phi_{P}}{|\mathbf{r}_{N_k'} - \mathbf{r}_{P'}|} +  \left(\frac{(\nabla \phi)_{N_k} \cdot (\mathbf{r}_{N_k'} - \mathbf{r}_{N_k}) - (\nabla \phi)_{P} \cdot (\mathbf{r}_{P'} - \mathbf{r}_{P})}{|\mathbf{r}_{N_k'} - \mathbf{r}_{P'}|} \right)_{0}\right)\Delta S \,\right)_{ip} + S \, \Delta V
# \end{equation}

# ## So,

# ### \begin{equation}
# \frac{(M \phi_P)}{\Delta t} \,-  \sum_{ip} \left(\Gamma^{\phi} \left(\frac{\phi_{N_k} - \phi_{P}}{|\mathbf{r}_{N_k'} - \mathbf{r}_{P'}|}\right)\Delta S \,\right)_{ip}  + \sum_{ip} \left( (\rho \mathbf{V}) \cdot (\mathbf{n} \, \Delta S) \right)_{ip} \bigg(\frac{1}{2}(\phi_P + \phi_{Nk})\bigg)
# \newline
# = \frac{(M \phi_P)_0}{\Delta t} + \left(\sum_{S} \Gamma^{\phi} \left(\left(\frac{(\nabla \phi)_{N_k} \cdot (\mathbf{r}_{N_k'} - \mathbf{r}_{N_k}) - (\nabla \phi)_{P} \cdot (\mathbf{r}_{P'} - \mathbf{r}_{P})}{|\mathbf{r}_{N_k'} - \mathbf{r}_{P'}|} \right)_{0}\right)\Delta S \,\right)
# \newline
# - \sum_{ip} \left( (\rho \mathbf{V}) \cdot (\mathbf{n} \, \Delta S) \right)_{ip} \bigg(\frac{1}{2}\bigg((\nabla \phi)_P \cdot (\mathbf{r}_P' - \mathbf{r}_P) + (\nabla \phi)_{Nk} \cdot (\mathbf{r}_{Nk}' - \mathbf{r}_{Nk})\bigg)_{0}\bigg)\,
# + S \, \Delta V
# \end{equation}

# ## So, this will give us a system of equation $A\phi = b$

# ## NOTE: The gradient terms are calculated using least square method

# ## This methodolgy is motivated by 
# ## [Maliska CR. Fundamentals of computational fluid dynamics: the finite volume method. Springer Nature; 2023 Jan 19.]
# ## [Chapter: 13, Section: 13.2]
# ## [Page: 336]    

# ****************

# # Jacobi Solver:

# *************************

# In[10]:


def Jacobi_Solver_Tol(A,b,x_0,req_Tol):
    """
    Input: 
    A: Coefficient Matrix
    b: RHS
    x_0: Initial guess for Ax = b
    req_Tol: Required Tolerance
    
    Output:
    x: Ax = b
    
    Description:
    This function solves Ax = b using Jacobi's method
    """
    
    diag_A = A.diagonal()
    temp = (A - (diag_A*np.eye(A.shape[0])))
    x = x_0
    diag_A_inv = diag_A**(-1)
    diag_A_inv = np.diag(diag_A_inv)
    Tol = req_Tol + 1
    ctr = 0
    
    while req_Tol < Tol:
        x = diag_A_inv@(b - temp@x)
        Tol = np.linalg.norm(A@x - b)
        # print(Tol)
        ctr  = ctr +  1
        
    print(f"Jacobi Iterations: {ctr}")
    
    return x


# In[11]:


def SOR_Solver_Tol(A,b,x_0,w,req_Tol):
    """
    Input: 
    A: Coefficient Matrix
    b: RHS
    x_0: Initial guess for Ax = b
    w: Relaxation factor (w < 2)
    req_Tol: Required Tolerance    
    
    Output:
    x: Ax = b
    
    Description:
    This function solves Ax = b using SOR method.
    It is found while solving that Jacobi's Method was faster.
    This is because of more efficient implementation.
    """
    
    x = x_0.copy()
    
    U = A - np.tril(A)
    
    L = A - np.triu(A)
    
    # D = A - U - L
    D = A.diagonal()
    Tol = req_Tol + 1
    ctr  = 0
    while req_Tol < Tol:
        U_x =U@x
        RHS = b - U_x
        # diag_A = A.diagonal()

        for i in range(A.shape[0]):
            x[i] = ((1-w)*x[i]) + (w*(RHS[i] - np.dot(L[i,:i],x[:i]))/D[i])
            
        Tol = np.linalg.norm(A@x - b)
        ctr = ctr + 1
        
    print(ctr)
    return x


# ***************

# # Advection Diffusion solver:

# ***************

# In[12]:


def Advec_Diff_Solver_Steady_State(scheme,rho,V,Gamma_phi,desired_Residual,Element_Element_Connectivity_new,Source_Term_diff,Source_Term_advec,Element_cen,Element_Edge_Connectivity_new,Edge_Len,Diffusion_mesh_data,Boundary_Edges):
    """
    Input:
    scheme: 0: CDS, 1: Upwind
    rho: Density
    V: Advective Coefficients: Format: np.array([u,v])
    Gamma_phi: Diffusion Coefficient
    desired_Residual: Desired residual
    Element_Element_Connectivity_new:
    Source_Term_diff: Source term due to diffusion
    Source_Term_advec: Source term due to advection
    Element_cen:
    Element_Edge_Connectivity_new:
    Edge_Len:
    Diffusion_mesh_data:
    Boundary_Edges:
    
    Output:
    phi_0: Solution
    """
    
    if (scheme != 0) and (scheme != 1):
        print(f"Choose the right scheme: 0: CDS, 1: Upwind")
        return -1
    
    Num_Triangles = Element_Element_Connectivity_new.shape[0]
    phi = np.zeros((Num_Triangles))
    phi_0 = np.zeros((Num_Triangles))
    
    res = desired_Residual + 1
    
    while res > desired_Residual:

        RHS = np.zeros(Num_Triangles)
        A = np.zeros((Num_Triangles,Num_Triangles))
        
        # LS
        # The gradient term needs to be  updated
        Element_grad_phi_LS = Grad_Phi_LS(Element_Element_Connectivity_new,Element_cen,phi_0)
        Element_grad_phi = Element_grad_phi_LS
        RHS = - (Source_Term_diff) + (Source_Term_advec)
        
        for i in range(Element_Element_Connectivity_new.shape[0]):
            Element = Element_Element_Connectivity_new[i,0]
            x = Element_cen[Element,1]
            y = Element_cen[Element,2]

            Nb_Elements = Element_Element_Connectivity_new[i,1:]
            Edges = Element_Edge_Connectivity_new[Element,1:]
            
            for j in range(Nb_Elements.shape[0]):
                nb_element = Nb_Elements[j]
                nb_edges = Element_Edge_Connectivity_new[nb_element,1:]
                common_edge = np.intersect1d(Edges,nb_edges)
                if common_edge.shape[0] == 1:
                    # Internal Edges
                    Edge = int(common_edge[0])
                    edge_len = Edge_Len[Edge,1]
                    ds = edge_len
                    dl = P_prime_Nk_prime_len(Diffusion_mesh_data,Element,Edge)
                    
                    # Debug Script:
                    # print(dl)
                    
                    A[Element,Element] = A[Element,Element] + (Gamma_phi*ds/dl)
                    
                    # Debug Script:
                    # print(nb_element)
                    
                    A[Element,nb_element] = A[Element,nb_element] + (-Gamma_phi*ds/dl) 

                    # Deferred Correction for diffusion
                    grad_phi_p = Element_grad_phi[Element,1:]
                    p_p_prime = p_p_prime_data(Diffusion_mesh_data,Element,Edge)
                    grad_phi_nb = Element_grad_phi[nb_element,1:]
                    nk_nk_prime = nk_nk_prime_data(Diffusion_mesh_data,nb_element,Edge)
                    temp = Gamma_phi*(np.dot(grad_phi_nb,nk_nk_prime) - np.dot(grad_phi_p,p_p_prime))*ds/dl
                    diff_temp = temp
                    
                    # Advection term
                    V_ip = V
                    n_ip = neg_k_p_prime_data(Diffusion_mesh_data,Element,Edge)
                    m_dot_ip = (rho*(np.dot(V_ip,n_ip))*ds)

                    # CDS
                    if scheme == 0:
                        A[Element,Element] = A[Element,Element] + (0.5*m_dot_ip)
                        A[Element,nb_element] = A[Element,nb_element] + (0.5*m_dot_ip) 
                        # Deferred Correction for advection
                        advec_temp = 0.5*(np.dot(grad_phi_p,p_p_prime) + np.dot(grad_phi_nb,nk_nk_prime))
                        advec_temp = advec_temp*m_dot_ip

                    # Upwind Scheme
                    if scheme == 1:
                        # To be consistent with Majumdar sir's note
                        F_ip = m_dot_ip
                        # This is Upwinding here
                        A[Element,Element] = A[Element,Element] + (0.5*(abs(F_ip)+F_ip))
                        A[Element,nb_element] = A[Element,nb_element] - (0.5*(abs(F_ip)-F_ip))
                        advec_temp = 0

                    RHS[Element] = RHS[Element] + (diff_temp) - (advec_temp)

                    # Debug Script:
                    # print(temp)
                    
                else:
                    # Boundary Edges
                    # So apply boundary condition here
                    
                    # NOTE:
                    # Do NOT do this, as the boundary condition imposed of the edge and not on the cells at the boundary
                    # A[Element,Element] = 1
                    # RHS[Element] = 0
                    
                    Edge = np.intersect1d(Boundary_Edges,common_edge)
                    Edge = int(Edge[0])
                    edge_len = Edge_Len[Edge,1]
                    ds = edge_len
                    dl = P_prime_Nk_prime_len(Diffusion_mesh_data,Element,Edge)
                    # print(dl)
                    # But this Nk_prime is NOT inside the computational domain
                    # This is |k_p_prime|
                    dl = dl/2
                    
                    # We know phi at k as it is the boundary edge
                    # See the Fig: 9.19 in Peric and think for 1 s
                    # So we have phi_k
                    A[Element,Element] = A[Element,Element] + (Gamma_phi*ds/dl)

                    # Deferred Correction
                    phi_k_edge = 0
                    phi_k_edge_term = (-Gamma_phi*ds/dl)*phi_k_edge 
                    grad_phi_p = Element_grad_phi[Element,1:]
                    p_p_prime = p_p_prime_data(Diffusion_mesh_data,Element,Edge)
                    temp = Gamma_phi*(- np.dot(grad_phi_p,p_p_prime))*ds/dl
                    diff_temp = temp
                    
                    # Advection term
                    # For boundaries we know phi_ip
                    # phi = 0 at the boundary
                    V_ip = V
                    n_ip = neg_k_p_prime_data(Diffusion_mesh_data,Element,Edge)
                    m_dot_ip = (rho*(np.dot(V_ip,n_ip))*ds)
                    
                    RHS[Element] = RHS[Element] + diff_temp + phi_k_edge_term

        # Debug Script:
        # This is a direct solver
        # phi = np.linalg.inv(A)@RHS

        # phi = Jacobi_Solver_Tol(A,RHS,phi,desired_Residual)

        A_mat = A.copy()
        A = scipy.sparse.csr_matrix(A)

        # preconditioner = pyamg.smoothed_aggregation_solver(A)
        # preconditioned_A = preconditioner.aspreconditioner()
        # ml = pyamg.ruge_stuben_solver(preconditioned_A,max_coarse=Num_Triangles)
        
        ml = pyamg.ruge_stuben_solver(A,max_coarse=Num_Triangles)
        # ml = pyamg.smoothed_aggregation_solver(A,max_coarse=Num_Triangles)
        # phi = ml.solve(RHS, x0=phi_0,tol=1e-20)
        phi = ml.solve(RHS,accel="gmres", x0=phi_0,tol=desired_Residual)

        # phi = Jacobi_Solver_Tol(A,RHS,phi,desired_Residual)

        res = np.linalg.norm((phi - phi_0)/(phi_0),1)

        # print(res)
        # print(np.sum((phi - phi_0) < desired_Residual) == Num_Triangles)

        # print(np.sum(np.abs((phi - phi_0)/phi_0) < desired_Residual))
        
        # res = np.linalg.norm((phi - phi_0),1)
        
        print(f"Residual: {res}")
        phi_0 = phi.copy()
        
    return A_mat,RHS,phi_0

