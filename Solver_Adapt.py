#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import subprocess


# In[2]:


from Mesh_Preprocess import *
from Source_Term import *
from LS_Solver import *
from Post_Process import *
from Advec_Diff_Solver import *
from Analytical_Solution_Contour_Plot import *
from Analytical_Sol import *
from Error_Estimate import *
from Metric_Calculate import *
from Error_Evaluate import *


# In[3]:


def Data_Writer(Num_Triangles,Mesh_File,Gamma_phi,u,v,phi_0,phi_0_u,Element_cen_phi_Actual,Element_Vertex_Avg_sol,metric_term_a_element,Element_Area, Element_Element_Connectivity_new,Element_cen,error,Source_Term_diff,Source_Term_advec):
    Area = 1
    Data = np.zeros((Num_Triangles,21))
    Data[:,0] = Element_Area[:,1]
    Data[:,1] = phi_0
    Data[:,2] = phi_0_u
    Data[:,3] = np.abs(phi_0 - phi_0_u)
    Element_grad_phi_LS = Grad_Phi_LS(Element_Element_Connectivity_new,Element_cen,phi_0)
    Data[:,4:6] = Element_grad_phi_LS[:,1:]
    Data[:,6] = (np.sum(Element_grad_phi_LS[:,1:]**2,axis=1))**0.5
    Data[:,7] = Element_cen_phi_Actual
    Data[:,8] = Element_Vertex_Avg_sol[:,1]
    Element_grad_phi_LS_Actual = Grad_Phi_LS(Element_Element_Connectivity_new,Element_cen,Element_cen_phi_Actual)
    Data[:,9:11] = Element_grad_phi_LS_Actual[:,1:]
    Data[:,11] = (np.sum(Element_grad_phi_LS_Actual[:,1:]**2,axis=1))**0.5
    Area_Uniform = Area/Num_Triangles
    Data[:,12] = Area_Uniform
    Data[:,13] = Num_Triangles
    Data[:,14] = error
    Data[:,15] = metric_term_a_element
    Data[:,16] = u
    Data[:,17] = v
    Data[:,18] = Source_Term_advec
    Data[:,19] = Source_Term_diff
    Data[:,20] = Gamma_phi
    Data_df = pd.DataFrame(Data)
    Data_File_name = f"Data_u_{u}_v_{v}_Gamma_phi_{Gamma_phi}_.csv"
    Data_df.to_csv(Data_File_name,header=None,index=None,mode="w")
    subprocess.run(f"cp Prev_{Mesh_File} Mesh_u_{u}_v_{v}_Gamma_phi_{Gamma_phi}_{Mesh_File}",shell =  True)
    return Data_File_name


# In[4]:


def Mesh_Adapt(Mesh_Adaptation_Cycles,rho,Gamma_phi,V,pts,Target_Dof,p,D,q):

    subprocess.run("cp ./Initial_Mesh/* ./",shell = True)

    for i in range(Mesh_Adaptation_Cycles):
        # ## Reading the contents of mesh file
    
        Mesh_File = "mesh.msh"
        
        # Mesh file format
        Mesh_Ascii_Format, Mesh_File_Type, Mesh_Data_Size_Native = Mesh_Format(Mesh_File)
        print(f"Mesh File Format: ")
        print(f"Mesh Ascii Format: {Mesh_Ascii_Format}")
        print(f"Mesh File Type: {Mesh_File_Type}")
        print(f"Mesh Data Size_Native: {Mesh_Data_Size_Native}")
        
        # Mesh Elements
        Mesh_Elements_Data = Mesh_Elements(Mesh_File)
        
        # This will count the number of triangles in the mesh
        Num_Triangles = np.count_nonzero(Mesh_Elements_Data[:,-1] == 2)
        
        Element_Node_Connectivity = Element_Node_Connectivity_Calculate(Num_Triangles,Mesh_Elements_Data)
        
        Edge_Node_Connectivity,Boundary_Edges = Edge_Node_Connectivity_Calculate(Element_Node_Connectivity,Mesh_Elements_Data)
        
        Element_Edge_Connectivity = Element_Edge_Connectivity_Calculate(Num_Triangles,Element_Node_Connectivity,Edge_Node_Connectivity)
        
        Node_Coordinates,Point_Nodes,Curve_Nodes,Surface_Nodes = Mesh_Nodes(Mesh_File)
        
        Num_Nodes = Node_Coordinates.shape[0]
        
        Element_Element_Connectivity = Element_Element_Connectivity_Calculate_fast(Num_Triangles,Num_Nodes,Element_Node_Connectivity)
        # Element_Element_Connectivity = Element_Element_Connectivity_Calculate(Num_Triangles,Element_Node_Connectivity)
        
        # ### We need to renumber the elements
        
        Element_Element_Connectivity_new,Element_Edge_Connectivity_new,Element_Node_Connectivity_new,Edge_Node_Connectivity_new = Renumbering(Element_Element_Connectivity,Element_Edge_Connectivity,Element_Node_Connectivity,Edge_Node_Connectivity)
        
        Face_Centroid = Face_Centroid_Calculate(Edge_Node_Connectivity_new,Node_Coordinates)
        
        # Number of vertices in each element
        Num_vertices = 3
        
        # Dimension of the problem
        dim = 2
        
        Element_Node_Connectivity[0]
        Element_Node_Coordinates_l = np.zeros((3,2))
        Element_Node_Coordinates_l[0,:] = Node_Coordinates[int(Element_Node_Connectivity[0,1])][0:2]
        Element_Node_Coordinates_l[1,:] = Node_Coordinates[int(Element_Node_Connectivity[0,2])][0:2]
        Element_Node_Coordinates_l[2,:] = Node_Coordinates[int(Element_Node_Connectivity[0,3])][0:2]
        
        Anticlock_vertices = Anti_Clock_Triangle_vertices(Element_Node_Coordinates_l)
        
        # Check_Element_all_prop(Element_Node_Connectivity_new,Node_Coordinates,Edge_Node_Connectivity_new)
        
        Edge_Element_Connectivity = Edge_Element_Connectivity_Calculate(Edge_Node_Connectivity_new,Element_Element_Connectivity_new,Element_Edge_Connectivity_new,Boundary_Edges)
        
        Diffusion_mesh_data,Element_cen = Diffusion_mesh_data_Calculate(Num_Triangles,Element_Node_Connectivity_new,Element_Edge_Connectivity_new,Node_Coordinates,Edge_Node_Connectivity_new,Edge_Element_Connectivity)
        
        Element_Area = Element_Area_Calculate(Num_Triangles,Element_Node_Connectivity_new,Node_Coordinates)
        
        Num_Edges = Edge_Node_Connectivity_new.shape[0]
        
        Edge_Len = Edge_Len_Calculate(Num_Edges,Edge_Node_Connectivity_new,Node_Coordinates)
        
        
        # ****************************************************
        
        # # Physical Variables:
        
        # *******************************************
        
        u = V[0]
        v = V[1]
        
        
        # *************************************************
        
        # # Calculating the source term:
        
        # ******************************************
        
        Element_Mass = Elements_Mass_Calculate(rho,Element_Area)
        
        Element_Source_diff = Source_Cal_diff_Elements(Num_Triangles,Element_cen,Gamma_phi,u,v)
        
        Source_Term_diff = Source_Term_Elements(rho,Element_Area,Element_Source_diff)
        
        Element_Source_advec = Source_Cal_advec_Elements(Num_Triangles,Element_cen,rho,Gamma_phi,u,v)
        
        Source_Term_advec = Source_Term_Elements(rho,Element_Area,Element_Source_advec)
        
        
        # ******************************************
        
        # # Advection Diffusion solver:
        
        # *****************************************************
        
        A,RHS,phi_0 = Advec_Diff_Solver_Steady_State(0,rho,V,Gamma_phi,1e-10,Element_Element_Connectivity_new,Source_Term_diff,Source_Term_advec,Element_cen,Element_Edge_Connectivity_new,Edge_Len,Diffusion_mesh_data,Boundary_Edges)
        
        A_u,RHS_u,phi_0_u = Advec_Diff_Solver_Steady_State(1,rho,V,Gamma_phi,1e-10,Element_Element_Connectivity_new,Source_Term_diff,Source_Term_advec,Element_cen,Element_Edge_Connectivity_new,Edge_Len,Diffusion_mesh_data,Boundary_Edges)
        
        # *******************************************
        
        # # Analytical Solution Contour Plot
        
        # **********************************************

        pts = 250
        Analytical_Solution_Contour_Plot(pts,u,v,Gamma_phi,"Analytical_Contour.svg")
        
        
        # ***********************************
        
        # # Analytical Solution
        
        # ***********************************************
        
        Anal_sol_over_Area, Num_sol_over_Area = Sol_Over_Area(u,v,Gamma_phi,phi_0,Element_Area)
        
        Element_cen_phi_Actual = Analytical_Solution(Element_cen,V,Gamma_phi)
        
        Element_Vertex_Avg_sol = Element_Vertex_Avg_Anal_Sol(Num_Triangles,Element_Node_Connectivity_new,Node_Coordinates,V,Gamma_phi)
        
        
        # *****************************************************
        
        # # Error
        
        # **************************************
        
        # Calculates the exact error using the solution at the centroid of the element
        # error = Error_Estimate_CDAS(Element_cen_phi_Actual,phi_0,Element_Area)
        
        # Calculates the exact error using the average of the solution at the vertices of the element
        # error = Error_Estimate_CDEVAAS(Element_Vertex_Avg_sol,phi_0,Element_Area)
        
        # Calculates the error estimate using: (phi_0 - phi_0_u)*(|Gradient(phi_0)|)*(Element_Area)
        error,mod_grad = Error_Estimate_EMGA(phi_0,phi_0_u,Element_Element_Connectivity_new,Element_cen,Element_Area)
        
        # Post_Process_Without_Grid(mod_grad,r"$|\nabla\Phi|$",Element_Node_Connectivity_new,Node_Coordinates)
        
        Error_Discrete = abs(error)
        
        # Post_Process(Error_Discrete,"Error Estimate",Element_Node_Connectivity_new,Node_Coordinates)
        
        # Post_Process_Without_Grid(Error_Discrete,"Error Estimate",Element_Node_Connectivity_new,Node_Coordinates)
        
        # Post_Process((Error_Discrete/Error_Discrete.mean())*Element_Area[:,1],"Mean Ratio True Error",Element_Node_Connectivity_new,Node_Coordinates)
        
        Post_Process(phi_0,r"$\Phi$",Element_Node_Connectivity_new,Node_Coordinates,"phi_0.svg")
        
        Post_Process_Without_Grid(phi_0,r"$\Phi$",Element_Node_Connectivity_new,Node_Coordinates,"phi_0_without_grid.svg")
        
        # Post_Process(Element_cen_phi_Actual,r"$\Phi: Actual$",Element_Node_Connectivity_new,Node_Coordinates)
        
        
        # *******************************
        
        # # Metric
        
        # *********************************************
        
        d = Calculate_Optimal_Mesh_Density(Error_Discrete,Element_Area,Target_Dof,p,D,q)
        
        metric_term_a_element = d
        
        # ## We need to scale the metric
        
        metric_scale = Metric_Scale_Calculate(Target_Dof,Num_Triangles,metric_term_a_element,Element_Area)
        
        metric_term_a_element = metric_term_a_element*metric_scale
        
        Node_metric = Node_Wise_Metric(metric_term_a_element,Node_Coordinates,Element_Node_Connectivity_new,Element_Area)
        
        adj_metric_file = "adj_metric_file.mtr"
        
        metric_file_writer(adj_metric_file,Node_Coordinates,Node_metric)
        
        Adapted_mesh_file_name = "bamg_adapted.mesh"
        
        Bamg_mesh_filename = "Bamg.mesh"
        
        subprocess.run(f"bamg -b {Bamg_mesh_filename} -M {adj_metric_file} -v 3 -o {Adapted_mesh_file_name} -nbv 200000",shell = True)
        
        Adapted_mesh_file_writer(Adapted_mesh_file_name)
        
        subprocess.run(f"cp {Mesh_File} Prev_{Mesh_File}",shell =  True)
        
        subprocess.run(f"cp {Adapted_mesh_file_name} ../gmsh-4.12.1-Linux64/bin/{Adapted_mesh_file_name}",shell =  True)
        
        subprocess.run(f"cd ../gmsh-4.12.1-Linux64/bin; ./gmsh {Adapted_mesh_file_name} -2 -o {Mesh_File} -save_all",shell =  True)
        
        subprocess.run(f"cp ../gmsh-4.12.1-Linux64/bin/{Mesh_File} {Mesh_File}",shell =  True)
        
        subprocess.run(f"cp {Adapted_mesh_file_name} {Bamg_mesh_filename}",shell =  True)
        
        # Plot_Edge_Number_Cell_Number(Node_Coordinates,Element_Node_Connectivity,Face_Centroid,Element_cen)
        
        Cen_Error_Data_Writer(Error_Discrete,Element_Area,Anal_sol_over_Area,Num_sol_over_Area,Num_Triangles,u,v,Gamma_phi)
        
        Vertex_Based_Error_Data_Writer(Element_Vertex_Avg_sol,phi_0,Num_Triangles,Element_Area,u,v,Gamma_phi)
        
    Data_Writer(Num_Triangles,Mesh_File,Gamma_phi,u,v,phi_0,phi_0_u,Element_cen_phi_Actual,Element_Vertex_Avg_sol,metric_term_a_element,Element_Area, Element_Element_Connectivity_new,Element_cen,error,Source_Term_diff,Source_Term_advec)

