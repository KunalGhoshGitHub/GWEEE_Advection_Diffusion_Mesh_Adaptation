#!/usr/bin/env python
# coding: utf-8

# # Importing necessary libraries:

# ******************************************************

# In[1]:


import numpy as np


# In[2]:


import pandas as pd


# In[3]:


import matplotlib.pyplot as plt


# In[4]:


import matplotlib


# In[5]:


from matplotlib.colors import Normalize


# In[6]:


import subprocess


# In[9]:


import scipy


# In[10]:


from numba import jit


# ******************************************

# # Preprocessing the mesh file:

# ****************************************************

# ## We are using the Gmsh to mesh the geometry:
# ### Gmsh website: https://gmsh.info/
# ### Reference to Gmsh file formats: https://gmsh.info/doc/texinfo/gmsh.html#Gmsh-file-formats

# ### NOTE: Enable save all nodes option while saving the mesh file

# ## Reading the contents of mesh file

# In[12]:


# Mesh_File should be the name of the Gmsh file which is to be used
Mesh_File = "mesh.msh"


# In[13]:


def Mesh_String_Extractor(mesh_file_content,Start,End):
    """
    Input: 
    mesh_file_content: Content of the mesh file as string
    Start: Start of the string to be extracted, excluding Start
    End: End of the string to be extracted, excluding End
    
    Output:
    Mesh_String: Content of the mesh file between Start and End
    
    Description:
    Extracting the string between Start and End in the string mesh_file_content
    """
    
    # Acquiring the index of Start
    Mesh_Start_index = mesh_file_content.find(Start)
    
    # Acquiring the index of End
    Mesh_End_index = mesh_file_content.find(End)
    
    # Acquiring the content between Start and End
    Mesh_String = mesh_file_content[Mesh_Start_index + len(Start)+1:Mesh_End_index-1]
    
    return Mesh_String


# In[14]:


def Mesh_Format(Mesh_File):
    """
    Input: 
    mesh_file: Path to the mesh file
    
    Output:
    Mesh_Ascii_Format: Ascii Format of the mesh file
    Mesh_File_Type: 0 for ASCII mode, 1 for binary mode
    Mesh_Data_Size_Native: Data-size in the mesh_file
    
    Description:
    Extracting the mesh file format
    """
    
    # Reading the mesh file
    mesh_file = open(Mesh_File,"r")
    
    # Creating the string of the entire mesh file
    mesh_file_content = mesh_file.read()
    
    # Closing the mesh file
    mesh_file.close()
    
    # Extracting the mesh format
    Start = "$MeshFormat"
    End = "$EndMeshFormat"
    Mesh_Format_string = Mesh_String_Extractor(mesh_file_content,Start,End)
    
    # Mesh_Format_string: Ascii Format of the mesh file
    Mesh_Ascii_Format = float(Mesh_Format_string.split(sep=" ")[0])
    
    # Mesh_File_Type: 0 for ASCII mode, 1 for binary mode
    Mesh_File_Type = int(Mesh_Format_string.split(sep=" ")[1])
    
    # Mesh_Data_Size_Native: Data-size in the mesh_file
    Mesh_Data_Size_Native = int(Mesh_Format_string.split(sep=" ")[2])
    
    return Mesh_Ascii_Format, Mesh_File_Type, Mesh_Data_Size_Native


# In[15]:


# Mesh file format
# Mesh_Ascii_Format, Mesh_File_Type, Mesh_Data_Size_Native = Mesh_Format(Mesh_File)
# print(f"Mesh File Format: ")
# print(f"Mesh Ascii Format: {Mesh_Ascii_Format}")
# print(f"Mesh File Type: {Mesh_File_Type}")
# print(f"Mesh Data Size_Native: {Mesh_Data_Size_Native}")


# In[16]:


def Mesh_Physical_Names(Mesh_File):
    """
    Input: 
    Mesh_File: Path to the mesh file
    
    Output:
    Physical_Tags_Dimension_Array[Physical_Tag,Dimension]
    Physical_Tags_Name[Physical_Tag_Name]
    
    Description:
    Extracting the physical names in the mesh
    
    NOTE:
    This function will be used in future
    """
    
    # Reading the mesh file
    mesh_file = open(Mesh_File,"r")
    
    # Creating the string of the entire mesh file
    mesh_file_content = mesh_file.read()
    
    # Closing the mesh file
    mesh_file.close()
    
    # Extracting the physical names
    Start = "$PhysicalNames"
    End = "$EndPhysicalNames"
    Mesh_String = Mesh_String_Extractor(mesh_file_content,Start,End)
    
    # Extracting the lines
    Lines = Mesh_String.split(sep="\n")
    
    # Extracting the number of the lines
    num_Physical_Names = int(Lines[0])
    
    # Memory Allocation
    # Physical_Tags_Dimension_Array: 
    # Physical_Tags_Dimension_Array[Physical_Tag,Dimension]
    Physical_Tags_Dimension_Array = np.zeros((num_Physical_Names,2),dtype=int)
    # Physical_Tags_Name[Physical_Tag_Name]
    Physical_Tags_Name = np.empty(num_Physical_Names,dtype = object)
    
    # Populating the arrays
    for i in range(1,num_Physical_Names+1):
        
        # Debugging Script:
        # print(i)
        # dimension = int(Lines[i].split(" ")[0])
        # Physical_Tag = int(Lines[i].split(" ")[1])
        # Physical_Tag_Name = Lines[i].split(" ")[2][1:-1]
        # print(Physical_Tag_Name)
        
        Physical_Tags_Dimension_Array[i-1,0] = int(Lines[i].split(" ")[0])
        Physical_Tags_Dimension_Array[i-1,1] = int(Lines[i].split(" ")[1])
        Physical_Tags_Name[i-1] = Lines[i].split(" ")[2][1:-1]
    
    return Physical_Tags_Dimension_Array, Physical_Tags_Name


# In[17]:


# Physical names in the mesh file
# Mesh_Physical_Names(Mesh_File)


# In[18]:


def Mesh_Entities(Mesh_File):
    """
    Input: 
    Mesh_File: Path to the mesh file
    
    Output:
    Point_Data: [pointTag(int),X(double),Y(double),Z(double),numPhysicalTags(size_t),physicalTag(int)]
    Curve_Data: [curveTag(int),minX(double),minY(double),minZ(double),maxX(double),maxY(double),maxZ(double),numPhysicalTags(size_t),physicalTag(int),numBoundingPoints(size_t),pointTag]
    Surface_Data: [surfaceTag(int),minX(double),minY(double),minZ(double),maxX(double),maxY(double),maxZ(double),numPhysicalTags(size_t),physicalTag(int),numBoundingCurves(size_t),curveTag]
    Volume_Data: [volumeTag(int),minX(double),minY(double),minZ(double),maxX(double),maxY(double),maxZ(double),numPhysicalTags(size_t),physicalTag(int),numBoundngSurfaces(size_t),surfaceTag]
    
    Description:
    Extracting the data about different types of entities in the mesh 
    """
    
    # Reading the mesh file
    mesh_file = open(Mesh_File,"r")
    
    # Creating the string of the entire mesh file
    mesh_file_content = mesh_file.read()
    
    # Closing the mesh file
    mesh_file.close()
    
    # Extracting the Entities
    Start = "$Entities"
    End = "$EndEntities"
    Mesh_String = Mesh_String_Extractor(mesh_file_content,Start,End)
    
    # Extracting the lines
    Lines = Mesh_String.split(sep="\n")
    
    num_Mesh_Points = int(Lines[0].split(" ")[0])
    num_Mesh_Curves = int(Lines[0].split(" ")[1])
    num_Mesh_Surfaces = int(Lines[0].split(" ")[2])
    num_Mesh_Volumes = int(Lines[0].split(" ")[3])
    
    # Memory allocation
    Point_Attributes = len(Lines[1:num_Mesh_Points+1][0].split(" "))
    Point_Data = np.zeros((num_Mesh_Points,Point_Attributes))
    
    # Populating the Points_Data
    for i in range(1,num_Mesh_Points+1):
    # Make this True if you are using Physical Tags for the points
        Point_Pysical_Tags = False
        if Point_Pysical_Tags == False:
            # print(Lines[1:num_Mesh_Points+1][i-1])
            Point_Data[i-1][:Point_Attributes-1] = Lines[1:num_Mesh_Points+1][i-1].split(" ")[:Point_Attributes-1]
        else:
            Point_Data[i-1] = Lines[1:num_Mesh_Points+1][i-1].split(" ")
    
    # Memory allocation
    Curve_Attributes = len(Lines[num_Mesh_Points+1:num_Mesh_Points+num_Mesh_Curves+1][0].split(" "))
    Curve_Data = np.zeros((num_Mesh_Curves,Curve_Attributes))

    # Populating the Curve_Data
    for i in range(1,num_Mesh_Curves+1):
        # Make this True if you are using Physical Tags for the points
        Point_Pysical_Tags = False
        if Point_Pysical_Tags == False:
            Curve_Data[i-1][:Curve_Attributes-1] = Lines[num_Mesh_Points+1:num_Mesh_Points+num_Mesh_Curves+1][i-1].split(" ")[:Curve_Attributes-1]
        else:
            Curve_Data[i-1] = Lines[num_Mesh_Points+1:num_Mesh_Points+num_Mesh_Curves+1][i-1].split(" ")
    
    # Memory allocation
    Surface_Attributes = len(Lines[num_Mesh_Points+num_Mesh_Curves+1:num_Mesh_Points+num_Mesh_Curves+num_Mesh_Surfaces+1][0].split(" "))
    Surface_Data = np.zeros((num_Mesh_Surfaces,Surface_Attributes))

    # Populating the Surface_Data
    for i in range(1,num_Mesh_Surfaces+1):
        # Make this True if you are using Physical Tags for the points
        Point_Pysical_Tags = False
        if Point_Pysical_Tags == False:
            Surface_Data[i-1][:Surface_Attributes-1] = Lines[num_Mesh_Points+num_Mesh_Curves+1:num_Mesh_Points+num_Mesh_Curves+num_Mesh_Surfaces+1][i-1].split(" ")[:Surface_Attributes-1]
        else:
            Surface_Data[i-1] = Lines[num_Mesh_Points+num_Mesh_Curves+1:num_Mesh_Points+num_Mesh_Curves+num_Mesh_Surfaces+1][i-1].split(" ")

    if num_Mesh_Volumes == 0:
        Volume_Data = 0
    else:
        # Memory Allocation
        Volume_Attributes = len(Lines[num_Mesh_Points+num_Mesh_Curves+num_Mesh_Surfaces+1:num_Mesh_Points+num_Mesh_Curves+num_Mesh_Surfaces+num_Mesh_Volumes+1][0].split(" "))
        Volume_Data = np.zeros((num_Mesh_Volumes,Volume_Attributes))

        # Populating the Curve_Data
        for i in range(1,num_Mesh_Volumes+1):
            # Make this True if you are using Physical Tags for the points
            Point_Pysical_Tags = False
            if Point_Pysical_Tags == False:
                Volume_Data[i-1][:Volume_Attributes-1] = Lines[num_Mesh_Points+num_Mesh_Curves+num_Mesh_Surfaces+1:num_Mesh_Points+num_Mesh_Curves+num_Mesh_Surfaces+num_Mesh_Volumes+1][i-1].split(" ")[:Volume_Attributes-1]
            else:
                Volume_Data[i-1] = Lines[num_Mesh_Points+num_Mesh_Curves+num_Mesh_Surfaces+1:num_Mesh_Points+num_Mesh_Curves+num_Mesh_Surfaces+num_Mesh_Volumes+1][i-1].split(" ")

    return Point_Data,Curve_Data,Surface_Data,Volume_Data


# In[19]:


def Mesh_Nodes(Mesh_File):
    """
    Input: 
    Mesh_File: Path to the mesh file
    
    Output:
    Point_Data: [pointTag(int),X(double),Y(double),Z(double),numPhysicalTags(size_t),physicalTag(int)]
    Curve_Data: [curveTag(int),minX(double),minY(double),minZ(double),maxX(double),maxY(double),maxZ(double),numPhysicalTags(size_t),physicalTag(int),numBoundingPoints(size_t),pointTag]
    Surface_Data: [surfaceTag(int),minX(double),minY(double),minZ(double),maxX(double),maxY(double),maxZ(double),numPhysicalTags(size_t),physicalTag(int),numBoundingCurves(size_t),curveTag]
    Volume_Data: [volumeTag(int),minX(double),minY(double),minZ(double),maxX(double),maxY(double),maxZ(double),numPhysicalTags(size_t),physicalTag(int),numBoundngSurfaces(size_t),surfaceTag]
    
    Description:
    Extracting different types of mesh nodes in the mesh
    """
    
    # Reading the mesh file
    mesh_file = open(Mesh_File,"r")
    
    # Creating the string of the entire mesh file
    mesh_file_content = mesh_file.read()
    
    # Closing the mesh file
    mesh_file.close()
    
    # Extracting the Entities
    Start = "$Entities"
    End = "$EndEntities"
    Mesh_String = Mesh_String_Extractor(mesh_file_content,Start,End)
    
    # Extracting the lines
    Lines = Mesh_String.split(sep="\n")
    
    num_Mesh_Points = int(Lines[0].split(" ")[0])
    num_Mesh_Curves = int(Lines[0].split(" ")[1])
    num_Mesh_Surfaces = int(Lines[0].split(" ")[2])
    num_Mesh_Volumes = int(Lines[0].split(" ")[3])
    
    # Extracting the Nodes
    Start = "$Nodes"
    End = "$EndNodes"
    Mesh_String = Mesh_String_Extractor(mesh_file_content,Start,End)
    
    # Extracting the lines
    Lines = Mesh_String.split(sep="\n")
    
    num_Entity_Blocks = int(Lines[0].split(" ")[0])
    num_Nodes = int(Lines[0].split(" ")[1])
    min_Node_Tag = int(Lines[0].split(" ")[2])
    max_Node_Tag = int(Lines[0].split(" ")[3])
    
    # Memory allocation
    Node_Coordinates = np.zeros((num_Nodes,3))
    
    # Storing the coordinates of all the nodes
    # The index of the array is the (Node_Tag - 1)
    for i in range(1,len(Lines[1:]) - 1):
        # Description Line
        # entityDim(int),entityTag(int),parametric(int; 0 or 1),numNodesInBlock(size_t),nodeTag(size_t)
        if (len(Lines[i].split()) == 4) and (len(Lines[i+1].split()) != 4):
            # print(Lines[i].split(" "))
            temp = int(Lines[i].split(" ")[-1])
            for j in range(i+1,temp+i+1):
                Node_Tag = int(Lines[j])
                # print(Lines[temp + j].split(" "))
                Node_Coordinates[Node_Tag-1][:] = Lines[temp + j].split(" ")[:3]
            i = i+(2*temp)
    
    # Memory allocation
    Point_Nodes = np.zeros((num_Mesh_Points,1))
    Curve_Nodes = np.empty(num_Mesh_Curves,dtype = object)
    Surface_Nodes = np.empty(num_Mesh_Surfaces,dtype = object)
    
    # Storing the nodes in the different entities (Points, Curves and Surfaces)
    # The index of the arrays are the (Tag - 1) and the list output is the list of the node in that entity
    for i in range(1,len(Lines[1:]) - 1):
        # Description Line
        # entityDim(int),entityTag(int),parametric(int; 0 or 1),numNodesInBlock(size_t),nodeTag(size_t)
        if (len(Lines[i].split()) == 4) and (len(Lines[i+1].split()) != 4):
            # print(Lines[i].split(" "))
            temp = int(Lines[i].split(" ")[0])
            if temp == 0:
                temp_1 = int(Lines[i].split(" ")[1])
                Point_Nodes[temp_1 - 1] = int(Lines[i+1].split(" ")[0]) - 1
            if temp == 1:
                temp_2 = int(Lines[i].split(" ")[-1])
                Curve_List = []
                for j in range(0,temp_2):
                    Curve_List.append(int(Lines[i+j+1].split(" ")[0])-1)
                temp_1 = int(Lines[i].split(" ")[1])
                Curve_Nodes[temp_1 - 1] = Curve_List
            if temp == 2:
                temp_2 = int(Lines[i].split(" ")[-1])
                Surface_List = []
                for j in range(0,temp_2):
                    Surface_List.append(int(Lines[i+j+1].split(" ")[0])-1)
                temp_1 = int(Lines[i].split(" ")[1])
                Surface_Nodes[temp_1 - 1] = Surface_List
            i = i+(2*temp)
    
    return Node_Coordinates,Point_Nodes,Curve_Nodes,Surface_Nodes


# In[20]:


def Mesh_Elements(Mesh_File):
    """
    Input: 
    Mesh_File: Path to the mesh file
    
    Output:
    Element_Nodes: [X,Y,Z,Element Type]
    Index is the (Element_Tag -1)
    
    Description:
    This will contain different types of elements in 0D, 1D, 2D and 3D
    """
    
    # Reading the mesh file
    mesh_file = open(Mesh_File,"r")

    # Creating the string of the entire mesh file
    mesh_file_content = mesh_file.read()

    # Closing the mesh file
    mesh_file.close()

    # Extracting the Elements
    Start = "$Elements"
    End = "$EndElements"
    Mesh_String = Mesh_String_Extractor(mesh_file_content,Start,End)

    # Extracting the lines
    Lines = Mesh_String.split(sep="\n")

    num_Entity_Blocks = int(Lines[0].split(" ")[0])
    num_Elements = int(Lines[0].split(" ")[1])
    min_Element_Tag = int(Lines[0].split(" ")[2])
    max_Element_Tag = int(Lines[0].split(" ")[3])
    
    # Memory allocation
    Element_Nodes = np.zeros((num_Elements,4),dtype = int)

    for i in range(1,len(Lines[1:])):
        # print(Lines[i].split())
        if len(Lines[i].split()) == 4:
            if int(Lines[i].split()[0]) <= 3:
                # print(Lines[i])
                for j in range(i+1,i+1+int(Lines[i].split()[-1])):
                    # print(Lines[j])
                    index = int(Lines[j].split()[0])
                    if len(Lines[j].split()) == 2:
                        Element_Nodes[index-1,-1] = Lines[i].split()[0]
                        Element_Nodes[index-1,0] = Lines[j].split()[1]
                        Element_Nodes[index-1,0] = Element_Nodes[index-1,0] - 1
                    if len(Lines[j].split()) == 3:
                        Element_Nodes[index-1,-1] = Lines[i].split()[0]
                        Element_Nodes[index-1,0:2] = Lines[j].split()[1:]
                        Element_Nodes[index-1,0:2] = Element_Nodes[index-1,0:2] - 1
                    if len(Lines[j].split()) == 4:
                        Element_Nodes[index-1,-1] = Lines[i].split()[0]
                        Element_Nodes[index-1,0:3] = Lines[j].split()[1:]
                        Element_Nodes[index-1,0:3] = Element_Nodes[index-1,0:3] - 1
                        
    return Element_Nodes


# In[21]:


# Mesh Elements
Mesh_Elements_Data = Mesh_Elements(Mesh_File)


# In[22]:


# This will count the number of triangles in the mesh
Num_Triangles = np.count_nonzero(Mesh_Elements_Data[:,-1] == 2)


# In[23]:


def Element_Node_Connectivity_Calculate(Num_Triangles,Mesh_Elements_Data):
    """
    Input:
    Num_Triangles: Number of triangles
    Mesh_Elements_Data: Mesh Elements Data (Previously Extracted)
    
    Output:
    Element Node Connectivity: Format: [Element, Node1, Node2, Node3]
    """
    
    # Memory allocation
    # Format: [Element, Node1, Node2, Node3]
    Element_Node_Connectivity = np.zeros((Num_Triangles,4))
    
    #  This a counter
    j = 0
    
    # Looping over all type of mesh elements in the mesh
    for i in range(Mesh_Elements_Data.shape[0]):
        
        # Only triangle mesh elements will be considered
        if Mesh_Elements_Data[i,-1] == 2:
            # Element is stored
            Element_Node_Connectivity[j,0] = i
            # Nodes are stored
            Element_Node_Connectivity[j,1:] = Mesh_Elements_Data[i,:3]
            j = j+1
    
    return Element_Node_Connectivity


# In[24]:


Element_Node_Connectivity = Element_Node_Connectivity_Calculate(Num_Triangles,Mesh_Elements_Data)


# In[25]:


@jit(nopython=True)
def count_matching_elements(arr1,arr2):
    count = 0
    for elm in arr1:
        if elm in arr2:
            count = count + 1
    return count


# In[26]:


@jit(nopython=True)
def intersect1d_numba(arr1, arr2):
    # Sort the input arrays
    arr1_sorted = np.sort(arr1)
    arr2_sorted = np.sort(arr2)
    
    # Initialize variables to track unique elements and intersection
    intersection = []
    i, j = 0, 0
    
    # Find the intersection of the sorted arrays
    while i < len(arr1_sorted) and j < len(arr2_sorted):
        if arr1_sorted[i] < arr2_sorted[j]:
            i += 1
        elif arr1_sorted[i] > arr2_sorted[j]:
            j += 1
        else:
            intersection.append(arr1_sorted[i])
            i += 1
            j += 1
            
    return np.array(intersection)


# In[27]:


def Edge_Node_Connectivity_Calculate(Element_Node_Connectivity,Mesh_Elements_Data):
    """
    Input:
    Element_Node_Connectivity: Element Node Connectivity (Previously Extracted)
    Mesh_Elements_Data: Mesh Elements Data (Previously Extracted)
    
    Output:
    Edge_Node_Connectivity: Format: [Edge, Node1, Node2]
    Boundary_Edges: [Edge]
    
    Description:
    This function calculates edge node connectivity
    
    NOTE:
    This function also returns boundary edges
    """
    
    # List to store edge node connectivity
    Edge_Node_Connectivity = []
    
    # List to store boundary edges
    Boundary_Edges = []
    
    # Edges inside the domain
    Edge_num = 0
    for i in range(Element_Node_Connectivity.shape[0]):
        Element = Element_Node_Connectivity[i][0]
        Nodes = Element_Node_Connectivity[i][1:]
        for j in range(i+1,Element_Node_Connectivity.shape[0]):
            element = Element_Node_Connectivity[j][0]
            nodes = Element_Node_Connectivity[j][1:]
            
            # Ordinary
            # value = np.isin(Nodes,nodes).sum()
            
            # Numba 
            value = count_matching_elements(Nodes,nodes)
            
            temp = []
            if value == 2:
                if Element != element:

                    #Ordinary
                    temp_node = np.intersect1d(Nodes,nodes)
                    
                    # Numba
                    # temp_node = intersect1d_numba(Nodes,nodes)
                    
                    temp_node = np.sort(temp_node)
                    
                    # Debugging Script:
                    # print(Nodes)
                    # print(nodes)
                    # print(element)
                    # print(Element)
                    
                    temp.append(Edge_num)
                    temp.append(temp_node[0])
                    temp.append(temp_node[1])
                    Edge_Node_Connectivity.append(temp)
                    Edge_num = Edge_num + 1

    # Edges at the boundary
    for i in range(Mesh_Elements_Data.shape[0]):
        if Mesh_Elements_Data[i,-1] == 1:
            temp = []
            temp.append(Edge_num)
            temp_node = Mesh_Elements_Data[i][0:2]
            temp_node = np.sort(temp_node)        
            temp.append(temp_node[0])
            temp.append(temp_node[1])
            Edge_Node_Connectivity.append(temp)
            Boundary_Edges.append(Edge_num)
            Edge_num = Edge_num + 1
            
            # Debugging Script:
            # print(temp)
            # print(np.isin(Nodes,nodes).sum())
            
    Edge_Node_Connectivity = np.array(Edge_Node_Connectivity)
    Boundary_Edges = np.array(Boundary_Edges)
    return Edge_Node_Connectivity,Boundary_Edges


# In[28]:


Edge_Node_Connectivity,Boundary_Edges = Edge_Node_Connectivity_Calculate(Element_Node_Connectivity,Mesh_Elements_Data)


# In[29]:


@jit(nopython=True)
def count_matching_elements(arr1,arr2):
    count = 0
    for elm in arr1:
        if elm in arr2:
            count = count + 1
    return count


# In[30]:


@jit(nopython = True)
def Element_Edge_Connectivity_Calculate(Num_Triangles,Element_Node_Connectivity,Edge_Node_Connectivity):
    """
    Input: 
    Num_Triangles: Number of triangles
    Element_Node_Connectivity: Element Node Connectivity (Previously Calculated)
    Edge_Node_Connectivity: Edge Node Connectivity (Previously Calculated)
    
    Output:
    Element_Edge_Connectivity: Element Edge Connectivity
    """
    # Memory allocation
    Element_Edge_Connectivity = np.zeros((Num_Triangles,4))

    for i in range(Element_Node_Connectivity.shape[0]):
        Element = Element_Node_Connectivity[i][0]
        Nodes = Element_Node_Connectivity[i][1:]
        Element_Edge_Connectivity[i,0] = Element
        # print(Element)
        ctr = 1
        for j in range(Edge_Node_Connectivity.shape[0]):
            edge = Edge_Node_Connectivity[j][0]
            # node_1 = Edge_Node_Connectivity[j][1]
            # node_2 = Edge_Node_Connectivity[j][2]
            nodes = Edge_Node_Connectivity[j][1:]

            # Ordinary
            # value = np.isin(Nodes,nodes).sum()
            
            # Numba
            value = count_matching_elements(Nodes,nodes)
            # temp = []
            if value == 2:
                Element_Edge_Connectivity[i,ctr] = edge
                ctr = ctr + 1
                if ctr == 4:
                    i = i + 1
                    j = Edge_Node_Connectivity.shape[0]
                
    return Element_Edge_Connectivity


# In[31]:


Element_Edge_Connectivity = Element_Edge_Connectivity_Calculate(Num_Triangles,Element_Node_Connectivity,Edge_Node_Connectivity)


# In[32]:


Node_Coordinates,Point_Nodes,Curve_Nodes,Surface_Nodes = Mesh_Nodes(Mesh_File)


# In[33]:


def Plot_Nodes(Node_Coordinates,Marker):
    """
    Input: 
    Node_Coordinates: Node Coordinates (Previously Calculated)
    Marker: Marker for each of the nodes
    
    Output:
    Plots the nodes of the elements
    """
    x = Node_Coordinates[:,0]
    y = Node_Coordinates[:,1]
    plt.plot(x,y,Marker)
    plt.axis('scaled')
    plt.show()


# Plot_Nodes(Node_Coordinates,"g*")

# In[34]:


def Plot_Edges(Element_Node_Connectivity,Node_Coordinates):
    """
    Input: 
    Element_Node_Connectivity: Element Node Connectivity (Previously Calculated)
    Node_Coordinates: Node Coordinates (Previously Calculated)
    
    Output:
    Plots the edges in the domain
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
    plt.axis('scaled')
    plt.show()


# Plot_Edges(Element_Node_Connectivity,Node_Coordinates)

# In[35]:


def Element_Element_Connectivity_Calculate(Num_Triangles,Element_Node_Connectivity):
    """
    Input:
    Num_Triangles: Number of Triangles
    Element_Node_Connectivity: Element Node Connectivity (Previously Calculated)
    
    Output:
    Element_Element_Connectivity: Format [Element, Nb_Element_1, Nb_Element_2, Nb_Element_3]
    """
    
    Element_Element_Connectivity = np.zeros((Num_Triangles,4),dtype = int)

    Edge_num = 0
    for i in range(Element_Node_Connectivity.shape[0]):
        # print(i)
        Element = Element_Node_Connectivity[i][0]
        # This is where all the entries are made equal to the original element
        Element_Element_Connectivity[i] = Element
        Nodes = Element_Node_Connectivity[i][1:]
        ctr = 1
        for j in range(Element_Node_Connectivity.shape[0]):
            element = Element_Node_Connectivity[j][0]
            nodes = Element_Node_Connectivity[j][1:]
            value = np.isin(Nodes,nodes).sum()
            if value == 2:
                if Element != element:
                    # print(Element)
                    # print(element)
                    Element_Element_Connectivity[i][ctr] =  element
                    ctr = ctr + 1
                    if ctr == 4:
                        # print(ctr)
                        i = i + 1
                        j = Element_Node_Connectivity.shape[0]
        # print(i)
        # Debug Script
        # print(ctr)
        # if ctr == 3:
            # print(Element_Element_Connectivity[i][:])
            
    return Element_Element_Connectivity


# In[36]:


Node_Coordinates.shape[0]


# In[37]:


Element_Node_Connectivity


# In[38]:


Element_Node_Connectivity[:,0].min()


# In[39]:


# Fast
def Element_Element_Connectivity_Calculate_fast(Num_Triangles,Num_Nodes,Element_Node_Connectivity):
    Element_Elements_Array = np.zeros((Num_Triangles,4),dtype = int)
    # Node loop
    min_element = int(Element_Node_Connectivity[:,0].min())
    for i in range(Num_Nodes):
        Node_Elements = []
        for k in range(1,4,1):
            for j in range(Element_Node_Connectivity[Element_Node_Connectivity[:,k] == i].shape[0]):
                # print(Mesh_Element_Points_dict[key][Mesh_Element_Points_dict[key][:,k] == i][j])
                Node_Elements.append(Element_Node_Connectivity[Element_Node_Connectivity[:,k] == i][j])
        Node_Elements = np.array(Node_Elements)
        Num_Elements_in_Node = Node_Elements.shape[0]
        # print(Num_Elements_in_Node)
        # print(Node_Elements)
        # Cell loop
        for j in range(Num_Elements_in_Node):
            Element = int(Node_Elements[j,0])
            Points = Node_Elements[j,1:]
            # print(Element)
            Element_Elements_Array[Element-min_element,0] = Element
            for k in range(Num_Elements_in_Node):
                element = Node_Elements[k,0]
                points = Node_Elements[k,1:]
                # Avoid the cell in consideration
                if element != Element:
                    value = np.isin(Points,points).sum()
                    # Share 2 points
                    if value == 2:
                        temp_array = np.where(Element_Elements_Array[Element -min_element][1:] == 0)[0]
                        # Still entries are empty
                        if temp_array.shape[0] != 0:
                            # No duplicate entries
                            if np.isin(Element_Elements_Array[Element-min_element,1:],element).sum() == 0:
                                zero_index = temp_array[0]
                                Element_Elements_Array[Element-min_element,zero_index+1] = element    
                                
                            # Debug
                            # print(f"{Element}\t{element}")
                            # print(zero_index)
                            # print(Element_Elements_Array[Element-1,:])
    # Element_Elements_Array[Element_Elements_Array == 0] = -1
    # Element_Elements_Array = Element_Elements_Array - 1
    
    for i in range(1,4,1):
        Element_Elements_Array[Element_Elements_Array[:,i] == 0,i] = Element_Elements_Array[Element_Elements_Array[:,i] == 0,0]
    return Element_Elements_Array


# In[40]:


Num_Nodes = Node_Coordinates.shape[0]


# In[41]:


Num_Triangles


# In[42]:


Element_Element_Connectivity = Element_Element_Connectivity_Calculate_fast(Num_Triangles,Num_Nodes,Element_Node_Connectivity)


# Element_Element_Connectivity = Element_Element_Connectivity_Calculate(Num_Triangles,Element_Node_Connectivity)


# ### We need to renumber the elements

# In[43]:


def Renumbering(Element_Element_Connectivity,Element_Edge_Connectivity,Element_Node_Connectivity,Edge_Node_Connectivity):
    # Renumbering
    Element_Element_Connectivity_new = Element_Element_Connectivity - np.min(Element_Element_Connectivity)

    # Renumbering
    Element_Edge_Connectivity_new = Element_Edge_Connectivity
    Element_Edge_Connectivity_new[:,0] = Element_Edge_Connectivity_new[:,0] - np.min(Element_Edge_Connectivity_new[:,0])

    # Renumbering
    Element_Node_Connectivity_new = Element_Node_Connectivity

    # Renumbering
    Element_Node_Connectivity_new[:,0] = Element_Node_Connectivity_new[:,0] - np.min(Element_Node_Connectivity_new[:,0])

    # Renumbering
    Edge_Node_Connectivity_new = Edge_Node_Connectivity 
    
    return Element_Element_Connectivity_new,Element_Edge_Connectivity_new,Element_Node_Connectivity_new,Edge_Node_Connectivity_new


# In[44]:


Element_Element_Connectivity_new,Element_Edge_Connectivity_new,Element_Node_Connectivity_new,Edge_Node_Connectivity_new = Renumbering(Element_Element_Connectivity,Element_Edge_Connectivity,Element_Node_Connectivity,Edge_Node_Connectivity)


# In[45]:


def Face_Centroid_Calculate(Edge_Node_Connectivity_new,Node_Coordinates):
    """
    Input: 
    Edge_Node_Connectivity_new: Edge Node Connectivity renumbered
    Node_Coordinates: Node Coordinates (Previously Calculated)
    
    Output:
    Face_Centroid: Format: [Edge,centroid x, centroid y]
    """
    
    # Memory allocation
    Face_Centroid = np.zeros((Edge_Node_Connectivity_new.shape[0],3))
    
    # Converting the 3D coordinates to 2D coordinates
    Node_Coordinates_2D = Node_Coordinates[:,:2]
    
    for i in range(Edge_Node_Connectivity_new.shape[0]):
        Face_Centroid[i,0] = Edge_Node_Connectivity_new[i,0]
        Nodes = Edge_Node_Connectivity_new[i,1:]
        # print(Nodes)
        temp = 0
        for node in Nodes:
            temp = temp + Node_Coordinates_2D[int(node)]
        temp = temp*0.5
        Face_Centroid[i,1:] = temp
        
    return Face_Centroid


# In[46]:


Face_Centroid = Face_Centroid_Calculate(Edge_Node_Connectivity_new,Node_Coordinates)


# In[47]:


def Plot_Mesh_with_Face_Centroid(Node_Coordinates,Element_Node_Connectivity,Face_Centroid,Marker,Markersize):
    """
    Input:
    Node_Coordinates: Node Coordinates (Previously Calculated)
    Element_Node_Connectivity: Element Node Connectivity (Previously Calculated)
    Face_Centroid: Face Centroid (Previously Calculated)
    Marker: Marker for the face centroid
    Markersize: Marker size of the face centroid
    
    Output:
    This function plots mesh with face centroid
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
    plt.plot(Face_Centroid[:,1],Face_Centroid[:,2],Marker,markersize=Markersize)
    plt.axis('scaled')
    plt.show()
    


# Plot_Mesh_with_Face_Centroid(Node_Coordinates,Element_Node_Connectivity,Face_Centroid,"gd",3.5)

# In[48]:


# Number of vertices in each element
Num_vertices = 3


# In[49]:


# Dimension of the problem
dim = 2


# In[50]:


def Anti_Clock_Triangle_vertices(vertices):
    """
    Input:
    vertices: These are the coordinates of the vertices of an element
    
    Output:
    new_vertices: These are the coordinates of the vertices of the same element arranged in anticlock-wise fashion
    
    NOTE:
    This is requied because while using Gauss divergence theorem we need to take the integral over the surface in the outward direction
    """
    
    new_vertices = np.zeros(vertices.shape)
    Num_vertices = vertices.shape[0]
    dim = 2
    Initial_vertex = vertices[0,:]
    vec_1 = Initial_vertex - vertices[1,:]
    vec_2 = Initial_vertex - vertices[2,:]
    sign = np.cross(vec_1,vec_2)
    if sign < 0:
        new_vertices[0,:] = Initial_vertex
        temp = vertices[1,:]
        new_vertices[1,:] = vertices[2,:]
        new_vertices[2,:] = temp
    else:
        new_vertices = vertices
    return new_vertices


# In[51]:


@jit(nopython=True)
def Calculate_Face_Centroid(Anticlock_vertices):
    """
    Input: 
    Anticlock_vertices: Vertices of an element in the anticlock-wise direction
    
    Output:
    Centroid of the same vertices of the same element arranged in anticlock-wise fashion
    
    NOTE: 
    This is different from Face_Centroid_Calculate(Edge_Node_Connectivity_new,Node_Coordinates) function.
    Here, face centroid is arranged in the anticlock-wise fashion.
    """
    
    Num_vertices = Anticlock_vertices.shape[0]
    dim = Anticlock_vertices.shape[1]
    face_centroid = np.zeros((Num_vertices,dim))
    for i in range(Num_vertices):
        if i == (Num_vertices - 1):
            face_centroid[i,:] = (Anticlock_vertices[0] + Anticlock_vertices[i])*0.5
        else:
            face_centroid[i,:] = (Anticlock_vertices[i] + Anticlock_vertices[i+1])*0.5
    return face_centroid


# In[52]:


@jit(nopython=True)
def Anti_Clock_Edge_Vectors(Anticlock_vertices):
    """
    Input: 
    Anticlock_vertices: Vertices of an element in the anticlock-wise direction
    
    Output:
    Edge_vectors: Vectors along the edges of the same element such that they are along the anticlock wise direction
    Edge_vectors: Format: [x,y]
    """
    
    Num_vertices = Anticlock_vertices.shape[0]
    dim = Anticlock_vertices.shape[1]
    Edge_vectors = np.zeros((Num_vertices,dim))
    for i in range(Num_vertices):
        if i == (Num_vertices - 1):
            Edge_vectors[i,:] = Anticlock_vertices[0,:] - Anticlock_vertices[i,:]
        else:
            Edge_vectors[i,:] = Anticlock_vertices[i+1,:] - Anticlock_vertices[i,:]
    return Edge_vectors


# @jit(nopython=True)
# def AntiClock_Edges_local(Anticlock_vertices,Node_Coordinates,Edge_Node_Connectivity_new):
#     """
#     Input:
#     Anticlock_vertices: Vertices of an element aligned in anticlock-wise fashion
#     Node_Coordinates_2D: Node Coordinates 2D (Previously Calculated)
#     Edge_Node_Connectivity_new: Edge Node Connectivity new (Previously Calculated)
#     
#     Output:
#     Anticlock_Edges: Arrange the edges of the same element in the anticlock-wise fashion locally.
#     """
#     
#     Num_vertices = 3
#     Anticlock_nodes = np.zeros((Num_vertices,1))
#     
#     # Converting the 3D coordinates to 2D coordinates
#     Node_Coordinates_2D = Node_Coordinates[:,:2]
#     
#     ctr = 0
#     for vertices in Anticlock_vertices:
#         node = 0
#         for coordinates in Node_Coordinates_2D:
#             if (vertices == coordinates).all():
#                 Anticlock_nodes[ctr,0] = node
#             node = node + 1
#         ctr = ctr + 1
#     Anticlock_Edges = np.zeros((Num_vertices,1))
#     for k in range(Anticlock_nodes.shape[0]):
#         if k == (Anticlock_nodes.shape[0] - 1):
#             node = Anticlock_nodes[k,0]
#             node_1 = Anticlock_nodes[0,0]
#         else:
#             node = Anticlock_nodes[k,0]
#             node_1 = Anticlock_nodes[k+1,0]
#         if np.argwhere(np.sum(Edge_Node_Connectivity_new[:,1:] == [node,node_1],axis=1) == 2).shape == (0,1):
#             Edge_nodes = Edge_Node_Connectivity_new[np.argwhere(np.sum(Edge_Node_Connectivity_new[:,1:] == [node_1,node],axis=1) == 2)[0,0]]
#         else:
#             Edge_nodes = Edge_Node_Connectivity_new[np.argwhere(np.sum(Edge_Node_Connectivity_new[:,1:] == [node,node_1],axis=1) == 2)[0,0]]
#         Anticlock_Edges[k] = Edge_nodes[0]
#     return Anticlock_Edges

# In[53]:


@jit(nopython=True)
def AntiClock_Edges_local(Anticlock_vertices,Node_Coordinates,Edge_Node_Connectivity_new):
    """
    Input:
    Anticlock_vertices: Vertices of an element aligned in anticlock-wise fashion
    Node_Coordinates_2D: Node Coordinates 2D (Previously Calculated)
    Edge_Node_Connectivity_new: Edge Node Connectivity new (Previously Calculated)
    
    Output:
    Anticlock_Edges: Arrange the edges of the same element in the anticlock-wise fashion locally.
    """
    
    Num_vertices = 3
    Anticlock_nodes = np.zeros((Num_vertices,1))
    
    # Converting the 3D coordinates to 2D coordinates
    Node_Coordinates_2D = Node_Coordinates[:,:2]
    
    ctr = 0
    for vertices in Anticlock_vertices:
        node = 0
        for coordinates in Node_Coordinates_2D:
            if (vertices == coordinates).all():
                Anticlock_nodes[ctr,0] = node
            node = node + 1
        ctr = ctr + 1
    Anticlock_Edges = np.zeros((Num_vertices,1))
    for k in range(Anticlock_nodes.shape[0]):
        if k == (Anticlock_nodes.shape[0] - 1):
            node = Anticlock_nodes[k,0]
            node_1 = Anticlock_nodes[0,0]
        else:
            node = Anticlock_nodes[k,0]
            node_1 = Anticlock_nodes[k+1,0]

        # print(Edge_Node_Connectivity_new[:,1:] == np.array([node,node_1]))
        
        temp = Edge_Node_Connectivity_new[:,1:] == np.array([node,node_1])
        temp_1 = Edge_Node_Connectivity_new[:,1:] == np.array([node_1,node])
        temp = temp.sum(axis = 1)
        temp_1 = temp_1.sum(axis = 1)

        key = np.array(np.argwhere(temp == 2).shape) == np.array([0,1])
        # print(key.sum())

        if key.sum() == 2:
            Edge_nodes = Edge_Node_Connectivity_new[np.argwhere(temp_1 == 2)[0,0]]
        else:
            Edge_nodes = Edge_Node_Connectivity_new[np.argwhere(temp == 2)[0,0]]
        Anticlock_Edges[k] = Edge_nodes[0]
        
        
    return Anticlock_Edges


# In[54]:


Element_Node_Connectivity[0]
Element_Node_Coordinates_l = np.zeros((3,2))
Element_Node_Coordinates_l[0,:] = Node_Coordinates[int(Element_Node_Connectivity[0,1])][0:2]
Element_Node_Coordinates_l[1,:] = Node_Coordinates[int(Element_Node_Connectivity[0,2])][0:2]
Element_Node_Coordinates_l[2,:] = Node_Coordinates[int(Element_Node_Connectivity[0,3])][0:2]


# In[55]:


Anticlock_vertices = Anti_Clock_Triangle_vertices(Element_Node_Coordinates_l)
AntiClock_Edges_local(Anticlock_vertices,Node_Coordinates,Edge_Node_Connectivity_new)


# In[ ]:





# In[56]:


def Element_Wise_Plotter(Anticlock_vertices,face_centroid,centroid):
    """
    Input:
    Anticlock_vertices: Vertices of an element in the anticlock-wise fashion
    face_centroid: Centroid of the faces of the same element in the anticlock-wise fashion
    centroid: Centroid of the element
    
    Output:
    Plots one element locally
    """
    
    Num_vertices = 3
    dim = 2
    for i in range(Num_vertices):
        plt.plot(Anticlock_vertices[i,0],Anticlock_vertices[i,1],"d",label = f"v: {i}")
    for i in range(Num_vertices):
        plt.plot(face_centroid[i,0],face_centroid[i,1],"d",label = f"cen: {i}")
    plt.plot(centroid[0],centroid[1],"kd")
    plt.legend()
    plt.axis('scaled')
    plt.show()


# In[57]:


def Element_all_prop(vertices,Node_Coordinates,Edge_Node_Connectivity_new):
    """
    Inputs: 
    vertices: Vertices of an element
    Node_Coordinates: Node Coordinates (Previously Calculated)
    Edge_Node_Connectivity_new: Edge Node Connectivity new (Previously Calculated)
    
    Outputs:
    centroid: Centroid of the element
    cen_to_face_cen: Centroid to face centroid vectors
    k_p_prime_vec: kp' vector of the fig. 9.19, page 312, chapter 9, the below book
    [Ferziger JH, Perić M, Street RL. Computational methods for fluid dynamics. springer; 2019 Aug 16.]
    [Chapter: 9,Fig. 9.19: On the approximation of diffusion fluxes for arbitrary polyhedral CVs]
    [Page: 312]
    Anticlock_vertices: Vertices of the same element arranged in the anticlock-wise fashion
    Anticlock_Edges: Edges of the same element arranged in the anticlock-wise fashion
    """
    
    Num_vertices = 3
    dim = 2
    
    # Converting the 3D coordinates to 2D coordinates
    Node_Coordinates_2D = Node_Coordinates[:,:2]
    
    centroid = vertices.mean(axis = 0)
    Anticlock_vertices = Anti_Clock_Triangle_vertices(vertices)
    face_centroid = Calculate_Face_Centroid(Anticlock_vertices)
    Edge_vectors = np.zeros((Num_vertices,dim))
    for i in range(Num_vertices):
        if i == (Num_vertices - 1):
            Edge_vectors[i,:] = Anticlock_vertices[0,:] - Anticlock_vertices[i,:]
        else:
            Edge_vectors[i,:] = Anticlock_vertices[i+1,:] - Anticlock_vertices[i,:]
    Anti_clock_edge_vec = Anti_Clock_Edge_Vectors(Anticlock_vertices)
    Edge_perp_vectors = np.zeros((Num_vertices,dim))
    Edge_perp_vectors[:,0] = Anti_clock_edge_vec[:,1]
    Edge_perp_vectors[:,1] = -Anti_clock_edge_vec[:,0]
    cen_to_face_cen = np.zeros((Num_vertices,dim))
    cen_to_face_cen = face_centroid - centroid
    Edge_perp_vectors_out = Edge_perp_vectors.copy()
    for i in range(Num_vertices):
        temp = np.dot(Anti_clock_edge_vec[i],cen_to_face_cen[i])
        # If it is outward then it would be positive
        if temp < 0:
            Edge_perp_vectors_out[i,:] = -Edge_perp_vectors_out[i,:]
    Edge_perp_vectors_out_unit = np.zeros(Edge_perp_vectors_out.shape)
    for i in range(Num_vertices):
        Edge_perp_vectors_out_unit[i] = (Edge_perp_vectors_out[i])/(((Edge_perp_vectors_out[i]**2).sum())**0.5)
    cen_to_face_cen_perp_distance = (cen_to_face_cen*Edge_perp_vectors_out_unit).sum(axis = 1)
    # Assume this for now
    # See this in Peric's book and modify will integrating
    # Page no.: 310
    # Equation: 9.40
    a = cen_to_face_cen_perp_distance
    # We should make a edge wise array for a while integration
    k_p_prime_vec = np.zeros((Num_vertices,2))
    for i in range(Num_vertices):
        k_p_prime_vec[i] = -Edge_perp_vectors_out_unit[i]*a[i]
    p_p_prime_vec = k_p_prime_vec + cen_to_face_cen
    Anticlock_Edges = AntiClock_Edges_local(Anticlock_vertices,Node_Coordinates_2D,Edge_Node_Connectivity_new)
    # This will plot the local element
    # Element_Wise_Plotter(Anticlock_vertices,face_centroid,centroid)
    return centroid,cen_to_face_cen,k_p_prime_vec,Anticlock_vertices,Anticlock_Edges


# In[58]:


def Check_Element_all_prop(Element_Node_Connectivity_new,Node_Coordinates,Edge_Node_Connectivity_new):
    """
    Input:
    Element_Node_Connectivity_new: Element Node Connectivity new (Previously Calculated)
    Node_Coordinates_2D: Node Coordinates 2D (Previously Calculated)
    Edge_Node_Connectivity_new: Edge Node Connectivity new (Previously Calculated)
    Output:
    Checks if Element_all_prop if working or not.
    If no error is raised, then it is working properly.
    """
    # Converting the 3D coordinates to 2D coordinates
    Node_Coordinates_2D = Node_Coordinates[:,:2]
    
    for i in range(Element_Node_Connectivity_new.shape[0]):
        Element_Node_Coordinates_l = np.zeros((Num_vertices,dim))
        Nodes = Element_Node_Connectivity_new[i,1:]
        ctr = 0
        for node in Nodes:
            Element_Node_Coordinates_l[ctr,:] = Node_Coordinates_2D[int(node)]
            ctr = ctr +1
        centroid,cen_to_face_cen,k_p_prime_vec,Anticlock_vertices,Anticlock_Edges = Element_all_prop(Element_Node_Coordinates_l,Node_Coordinates_2D,Edge_Node_Connectivity_new)    


# Check_Element_all_prop(Element_Node_Connectivity_new,Node_Coordinates,Edge_Node_Connectivity_new)

# In[59]:


def Edge_Element_Connectivity_Calculate(Edge_Node_Connectivity_new,Element_Element_Connectivity_new,Element_Edge_Connectivity_new,Boundary_Edges):
    """
    Input:
    Edge_Node_Connectivity_new: Edge Node Connectivity new (Previously Calculated)
    Element_Element_Connectivity_new: Element Element Connectivity new (Previously Calculated)
    Element_Edge_Connectivity_new: Element Edge Connectivity new (Previously Calculated)
    Boundary_Edges: Boundary Edges (Previously Calculated)
    
    Output:
    Edge_Element_Connectivity: Format: [Edge,Element1,Element2]
    """
    
    Num_Edges = Edge_Node_Connectivity_new.shape[0]
    Edge_Element_Connectivity = np.zeros((Num_Edges,3),dtype = int)
    common_edge_set = set()
    ctr = 0
    for i in range(Element_Element_Connectivity_new.shape[0]):
        Element = Element_Element_Connectivity_new[i,0]
        Element_nb = Element_Element_Connectivity_new[i,1:]
        Edges = Element_Edge_Connectivity_new[int(Element),1:]
        # print(Edges)
        for j in range(Element_nb.shape[0]):
            element = int(Element_nb[j])
            if Element != element:
                edges = Element_Edge_Connectivity_new[element,1:]
                # print(edges)
                # print(Element_Element_Connectivity_new[i])
                common_edge = np.intersect1d(Edges,edges)
                # print(common_edge)
                if common_edge[0] in common_edge_set:
                    pass
                else:
                    common_edge_set.add(common_edge[0])
                    Edge_Element_Connectivity[ctr,0] = common_edge[0]
                    Edge_Element_Connectivity[ctr,1] = Element
                    Edge_Element_Connectivity[ctr,2] = element
                    ctr = ctr +1

    for i in range(Element_Edge_Connectivity_new.shape[0]):
        Element = Element_Edge_Connectivity_new[i,0]
        Edges = Element_Edge_Connectivity_new[i,1:]
        common_edge = np.intersect1d(Edges,Boundary_Edges)
        # print(Edges)
        # print(common_edge)
        if len(common_edge) != 0:
            for j in range(common_edge.shape[0]):
                common_edge_set.add(common_edge[0])
                # print(common_edge)
                # GMSH BUG RESOLVED
                # This bug is also there in other versions  of the code
                Edge_Element_Connectivity[ctr,0] = common_edge[j]
                Edge_Element_Connectivity[ctr,1] = Element
                Edge_Element_Connectivity[ctr,2] = Element
                ctr = ctr +1
    temp = Edge_Element_Connectivity.copy()
    Edge_Element_Connectivity = temp[np.argsort(temp[:,0])]
    return Edge_Element_Connectivity


# In[60]:


Edge_Element_Connectivity = Edge_Element_Connectivity_Calculate(Edge_Node_Connectivity_new,Element_Element_Connectivity_new,Element_Edge_Connectivity_new,Boundary_Edges)


# In[61]:


def Diffusion_mesh_data_Calculate(Num_Triangles,Element_Node_Connectivity_new,Element_Edge_Connectivity_new,Node_Coordinates,Edge_Node_Connectivity_new,Edge_Element_Connectivity):    
    """
    Output:
    Diffusion_mesh_data
    Element_cen
    
    # We only need pp', NkNk' and |p'Nk'| from this data
    # Note |p'Nk'| = 2|kp'|
    # So, we are storing it in an array of the format
    # cell number, edge number, pp'[0], pp'[1], NkNk'[0], NkNk'[1],|p'Nk'|
    # More data needed for advection
    # cell number, edge number, pp'[0], pp'[1], NkNk'[0], NkNk'[1],|p'Nk'|,-unit(kp'[0]),-unit(kp'[1])
    """
    Diffusion_mesh_data = np.zeros((Num_Triangles*3,9))
    Element_cen = np.zeros((Num_Triangles,3))
    
    # Converting the 3D coordinates to 2D coordinates
    Node_Coordinates_2D = Node_Coordinates[:,:2]
    
    d_ctr = 0
    Num_vertices = 3
    dim = 2
    
    for i in range(Element_Node_Connectivity_new.shape[0]):
        # print(i)
        Element = Element_Edge_Connectivity_new[i,0]
        Element_Node_Coordinates_l = np.zeros((Num_vertices,dim))
        Nodes = Element_Node_Connectivity_new[i,1:]
        ctr = 0
        for node in Nodes:
            Element_Node_Coordinates_l[ctr,:] = Node_Coordinates_2D[int(node)]
            ctr = ctr +1
        centroid,cen_to_face_cen,k_p_prime_vec,Anticlock_vertices,Anticlock_Edges = Element_all_prop(Element_Node_Coordinates_l,Node_Coordinates_2D,Edge_Node_Connectivity_new)
        Element_cen[i,0] = Element
        Element_cen[i,1:] = centroid
        # print(centroid)
        for j in range(Anticlock_Edges.shape[0]):
            edge = Anticlock_Edges[j]
            # print(edge)
            # print(Edge_Element_Connectivity[int(edge)])
            Elements = Edge_Element_Connectivity[int(edge[0]),1:]
            if Elements[0] != Elements[1]:
                # Internal Elements
                Element_Node_Coordinates_l_1 = np.zeros((Num_vertices,dim))
                if Element == Elements[0]:
                    element = Elements[1]
                else:
                    element = Elements[0]
                # print(element)
                # print(Elements)

                Nodes_1 = Element_Node_Connectivity_new[element,1:]
                ctr = 0
                for node in Nodes_1:
                    Element_Node_Coordinates_l_1[ctr,:] = Node_Coordinates_2D[int(node)]
                    ctr = ctr +1
                centorid,cen_to_face_cen_1,k_p_prime_vec_1,Anticlock_vertices_1,Anticlock_Edges_1 = Element_all_prop(Element_Node_Coordinates_l_1,Node_Coordinates_2D,Edge_Node_Connectivity_new)
                temp = np.intersect1d(Anticlock_Edges,Anticlock_Edges_1)
                
                # print(temp)
                if temp.shape[0] == 1:
                    # print(k_p_prime_vec[(Anticlock_Edges == temp)[:,0]])
                    # print(k_p_prime_vec_1[(Anticlock_Edges_1 == temp)[:,0]])
                    # print(temp)
                    k_p_prime_vec_i = k_p_prime_vec[(Anticlock_Edges == temp)[:,0]]
                    k_p_prime_vec_e = k_p_prime_vec_1[(Anticlock_Edges_1 == temp)[:,0]]
                    cen_to_face_cen_i = cen_to_face_cen[(Anticlock_Edges == temp)[:,0]]
                    cen_to_face_cen_1_e = cen_to_face_cen_1[(Anticlock_Edges_1 == temp)[:,0]]
                    if np.dot(k_p_prime_vec_i[0],k_p_prime_vec_i[0]) > np.dot(k_p_prime_vec_e[0],k_p_prime_vec_e[0]):
                        # We need to replace if and only if the projected centroid on the normal line 
                        # in the neighbouring cell is smaller 
                        # This is the calculation from peric
                        # Remember to reverse the direction of the vector in this case
                        k_p_prime_vec_i = -k_p_prime_vec_e

                    pp_prime_vec = cen_to_face_cen_i + k_p_prime_vec_i
                    nk_nk_prime_vec = cen_to_face_cen_1_e - k_p_prime_vec_i

                    Diffusion_mesh_data[d_ctr,0] = Element
                    Diffusion_mesh_data[d_ctr,1] = edge[0]
                    Diffusion_mesh_data[d_ctr,2:4] = pp_prime_vec
                    Diffusion_mesh_data[d_ctr,4:6] = nk_nk_prime_vec
                    Diffusion_mesh_data[d_ctr,6] = 2*((np.dot(k_p_prime_vec_i[0],k_p_prime_vec_i[0]))**0.5)
                    # For advec
                    Diffusion_mesh_data[d_ctr,7] = -k_p_prime_vec_i[0][0]/((np.dot(k_p_prime_vec_i[0],k_p_prime_vec_i[0]))**0.5)
                    Diffusion_mesh_data[d_ctr,8] = -k_p_prime_vec_i[0][1]/((np.dot(k_p_prime_vec_i[0],k_p_prime_vec_i[0]))**0.5)
                    d_ctr = d_ctr + 1
                    # print(Element)
            else:
                # Boundary Elements
                cen_to_face_cen_i = cen_to_face_cen[j]
                k_p_prime_vec_i = k_p_prime_vec[j]
                pp_prime_vec = cen_to_face_cen_i + k_p_prime_vec_i
                # Same thing mirrored
                nk_nk_prime_vec = pp_prime_vec
                Diffusion_mesh_data[d_ctr,0] = Element
                Diffusion_mesh_data[d_ctr,1] = edge[0]
                Diffusion_mesh_data[d_ctr,2:4] = pp_prime_vec
                Diffusion_mesh_data[d_ctr,4:6] = nk_nk_prime_vec 
                Diffusion_mesh_data[d_ctr,6] = 2*((np.dot(k_p_prime_vec_i,k_p_prime_vec_i))**0.5)
                # For advec
                Diffusion_mesh_data[d_ctr,7] = -k_p_prime_vec_i[0]/((np.dot(k_p_prime_vec_i,k_p_prime_vec_i))**0.5)
                Diffusion_mesh_data[d_ctr,8] = -k_p_prime_vec_i[1]/((np.dot(k_p_prime_vec_i,k_p_prime_vec_i))**0.5)    
                d_ctr = d_ctr + 1
                # print(Element)
                # print(k_p_prime_vec_i)
                
    return Diffusion_mesh_data,Element_cen


# In[62]:


Diffusion_mesh_data,Element_cen = Diffusion_mesh_data_Calculate(Num_Triangles,Element_Node_Connectivity_new,Element_Edge_Connectivity_new,Node_Coordinates,Edge_Node_Connectivity_new,Edge_Element_Connectivity)


# In[63]:


@jit(nopython=True)
def Tri_area(Node_Coordinates_element):
    """
    Input:
    The coordinates of the triangle
    
    Output:
    The area of the triangle
    """
    temp = np.ones((3,3))
    temp[:,:2] = Node_Coordinates_element
    temp = 0.5*abs(np.linalg.det(temp))
    return temp


# In[64]:


@jit(nopython=True)
def Element_Area_Calculate(Num_Triangles,Element_Node_Connectivity_new,Node_Coordinates):
    """
    Input: 
    Element_Node_Connectivity_new: Element Node Connectivity new (Previuosly Calculated)
    Node_Coordinates: Node Coordinates (Previously Calculated)
    
    Output:
    Element_Area: Format: [Element, Area]
    """
    
    Element_Area = np.zeros((Num_Triangles,2))
    
    # Converting the 3D coordinates to 2D coordinates
    Node_Coordinates_2D = Node_Coordinates[:,:2]
    
    dim = 2
    
    for i in range(Element_Node_Connectivity_new.shape[0]):
        Element = Element_Node_Connectivity_new[i,0]
        Nodes = Element_Node_Connectivity_new[i,1:]
        # print(Nodes)
        Node_Coordinates_element = np.zeros((3,dim))
        for j in range(Nodes.shape[0]):
            node = Nodes[j]
            Node_Coordinates_element[j,:] = Node_Coordinates_2D[int(node)]
        Element_Area[i,0] = Element
        Element_Area[i,1] = Tri_area(Node_Coordinates_element)
        # print(Tri_area(Node_Coordinates_element))
        
    return Element_Area


# In[65]:


Element_Area = Element_Area_Calculate(Num_Triangles,Element_Node_Connectivity_new,Node_Coordinates)


# In[66]:


Num_Edges = Edge_Node_Connectivity_new.shape[0]


# In[67]:


def Edge_Len_Calculate(Num_Edges,Edge_Node_Connectivity_new,Node_Coordinates):
    """
    Input:
    Num_Edges: Number of edges in the mesh
    Edge_Node_Connectivity_new: Edge Node Connectivity new (Previously Calculated)
    Node_Coordinates: Node Coordinates (Previously Calculated)
    
    Output:
    Edge_Len: Format: [Edge,Edge_Len]
    """
    Edge_Len = np.zeros((Num_Edges,2))

    # Converting the 3D coordinates to 2D coordinates
    Node_Coordinates_2D = Node_Coordinates[:,:2]
    
    for i in range(Edge_Node_Connectivity_new.shape[0]):
        Edge = Edge_Node_Connectivity_new[i,0]
        Nodes = Edge_Node_Connectivity_new[i,1:]
        Coor_0 = Node_Coordinates_2D[int(Nodes[0])]
        Coor_1 = Node_Coordinates_2D[int(Nodes[1])]
        temp = Coor_0 - Coor_1
        edge_len = (np.dot(temp,temp))**0.5
        Edge_Len[i,0] = Edge
        Edge_Len[i,1] = edge_len
    
    return Edge_Len


# In[68]:


Edge_Len = Edge_Len_Calculate(Num_Edges,Edge_Node_Connectivity_new,Node_Coordinates)


# ****************************************************

# # Functions to collect mesh data for the solver:

# ********************

# In[84]:


@jit(nopython=True)
def P_prime_Nk_prime_len(Diffusion_mesh_data,Element,Edge):
    """
    Input:
    Diffusion_mesh_data,
    Element
    Edge
    
    Output:
    Length of the p_prime_Nk_prime vector
    """
    
    for i in range(3):
        # Element is matched
        if Diffusion_mesh_data[(3*Element)+i,1] == Edge:
            p_prime_Nk_prime_len = Diffusion_mesh_data[(3*Element)+i,6]
            return p_prime_Nk_prime_len


# In[85]:


@jit(nopython=True)
def p_p_prime_data(Diffusion_mesh_data,Element,Edge):
    """
    Input:
    Diffusion_mesh_data,
    Element
    Edge
    
    Output:
    p_p_prime vector
    """
    
    for i in range(3):
        # Element is matched
        if Diffusion_mesh_data[(3*Element)+i,1] == Edge:
            p_p_prime = Diffusion_mesh_data[(3*Element)+i,2:4]
            return p_p_prime


# In[86]:


@jit(nopython=True)
def nk_nk_prime_data(Diffusion_mesh_data,Element,Edge):
    """
    Input:
    Diffusion_mesh_data,
    Element
    Edge
    
    Output:
    nk_nk_prime vector
    """
    
    for i in range(3):
        # Element is matched
        if Diffusion_mesh_data[(3*Element)+i,1] == Edge:
            nk_nk_prime = Diffusion_mesh_data[(3*Element)+i,4:6]
            return nk_nk_prime


# In[87]:


@jit(nopython=True)
def neg_k_p_prime_data(Diffusion_mesh_data,Element,Edge):
    """
    Input:
    Diffusion_mesh_data,
    Element
    Edge
    
    Output:
    neg_k_p_prime vector
    """
    for i in range(3):
        # Element is matched
        if Diffusion_mesh_data[(3*Element)+i,1] == Edge:
            neg_k_p_prime = Diffusion_mesh_data[(3*Element)+i,7:9]
            return neg_k_p_prime

