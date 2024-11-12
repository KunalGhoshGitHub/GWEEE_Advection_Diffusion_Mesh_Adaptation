#!/usr/bin/env python
# coding: utf-8

# # Importing necessary libraries:

# *************************

# In[6]:


import subprocess


# In[7]:


import os


# In[8]:


Files = os.listdir(".")


# # Getting the jupyter-notebooks

# ********************

# In[9]:


ipynb_Files = []
for file in Files:
    extension = file[-5:]
    if extension == "ipynb":
        ipynb_Files.append(file)


# # Getting the jupyter-notebooks to python scripts

# ***************************

# In[10]:


for file in ipynb_Files:
    
    # jupyter-notebook need to be converted
    jupyter_notebook_file = file
    
    # Do NOT add the extension *.py as it will be taken care by the command itself
    py_file = file[:-6]

    # print(f"Converting the jupyter-notebook: {jupyter_notebook_file} to python file: {py_file}")

    subprocess.run(f"jupyter nbconvert --to script {jupyter_notebook_file} --output {py_file}",shell = True)

    # print(f"Converted the jupyter-notebook: {jupyter_notebook_file} to python file: {py_file}")


# In[ ]:




