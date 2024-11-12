import subprocess

print(f"Cleaning started !")

# Cleans the metric file
subprocess.run("rm *.mtr",shell = True)

# Cleans the error files
subprocess.run("rm *.csv",shell = True)

# Cleans the BAMG mesh files
subprocess.run("rm *.mesh",shell = True)

# Cleans the BAMG geometry file
subprocess.run("rm *.geo",shell = True)

# Cleans the GMSH mesh file
subprocess.run("rm *.msh",shell = True)

# Cleans the image file
subprocess.run("rm *.png",shell = True)
subprocess.run("rm *.svg",shell = True)

print(f"Cleaning done !")
