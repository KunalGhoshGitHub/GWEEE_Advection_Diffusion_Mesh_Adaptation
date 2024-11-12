import subprocess

print(f"Coping files for the initial mesh!")

path_to_initial_mesh = "./Initial_Mesh/*"

subprocess.run(f"cp {path_to_initial_mesh} .", shell = True)

print(f"Initial mesh copied!")
