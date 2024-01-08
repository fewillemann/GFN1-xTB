# Author: Felipe Reibnitz Willemann
from model import GFN1_xTB
from pathlib import Path
import os

# defining relevant paths
src_path = Path(__file__).parent
par_file = os.path.join(src_path, "parameters.dat")

# check parameter file
if not os.path.exists(par_file):
    print(f"ERROR: No parameters file found on {src_path}")
    exit()

# get coordinate files directory path
cwd_path = Path(os.getcwd())
coord_dir = os.path.join(
    cwd_path,
    input("Enter relative path of directory with coordinate files (from CWD): "),
)

# check coordinate directory and files
if not os.path.exists(coord_dir):
    print("ERROR: directory with coordinate files not found!")
    exit()

list_coords = os.listdir(coord_dir)
if ".xyz" not in [os.path.splitext(file)[1] for file in list_coords]:
    print(f"ERROR: No XYZ files found on {coord_dir}")
    exit()

# get output directory path
out_asw = input("Save outputs under current working directory? [y/n] ")
if out_asw.upper() == "Y":
    out_dir = os.path.join(cwd_path, "outputs")

elif out_asw.upper() == "N":
    out_dir = os.path.join(src_path.parent, "outputs")

else:
    print("ERROR: unrecognized option.")
    exit()

# check output path
if not os.path.exists(out_dir):
    print("INFO: Outputs directory not found, a new one was created.")
    os.mkdir(out_dir)

print(f"INFO: Outputs are going to be saved on {out_dir}")

# get max number of iterations
maxiter = int(input("Enter maximum number of SCF iterations: "))

# initializing model
model = GFN1_xTB(par_file=par_file)

for file in list_coords:
    base, ext = os.path.splitext(file)
    if ext != ".xyz":
        continue

    geom_file = os.path.join(coord_dir, file)
    out_file = os.path.join(out_dir, f"{base}.out")

    print(f"INFO: Loading {base} geometry.")
    load_result = model.load_geometry(geom_file=geom_file)
    if not load_result:
        print("Calculation aborted!")
        continue

    print(f"INFO: Staring {base} SCF calculation.")
    model.scf(maxiter=maxiter, out_file=out_file)
