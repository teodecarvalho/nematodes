import os
import shutil
from glob import glob
path = "/Volumes/ArquivosTeo\ 1/imagens\ mais\ novas/imagens/"
files = os.listdir("/Volumes/ArquivosTeo 1/imagens mais novas/imagens")
for i, file in enumerate(files):
    print(i)
    fsplit = file.split("__")
    x = fsplit[1]
    y = fsplit[2]
    os.system(f"mv {path}nem__{x}__{y}__.tif {path}nem__{x}__{int(y) - 17}__.tif")
    os.system(f"mv {path}nem__{x}__{y}__.png {path}nem__{x}__{int(y) - 17}__.png")