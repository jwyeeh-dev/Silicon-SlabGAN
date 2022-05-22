import glob
from ase.io import read,write
from ase import Atoms, Atom
from collections import Counter
import numpy as np
import pandas as pd
from tqdm import tqdm
from make_representation import *
import pickle

cif_path = './cifs'
vasp_path = '/home/jwyeeh/dataset/original_dataset/revised_data_vasp_sfe/'
cif_list = glob.glob(cif_path+'/*.cif')
vasp_list = glob.glob(vasp_path+'/*.vasp')
vasp_list1 = os.listdir(vasp_path)

results = []
name_list = [] 

for ele in tqdm(vasp_list1):
    with open(vasp_path + ele,'r') as f:
        lines = f.readlines()



for i in lines:
    temp = i.strip().split(',')

    #for i in cif_list:
for i in tqdm(vasp_list):
    name = i.split(',')[0]
    atoms =read(i)
    s = atoms.get_chemical_symbols()
    n_si = s.count('Si')
    print(atoms)

    image = do_feature(atoms)
    print(image)

    results.append(image)
    name_list.append(name)

results = np.array(results)
print(results.shape)
np.save('unique_slab',results)

with open("unique_slab_name_list",'wb') as f:
    pickle.dump(name_list,f)

