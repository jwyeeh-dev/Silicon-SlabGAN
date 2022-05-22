import numpy as np
import view_atoms_slab
import pickle
from ase.io import read,write
from ase import Atoms,Atom
from tqdm import tqdm
from collections import Counter



dat = np.load('unique_slab.npy')
with open("unique_slab_name_list", 'rb') as f:
    name_list = pickle.load(f)
comp_image_dict = {}
comp_dict = []
n_v_list =[]
n_o_list = []

for i in tqdm(range(dat.shape[0])):
    try:
        image = dat[i]
        name_index = name_list[i]
        atoms, image = view_atoms_slab.view_atoms(image,view=False)
        s = atoms.get_chemical_symbols()
        n_si = s.count('Si')
        comp = str(n_si)
        comp_dict.append(comp)
        dict_keys = comp_image_dict.keys()
        if not comp in dict_keys:
            comp_image_dict[comp] = [(image,name_index)]
        else:
            temp_list = comp_image_dict[comp]
            temp_list.append((image,name_index))
    except AssertionError:
        pass

with open('unique_slab_comp_dict','wb') as f:
    pickle.dump(comp_image_dict,f)

print(Counter(comp_dict))
