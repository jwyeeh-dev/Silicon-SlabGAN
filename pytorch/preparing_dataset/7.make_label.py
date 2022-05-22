import numpy as np
from ase.io import read,write
from ase import Atoms

from view_atoms_slab import *
import pickle
from tqdm import tqdm



def check(image):
	pos = image[2:,:]
	si = pos[:1002,:]
	sisi = np.sum(si,axis=1)
	sisisi = np.zeros((1000,1)) + 1
	sisisi[sisi < 0.4] = 0
	label = np.vstack(sisisi)
	print(label.shape)
	return label




a = np.load("slab_1000.npy")

m = a.shape[0]
output = []
for i in tqdm(range(m)):
	x = a[i]
	label = check(x)
	print(x)
	print(label)
	new_input = (x,label)


	output.append(new_input)

with open('slab_1000.pickle', 'wb') as f:
	pickle.dump(output,f)


