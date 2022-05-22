import numpy as np
from ase import Atoms
from ase.io import read,write
import sys
import torch
import ase

def back_to_10_cell(scaled_pos,n_si):
    cell = np.identity(3)*15
    atoms = Atoms('Si'+str(n_si))
    atoms.set_cell(cell)
    atoms.set_scaled_positions(scaled_pos)
    pos = atoms.get_positions()	

    cell = np.identity(3)*10
    pos = pos - np.array([2.5,2.5,2.5])
    atoms = Atoms('Si'+str(n_si))
    atoms.set_cell(cell)
    atoms.set_positions(pos)
    scaled_poss = atoms.get_scaled_positions()
    return scaled_poss

def back_to_real_cell(scaled_pos, real_cell, n_si):
    atoms = Atoms('Si'+str(n_si))
    atoms.set_cell(real_cell)
    atoms.set_scaled_positions(scaled_pos)
    return atoms

def remove_zero_padding(pos):
    criteria = 0.4
    si_pos = pos[:1000,:]
    si = np.sum(si_pos, axis=1)
    si_index = np.where(si > criteria)
    n_si = len(si_index[0])
    si_pos = si_pos[si_index]
    if n_si == 0:
        si_pos = np.array([0.1667,0.1667,0.1667]).reshape(1,3)
        n_si = 1

    pos = np.vstack(si_pos)
    return pos, n_si

def view_atoms(image, view = True):

    x = image
    x = x.reshape(-1,3)

    length = x[0,:]*30
    angle = x[1,:]*180
    cell = np.hstack((2*length, angle))
    pos=x[2:, :]
    re_pos = pos + (length / 2)
    re_pos,n_si = remove_zero_padding(re_pos)

    scaled_pos = back_to_10_cell(re_pos,n_si)
    atoms = back_to_real_cell(scaled_pos, cell, n_si)
    atoms.set_pbc([1,1,1])
    
    if view:
        atoms.edit()
    return atoms, x

def view_atoms_classifier(image, si_label, view=True):
	x= image.reshape(-1,3)
	si = x[2:1002,:]

	l = x[0,:]*30
	a = x[1,:]*180
	c = np.hstack((l,a))
	atoms = Atoms('H')
	atoms.set_cell(c)
	cell = atoms.get_cell()
	t = np.isnan(cell)
	tt = np.sum(t)
	isnan = False
	if not tt == 0:
		isnan = True
		print(cell)
		print(l)
		print(a)
	_,si_index = torch.max(si_label,dim=1)
	
	si_index = si_index.reshape(50,).detach().cpu().numpy()
	
	si_pos = si[np.where(si_index)]
	
	n_si = len(si_pos)
	
	if n_si == 0:
		si_pos = np.array([0.1667,0.1667,0.1667]).reshape(1,3)
		n_si = 1

	pos = np.vstack((si_pos))
	scaled_pos = back_to_10_cell(pos,n_si)
	atoms = back_to_real_cell(scaled_pos, cell, n_si)
	atoms.set_pbc([1,1,1])
	if view :
		atoms.edit()
			
	return atoms, x, isnan

if '__name__' == '__main__':
        pass
		
else:
        print("import")



