import os
import numpy as np
from ase.io import  read, write
from ase import Atoms,Atom
import glob
from collections import Counter
from tqdm import tqdm

cell_max = 30
def make_condition(n,n_class):
    temp = np.zeros((n_class,1))
    temp[n-1,0] = 1
    return temp

def read_poscar(poscar_path, isatoms=False, atoms=None):

    if not isatoms:
        atoms = read(poscar_path)
    
    else:
        atoms = atoms
#    print(atoms)
    cell = atoms.get_cell()
    temp = atoms.get_cell_lengths_and_angles()
    symbols = atoms.get_chemical_symbols()
    pos = atoms.get_scaled_positions()
    
    return atoms, cell, symbols, pos, temp[:3], temp[3:]



def go_to_10_cell(scaled_pos,n_si):
    cell = np.array([[10,0,0],[0,10,0],[0,0,10]]).astype(float)

    atoms = Atoms('Si'+str(n_si))
    atoms.set_cell(cell)
    print(atoms)
    atoms.set_scaled_positions(scaled_pos)
    
    pos = atoms.get_positions()
    return pos

def go_to_15_cell(pos_10,n_si):
    cell = np.array([[15,0,0],[0,15,0],[0,0,15]]).astype(float)
    pos = pos_10 + np.array([2.5,2.5,2.5])

    atoms = Atoms('Si'+str(n_si))
    atoms.set_cell(cell)
    atoms.set_positions(pos)
    scaled_pos = atoms.get_scaled_positions()
    return scaled_pos

def make_onehot(n,n_class,e_pos):
    temp = np.zeros((n_class,3))
#   temp = temp -0.5
    for i,p in enumerate(e_pos):
        temp[i,:] = p
    return temp

def do_feature(atoms):

    atoms, cell, symbols, pos , lengths, angles = read_poscar(poscar_path = None, isatoms = True, atoms = atoms)
#    cell = atoms.get_cell()
#    symbols = atoms.get_chemical_symbols()
#    pos = atoms.get_scaled_positions()
#    cell = cell/15
    length = lengths/30
    length = length.reshape(1,3)
    angle = angles/180
    angle = angle.reshape(1,3)
    cell = np.vstack((length,angle))
    
    n_si = symbols.count('Si')
    comp = str(n_si)
    pos_10 = go_to_10_cell(pos,n_si)
    scaled_pos_15 = go_to_15_cell(pos_10,n_si)
    mg_pos = scaled_pos_15[:n_si,:]
    mg_pos_onehot = make_onehot(n_si,1000,mg_pos)
    pos_onehot = np.vstack(mg_pos_onehot)
    temp = np.vstack((cell,pos_onehot))
    inp = temp.reshape(-1,3)
    return inp


if __name__ == "__main__":
    pass
