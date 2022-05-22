import numpy as np
from ase import Atoms
from ase.io import read,write
import sys

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
    si_pos = pos[:50,:]
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
    cell = np.hstack((length, angle))
    pos=x[2:, :]

    pos,n_si = remove_zero_padding(pos)

    scaled_pos = back_to_10_cell(pos,n_si)
    atoms = back_to_real_cell(scaled_pos, cell, n_si)
    atoms.set_pbc([1,1,1])
    if view:
        atoms.edit()
    return atoms, x


if '__name__' == '__main__':
        pass
else:
        print("import")


