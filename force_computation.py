#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""Module dealing with computing coulombic forces between atoms or residues."""

import numpy as np
from numba import jit

def compute_coulomb_from_atom(atom_index_1, atom_index_2, charges, pos_frame):
    """Return electrostatic force between 2 atoms, from indices and data tables. Not used."""
    pos_1 = pos_frame[atom_index_1]
    pos_2 = pos_frame[atom_index_2]
    charge_1 = charges[atom_index_1]
    charge_2 = charges[atom_index_2]
    force = compute_coulomb(pos_1,pos_2,charge_1,charge_2)

    return force

@jit(nopython = True)
def compute_coulomb(pos_1,pos_2,charge_1,charge_2):
    """
    Compute coulombic force for a pair of particles.

    Keyword arguments:
    pos_1 -- a numpy 1D array with length 3, containing the coordinates of the first particle
    pos_2 -- a numpy 1D array with length 3, containing the coordinates of the second particle
    charge_1 -- a numpy 0D array, containing the charge of the first particle
    charge_2 -- a numpy 0D array, containing the charge of the second particle
    
    Output:
    force -- a numpy 1D array with length 3, containing the electrostatic force

    """
    return charge_1*charge_2*(pos_1-pos_2)/(np.linalg.norm(pos_1-pos_2))**3

@jit(nopython = True)
def compute_energy(pos_1,pos_2,charge_1,charge_2):
    """
    Compute coulombic force for a pair of particles.

    Keyword arguments:
    pos_1 -- a numpy 1D array with length 3, containing the coordinates of the first particle
    pos_2 -- a numpy 1D array with length 3, containing the coordinates of the second particle
    charge_1 -- a numpy 0D array, containing the charge of the first particle
    charge_2 -- a numpy 0D array, containing the charge of the second particle
    
    Output:
    force -- a numpy 1D array with length 3, containing the electrostatic force

    """
    return charge_1*charge_2*(pos_1-pos_2)/(np.linalg.norm(pos_1-pos_2))**2

def compute_residue_force(position_1, position_2, charges_1, charges_2):
    """
    Compute sum of coulombic forces for all pairs of particles in two residues.

    Keyword arguments:
    position_1 -- a numpy 2D array of coordinates for elements of first residue
    position_2 -- a numpy 2D array of coordinates for elements of second residue
    charges_1 -- a numpy 2D array of charges for elements of first residue
    charges_2 -- a numpy 2D array of charges for elements of second residue
       
    Output:
    force -- a numpy 1D array with length 3, containing the electrostatic force
    """
    force = np.zeros(3)
    for i in range(len(position_1)):
        for j in range(len(position_2)):
            force += compute_coulomb(position_1[i],position_2[j],charges_1[i],charges_2[j])
    return force