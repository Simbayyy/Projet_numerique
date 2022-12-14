#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""Module dealing with splitting data by residues and computing corresponding graph edge matrix."""

import data_extraction as extract
import force_computation as force
import numpy as np
import os, pathlib, re
import pandas
from numba import jit

os.chdir(pathlib.Path(__file__).parent.resolve())

def same_system(atom_index_1, atom_index_2, molsizes):
    """
    Given two atoms, return True if they are in the same system and False otherwise.

    Keyword arguments:
    atom_index_1 -- index of first atom in the position array
    atom_index_2 -- index of second atom in the position array
    molsizes -- list of last atom numbers in each system of the PDB file

    Output:
    is_same_system -- boolean
    """
    for molsize in molsizes:
        if atom_index_1 < molsize and atom_index_2 >= molsize:
            return False
        if atom_index_2 < molsize and atom_index_1 >= molsize:
            return False
    return True

@jit(nopython = True)
def filter_residue_atoms(residue, positions, charges, resnums):
    """
    Generate a subset of position and charge arrays with only a given residue.

    Keyword arguments:
    residue -- the residue number
    positions -- the full position array for the current frame
    charges -- the full charge array
    resnums -- a numpy 1D array with the list of residue number for each atom (with length equal to atom count)
    
    Outputs:
    residue_position -- a numpy 2D array with cartesian coordinates for each atom in the residue
    residue_charge -- a numpy 1D array with charge for each atom in the residue
    """
    filter = resnums == residue
    return positions[filter], charges[filter]

def filter_residue_atoms_nojit(residue, positions, charges, resnums):
    """
    Generate a subset of position and charge arrays with only a given residue. No numba, slower but compatible with string-type residues.

    Keyword arguments:
    residue -- the residue number
    positions -- the full position array for the current frame
    charges -- the full charge array
    resnums -- a numpy 1D array with the list of residue number for each atom (with length equal to atom count)
    
    Outputs:
    residue_position -- a numpy 2D array with cartesian coordinates for each atom in the residue
    residue_charge -- a numpy 1D array with charge for each atom in the residue
    """
    filter = resnums == residue
    return positions[filter], charges[filter]


def split_guanines_in_three_string(resnums,atoms):
    """
    Modify the list of residue number into a list of strings, and decorate the residue string for guanine parts.

    Keyword arguments:
    resnums -- a list of residue number for each atom (with length equal to atom count)
    atoms -- a list of atom names from the topology file 

    Outputs:
    resnums -- a list of residue string for each atom, with decoration for guanine atoms
    """
    for index in range(len(atoms)):
        if "'" in atoms[index]:
            resnums[index] = str(resnums[index]) + "sugar"
        elif "P" in atoms[index]:
            resnums[index] = str(resnums[index]) + "P"
        else:
            resnums[index] = str(resnums[index])
    return resnums

def split_guanines_in_three_numbers(resnums,atoms):
    """
    Decorate the residue string for guanine parts by adding a large amount for sugars, and a larger amount for phosphates.

    Keyword arguments:
    resnums -- a list of residue number for each atom (with length equal to atom count)
    atoms -- a list of atom names from the topology file 

    Outputs:
    resnums -- a list of residue number for each atom, with phosphate and sugars singled out
    """
    shift = 10
    maxres = max(resnums)
    while shift < maxres:
        shift = shift*10
    shift = shift*10
    for index in range(len(atoms)):
        if "'" in atoms[index]:
            resnums[index] += shift
        elif "P" in atoms[index]:
            resnums[index] += 2*shift
    return resnums


def generate_edge_matrix(frame, charges, resnums, molsizes):
    """
    Compute coulombic force between pairs of residues of different systems, for a frame.
    
    Keyword arguments:
    frame -- a numpy 2D array w(th cartesian coordinates for each atom
    charges -- a numpy 1D array with charge for each atom
    resnums -- a numpy 1D array with the list of residue number for each atom (with length equal to atom count)
    molsizes -- list of last atom numbers in each system of the PDB file

    Outputs:
    edge_matrix -- a numpy 2D array with shape(residue amount, residue amount) containing electrostatic interaction values between pairs of residues 
    """
    unique_residues = list(dict.fromkeys(resnums))
    edge_matrix = np.zeros((len(unique_residues),len(unique_residues)))
    resnums_array = np.array(resnums)
    for i in range(len(unique_residues)):
        for j in range(i+1,len(unique_residues)):
            if not same_system(resnums.index(unique_residues[i]),resnums.index(unique_residues[j]),molsizes):
                #print(f"{i}, {j}, {np.count_nonzero(resnums_array[resnums_array == unique_residues[i]]) }")
                res_pos_1, res_charges_1 = filter_residue_atoms(unique_residues[i],frame,charges,resnums_array)
                res_pos_2, res_charges_2 = filter_residue_atoms(unique_residues[j],frame,charges,resnums_array) 
                edge_matrix[i][j] = force.compute_residue_energy(res_pos_1,res_pos_2, res_charges_1, res_charges_2)
                edge_matrix[j][i] = edge_matrix[i][j]
    return edge_matrix

def check_file_correctness(positions, resnums, molsizes, charges):
    """Check the variables produced by file reading, and raises explicit expressions based on selected issues."""
    if not isinstance(positions,np.ndarray) or not positions.ndim == 3:
        raise ValueError("The PDB file is not readable as a list of frames")
    if not isinstance(charges,np.ndarray) or not charges.ndim == 1:
        raise ValueError("The topology file is not readable as a list of charges")
    if type(resnums) != list or len(resnums) != len(positions[0]):
        raise ValueError("The residue list is not correct")
    if type(molsizes) != list:
        raise ValueError("The list of molecules in the system is not correct")
    if len(positions[0]) != len(charges):
        raise ValueError("There are more atoms in one file than the other")

def main(path_top='data/1kx5-b_sol_1prot.top' ,path_pdb='data/center_1kx5-b_1prot.pdb', split_guanines = False, debug = False):
    """Run all of the code, placeholder function."""
    try:
        print('Loading atom positions')
        positions, resnums, molsizes = extract.get_coordinates_from_pdb(path_pdb)
        print("Positions loaded")
        print("Loading charges")
        charges, atoms = extract.get_charges_from_top(path_top)
        print("Charges loaded")
    except FileNotFoundError:
        raise FileNotFoundError("Error in the file path(s) given")
    check_file_correctness(positions,resnums,molsizes,charges)
    if split_guanines:
        resnums = split_guanines_in_three_numbers(resnums,atoms)
    full_edges = []
    frame_num = 1
    for frame in positions[:10]:
        full_edges.append(generate_edge_matrix(frame,charges,resnums,molsizes))
        print(f"frame {frame_num} computed")
        frame_num += 1
    if debug:
        return [np.array(full_edges), positions, resnums, molsizes, charges, atoms]
    else:
        return np.array(full_edges)

a = main()
print(a)
