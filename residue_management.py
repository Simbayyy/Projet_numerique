import data_extraction as extract
import force_computation as force
import numpy as np
import os, pathlib, re
import pandas

os.chdir(pathlib.Path(__file__).parent.resolve())

def same_system(atom_index_1, atom_index_2, molsizes):
    for molsize in molsizes:
        if atom_index_1 < molsize and atom_index_2 >= molsize:
            return False
        if atom_index_2 < molsize and atom_index_1 >= molsize:
            return False
    return True

def filter_residue_atoms(residue, positions, charges, resnums):
    filter = resnums == residue
    return positions[filter], charges[filter]
    

def generate_edge_matrix(frame, charges, resnums, molsizes):
    """
    Compute coulombic force between pairs of residues of different systems, for a frame
    """
    unique_residues = list(dict.fromkeys(resnums))
    empty_matrix = np.zeros((len(unique_residues),len(unique_residues),3))
    resnums_array = np.array(resnums)
    for i in range(len(unique_residues)):
        for j in range(i+1,len(unique_residues)):
            if not same_system(resnums.index(unique_residues[i]),resnums.index(unique_residues[j]),molsizes):
                #print(f"{i}, {j}, {np.count_nonzero(resnums_array[resnums_array == unique_residues[i]]) }")
                res_pos_1, res_charges_1 = filter_residue_atoms(unique_residues[i],frame,charges,resnums_array)
                res_pos_2, res_charges_2 = filter_residue_atoms(unique_residues[j],frame,charges,resnums_array) 
                empty_matrix[i][j] = force.compute_residue_force(res_pos_1,res_pos_2, res_charges_1, res_charges_2)
                empty_matrix[j][i] = empty_matrix[i][j]
    return empty_matrix

def main(path_top,path_pdb):
    print('Loading atom positions')
    positions, resnums, molsizes = extract.get_coordinates_from_pdb('data/center_1kx5-b_1prot.pdb')
    print("Positions loaded")
    print("Loading charges")
    charges = extract.get_charges_from_top('data/1kx5-b_sol_1prot.top')
    print("Charges loaded")
    full_edges = []
    frame_num = 1
    for frame in positions:
        full_edges.append(generate_edge_matrix(frame,charges,resnums,molsizes))
        print(f"frame {frame_num} computed")
        frame_num += 1
    return np.array(full_edges)

a = main("","")
print(a)
