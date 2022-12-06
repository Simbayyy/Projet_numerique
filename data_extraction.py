import numpy as np
import os, pathlib, re
import pandas

os.chdir(pathlib.Path(__file__).parent.resolve())


def get_coordinates_from_pdb(file_path):
    """
    Takes in the path for a pdb file and returns a 3D numpy array

    Keyword arguments:
    file_path -- a path to a .pdb file

    Outputs:
    frames -- a numpy 3D array, containing n frames, with each frame containing a 2D array with each atom's cartesian coordinates 
    """
    with open(file_path) as fp:
        data = fp.readlines()
    frames = [[]]
    for line in data:
        try:
            if 'ENDMDL' in line:
                frames.append([])
            elif 'ATOM' in line:
                frames[-1].append([float(line[30:38]),float(line[38:46]),float(line[46:54])])
        except IndexError:
            print(line)
    return np.array(frames[:-1])

def get_charges_from_top(file_path):
    """
    Takes in the path for a topology file and returns a 2D numpy array

    Keyword arguments:
    file_path -- a path to a .pdb file

    Outputs:
    frames -- a numpy 2D array, containing charges for each atom in the simulation
    """
    with open(file_path) as fp:
        data = fp.readlines()
    atoms = False
    molecules = False

    lastindex = 0
    charges = [[]]
    molecule_amounts = []

    for line in data:
        if re.match(r'^\[',line):
            atoms = False
            molecules = False
        if re.match(r'\[ atoms \]', line):
            atoms = True
        if re.match(r'\[ molecules \]', line):
            molecules = True

        if atoms:
            if re.match(r'^ *\d+',line):
                if chargematch := re.match(r'.{45}(-?\d+.\d+)',line):
                    if lastindexmatch := re.match(r' *(\d+) +',line):
                        if int(lastindexmatch.group(1)) <= lastindex:
                            charges.append([])
                        lastindex = int(lastindexmatch.group(1))
                    charges[-1].append(float(chargematch.group(1)))

        if molecules:
            if name_amount := re.match(r'^([^;][\w\d\+]*) +(\d+)',line):
                molecule_amounts.append([name_amount.group(1),int(name_amount.group(2))])

    if len(charges) != len(molecule_amounts):
        raise IndexError('More molecules defined than atom groups in the file')

    big_charge_list = []
    for molecule_num in range(len(molecule_amounts)):
        for _ in range(molecule_amounts[molecule_num][1]):
            big_charge_list.extend(charges[molecule_num])

    big_charge_array = np.array(big_charge_list)
    return big_charge_array
