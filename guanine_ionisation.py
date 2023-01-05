import residue_management
import argparse
import os
import numpy as np
import data_extraction as extract
import force_computation as force

def compute_edges(positions,resnums,molsizes,charges,args):
    full_edges = []
    frame_num = 1
    max_frame = len(positions)
    if args.frame_begin < 1 or args.frame_begin > max_frame:
        raise ValueError("Invalid frame interval") 
    if args.frame_end == 0:
        positions_cut = positions[args.frame_begin-1:]
    elif args.frame_end > args.frame_begin and args.frame_end <= max_frame:
        positions_cut = positions[args.frame_begin-1:args.frame_end]
    else:
        raise ValueError("Invalid frame interval")

    for frame in positions_cut:
        if args.force:
            full_edges.append(residue_management.generate_edge_matrix_force(frame,charges,resnums,molsizes))
        else:
            full_edges.append(residue_management.generate_edge_matrix(frame,charges,resnums,molsizes))
        if args.verbose:
            print(f"frame {frame_num} computed")
            frame_num += 1

    return np.array(full_edges)

def main():
    parser = argparse.ArgumentParser(prog="COmpute ionisation of guanines")
    parser.add_argument(
        "--topology",
        type=str,
        default="data/1kx5-b_sol_1prot.top",
        help="path to a top file containing charge data",
    )
    parser.add_argument(
        "--pdb",
        type=str,
        default="data/center_1kx5-b_1prot.pdb",
        help="path to a pdb file containing geometry data, on multiple frames",
    )
    parser.add_argument(
        "--frame_begin",
        type=int,
        default=1,
        help="number of the first frame of the frame interval to consider in computations (starting at 1)",
    )
    parser.add_argument(
        "--frame_end",
        type=int,
        default=0,
        help="number of the first frame of the frame interval to consider in computations",
    )

    parser.add_argument(
        "--force", action="store_const", const=True, help="Compute electrostatic force instead of energy"
    )
    parser.add_argument(
        "--split", action="store_const", const=True, help="Consider guanines as three entities : phosphate, pentose, base"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_const", const=True, help="Verbose mode"
    )
    parser.add_argument(
        "-d", "--debug", action="store_const", const=True, help="Print arguments"
    )

    args = parser.parse_args()

    
    if args.debug:
        print(args)
    if not os.path.exists(args.topology) or not os.path.exists(args.pdb):
        raise FileNotFoundError("Wrong path for files entered")
    try:
        positions, resnums, molsizes = extract.get_coordinates_from_pdb(args.pdb)
        print("Positions loaded")
    except:
        raise FileNotFoundError("Error with the pdb file")
    try:
        charges, atoms = extract.get_charges_from_top(args.topology)
        print("Charges loaded")
    except:
        raise FileNotFoundError("Error with the topology file")
        
    residue_management.check_file_correctness(positions,resnums,molsizes,charges)

    if args.split:
        resnums = residue_management.split_guanines_in_three_numbers(resnums,atoms)



    full_edges = compute_edges(positions,resnums,molsizes,charges,args)

    print(full_edges)

if __name__ == "__main__":
    main()