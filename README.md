# Projet_numerique

usage: Compute ionisation of guanines [-h] [--topology TOPOLOGY] [--pdb PDB] [--frame_begin FRAME_BEGIN] [--frame_end FRAME_END] [--force]
                                      [--split] [-v] [-d]

options:
  -h, --help            show this help message and exit
  
  --topology TOPOLOGY   path to a top file containing charge data
  
  --pdb PDB             path to a pdb file containing geometry data, on multiple frames
  
  --frame_begin FRAME_BEGIN
  
                        number of the first frame of the frame interval to consider in computations (starting at 1)
                        
  --frame_end FRAME_END
  
                        number of the first frame of the frame interval to consider in computations
                        
  --force               Compute electrostatic force instead of energy
  
  --split               Consider guanines as three entities : phosphate, pentose, base
  
  -v, --verbose         Verbose mode
  
  -d, --debug           Print arguments
  
