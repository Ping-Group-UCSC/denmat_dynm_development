Folders and files:
1. executables are in ./bin (github does not want an empty folder, so please create this folder)
2. src_v4.4 : source codes
3. jdftx_src_for-FW-202004_and_DMD-v4.4 : 
        source files for the modified initialization program - lindbladInit_for-DMD-4.4
        the files should be put in FeynWann folders and the program can be 
        compiled similar to other FeynWann programs
4. RealTime_Example_GaAs_noPhononVscloc : an example with all input files and some output files

to run:
1. Initialization
   After finish JDFTx electron, phonon and wannier calculations,
   run initialization using the modified initialization code - lindbladInit_for-DMD-4.4
   the input of this modified code is similar to Shankar's lindbladInit program
   (see README_initialization_input.txt)
   after finished, all files required by dynamics code are written in folder "ldbd_data"
2. Dynamics
   a folder link to "ldbd_data" mentioned above must exist
   input param.in is needed, see meanings of input parameters in README_input.txt
3. Post-progressing
   In example RealTime_Example_GaAs, you can find
   fit.py : fit time-resolved spin lifetime and total one (zz componet in example)
   run_kerr.sh : Firstly run kerr.py to compute Kerr (also Faraday) rotation 
                         at energies given in input at each time step
                         Secondly run kerrt_atE.py multi times to extract time-dependent 
                         Kerr (Faraday) roration at selected energies

to install:
1. GSL and MKL must be installed and ensure MKLROOT is correct
2. modify GRL_DIR in make.inc and SRC_DIRS in Makefile, if necessary
3. If there is no "bin" folder, create it
4. type "make"