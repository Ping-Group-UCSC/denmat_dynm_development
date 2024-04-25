A detailed step-by-step pipline instruction is here:
https://github.com/Ping-Group-UCSC/Tutorials/blob/master/DMD-workflows/GaAs-DMD-example.md


Note:
Except (i) DMD sources files (ii) FeynWann source files and (iii) Silicon example are new, 
all others are from older version 4.4.3.
They should be updated, e.g., an example of monolayer WSe2 using QE2JDFTx scripts and with e-imp scattering needs to be added.

Folders and files:
1. executables are in ./bin (github does not want an empty folder, so please create this folder)
2. src_v4.5.7 : source codes
3. src_FeynWann_forDMD:
        source files for the modified initialization program - lindbladInit_for-DMD-4.5.6
        there is a compilation script "make-FeynWann-Kairay.sh" for installation in Kairay cluster
4. Examples:
        Silicon_T1-Rate-formula.tgz
        Silicene_Real-Time_DMD.tgz
        RealTime_Example_GaAs_noPhononVscloc

to run:
1. Initialization
   After finish JDFTx electron, phonon and wannier calculations,
   run initialization using the modified initialization code - lindbladInit_for-DMD-4.5.6/init_for-DMD
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
1. Make sure intel mpi compiler exists and command "mpiicpc" exists
2. GSL and MKL must be installed and ensure MKLROOT is correct
3. modify GRL_DIR in make.inc and SRC_DIRS in Makefile, if necessary
4. If there is no "bin" folder, create it
5. type "make"
