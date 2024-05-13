project(FeynWann)
cmake_minimum_required(VERSION 2.8.12...3.20)

set(JDFTX_SRC "<full-path-to-jdftx-src>" CACHE STRING "Full path to the jdftx source root")
set(JDFTX_BUILD "<full-path-to-jdftx-build>" CACHE STRING "Full path to the jdftx build directory")
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/CMake-Modules/ ${JDFTX_SRC}/CMake-Modules/)

#Package configuration:
set(CPACK_PACKAGE_NAME "FeynWann")
set(CPACK_PACKAGE_VERSION_MAJOR "1")
set(CPACK_PACKAGE_VERSION_MINOR "0")
set(CPACK_PACKAGE_VERSION_PATCH "alpha")
set(VERSION_STRING "${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH}")

#Get revision information
find_package(Git)
if(GIT_FOUND)
	execute_process(
		COMMAND ${GIT_EXECUTABLE} rev-parse --is-inside-work-tree
		WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
		OUTPUT_VARIABLE IS_GIT
		ERROR_QUIET
		OUTPUT_STRIP_TRAILING_WHITESPACE)
	if("true" STREQUAL "${IS_GIT}")	
		execute_process(
			COMMAND ${GIT_EXECUTABLE} log -1 --format=%h
			WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
			OUTPUT_VARIABLE GIT_HASH
			OUTPUT_STRIP_TRAILING_WHITESPACE)
		message(STATUS "Git revision hash: ${GIT_HASH}")
	endif()
endif()


#Static linking:
if(StaticLinking)
        if(WIN32)
                set(CMAKE_FIND_LIBRARY_SUFFIXES .lib .a ${CMAKE_FIND_LIBRARY_SUFFIXES})
        else()
                set(CMAKE_FIND_LIBRARY_SUFFIXES .a ${CMAKE_FIND_LIBRARY_SUFFIXES})
        endif()
        set(CMAKE_EXE_LINKER_FLAGS -static)
        set(CMAKE_EXE_LINK_DYNAMIC_C_FLAGS)       # remove -Wl,-Bdynamic
        set(CMAKE_EXE_LINK_DYNAMIC_CXX_FLAGS)
        set(CMAKE_SHARED_LIBRARY_C_FLAGS)         # remove -fPIC
        set(CMAKE_SHARED_LIBRARY_CXX_FLAGS)
        set(CMAKE_SHARED_LIBRARY_LINK_C_FLAGS)    # remove -rdynamic
        set(CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS)
endif()

#Whether Profiling is on or off should match the corresponding JDFTx build:
option(EnableProfiling "Enable timers and print profiling output" ON)
if(EnableProfiling)
	add_definitions("-DENABLE_PROFILING")
endif()

#Whether MPI is on or off should match the corresponding JDFTx build:
option(EnableMPI "Use MPI parallelization (in addition to threads / gpu)" ON)
if(EnableMPI)
	find_package(MPI REQUIRED)
	include_directories(${MPI_INCLUDE_PATH})
	add_definitions("-DMPI_ENABLED")
endif()

#MPI I/O options:
option(MPISafeWrite "Safe mode for MPI writes: write from one process; may be needed on some NFS installations")
if(MPISafeWrite)
	add_definitions("-DMPI_SAFE_WRITE")
endif()

find_package(GSL REQUIRED)
include_directories(${GSL_INCLUDE_DIR})

set(FFTW3_MPI_REQUIRED ON)
find_package(FFTW3 REQUIRED)
include_directories(${FFTW3_INCLUDE_DIR})
include_directories(${FFTW3_MPI_INCLUDE_DIR})

option(EnableLibXC "Use LibXC to provide additional exchange-correlation functionals")
if(EnableLibXC)
	add_definitions("-DLIBXC_ENABLED")
endif()

option(EnableTDEP "Use TDEP to replace phonon dispersion relations with finite T versions.")
if(EnableTDEP)
	find_package(TDEP REQUIRED)
	add_definitions("-DTDEP_ENABLED")
	enable_language(Fortran)
	set(TDEP_F90_FILES tdep_wrapper.f90)
	include_directories(${TDEP_INCLUDE_DIR})

	find_package(HDF5 COMPONENTS C Fortran REQUIRED)
	set(TDEP_LIBRARIES ${TDEP_LIBRARIES} ${HDF5_LIBRARIES})
endif()

configure_file(${CMAKE_SOURCE_DIR}/config.in.h ${CMAKE_BINARY_DIR}/config.h)
include_directories("${JDFTX_SRC}")
include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_BINARY_DIR})
add_definitions(-Wall -g -O3 -std=c++0x ${MPI_CXX_COMPILE_FLAGS})

#Link options:
if(StaticLinking)
        set(LINK_TYPE STATIC)
        set(LINK_PREFIX -Wl,--whole-archive)
        set(LINK_SUFFIX -Wl,--no-whole-archive)
else()
        set(LINK_TYPE SHARED)
endif()

#Optional ScaLAPACK:
option(EnableScaLAPACK "Enable ScaLAPACK support (currently used only by the dense diagonalization in lindblad/linear)")
if(EnableScaLAPACK)
	add_definitions("-DSCALAPACK_ENABLED")
	include_directories(/opt/intel/mkl/include ${MKL_PATH}/include) #needed for c function definitions
	enable_language(Fortran)
	set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -fallow-argument-mismatch")
	#set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -fallow-argument-mismatch") #comment out by JX
	file(GLOB SCALAPACK_F_FILES scalapack/*.f)
	#JDFTx must be built with scalapack support for this to work (not tested in this cmake script)
endif()

#Main library for FeynWann containing all the auxiliary functionality
file(GLOB FEYNWANN_SOURCES FeynWann*.cpp)
add_library(FeynWann ${LINK_TYPE} ${FEYNWANN_SOURCES} DistributedMatrix.cpp InputMap.cpp Histogram.cpp Interp1.cpp BlockCyclicMatrix.cpp lindbladInit_for-DMD-4.5.6/help.cpp ${SCALAPACK_F_FILES} ${TDEP_F90_FILES})
find_library(JDFTX_LIBRARY jdftx PATHS ${JDFTX_BUILD} NO_DEFAULT_PATH)
target_link_libraries(FeynWann ${JDFTX_LIBRARY} ${FFTW3_MPI_LIBRARY} ${FFTW3_THREADS_LIBRARY} ${FFTW3_LIBRARY} ${MPI_CXX_LIBRARIES} ${EXTRA_LIBRARIES} ${TDEP_LIBRARIES})

#Executables linked to above library
macro(add_FeynWann_executable targetName)
	add_executable(${targetName} ${targetName}.cpp)
	target_link_libraries(${targetName} ${LINK_PREFIX} FeynWann ${LINK_SUFFIX})
	set_target_properties(${targetName} PROPERTIES LINK_FLAGS "${MPI_CXX_LINK_FLAGS}")
endmacro()

add_FeynWann_executable(phononDOS)
add_FeynWann_executable(electronDOS)
add_FeynWann_executable(electronPhononCoupling)
add_FeynWann_executable(electronPhononLinewidth)
add_FeynWann_executable(electronDefectLinewidth)
add_FeynWann_executable(resistivity)
add_FeynWann_executable(resistivityBTE)
add_FeynWann_executable(ballisticConductance)
add_FeynWann_executable(ImEps)
add_FeynWann_executable(ImEpsNonTherm)
add_FeynWann_executable(circularDichroism)
add_FeynWann_executable(phononElectronLinewidth)
add_FeynWann_executable(ePhRelax)
add_FeynWann_executable(ePhSpinRelax)
add_FeynWann_executable(debugEph)
add_FeynWann_executable(debugDefect)
add_FeynWann_executable(debugE)

add_subdirectory(lindblad)
add_subdirectory(lindbladInit_for-DMD-4.5.6)
