#Optional PETSc to enable linear dynamics in lindblad/linear
option(EnablePETSc "Enable lindbladian linear dynamics using PETSc" OFF)
if(EnablePETSc)
	find_package(PETSc REQUIRED)
	include_directories("${PETSC_INCLUDE_DIR}")
	add_definitions("-DPETSC_ENABLED")
endif()

add_FeynWann_executable(init)
add_FeynWann_executable(makeModel)

#Combined linear/nonlinar dynamics executable:
add_executable(run run.cpp Lindblad.cpp LindbladDynamics.cpp LindbladNonlinear.cpp LindbladMatrix.cpp LindbladLinear.cpp LindbladSpectrum.cpp)
target_link_libraries(run ${LINK_PREFIX} FeynWann ${LINK_SUFFIX})
target_link_libraries(run ${GSL_BLAS_LIBRARY} ${PETSC_LIBRARY})
set_target_properties(run PROPERTIES LINK_FLAGS "${MPI_CXX_LINK_FLAGS}")
