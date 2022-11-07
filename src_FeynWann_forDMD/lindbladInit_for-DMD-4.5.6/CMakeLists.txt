add_executable(init_for-DMD init.cpp ElectronPhonon_ana.cpp parameters.cpp electron_k_select.cpp electron_save.cpp electron_ana.cpp help_electron.cpp)
target_link_libraries(init_for-DMD ${LINK_PREFIX} FeynWann ${LINK_SUFFIX})
target_link_libraries(init_for-DMD ${GSL_BLAS_LIBRARY} ${PETSC_LIBRARY})
set_target_properties(init_for-DMD PROPERTIES LINK_FLAGS "${MPI_CXX_LINK_FLAGS}")
