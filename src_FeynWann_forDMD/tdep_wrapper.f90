! Copyright 2022 Josh Quinton, Ravishankar Sundararaman
! 
! This file is part of JDFTx.
! 
! JDFTx is free software: you can redistribute it and/or modify
! it under the terms of the GNU General Public License as published by
! the Free Software Foundation, either version 3 of the License, or
! (at your option) any later version.
! 
! JDFTx is distributed in the hope that it will be useful,
! but WITHOUT ANY WARRANTY; without even the implied warranty of
! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
! GNU General Public License for more details.
! 
! You should have received a copy of the GNU General Public License
! along with JDFTx.  If not, see <http://www.gnu.org/licenses/>.


! Contains variables created during tdep_initialize and needed for tdep_compute
module globals
  use type_crystalstructure, only: lo_crystalstructure
  use type_forceconstant_secondorder, only: lo_forceconstant_secondorder
  use lo_memtracker, only: lo_mem_helper
  use type_qpointmesh, only: lo_qpoint
  use mpi_wrappers, only: lo_mpi_helper, lo_stop_gracefully
  use lo_phonon_bandstructure_on_path, only: lo_phonon_bandstructure
  use type_phonon_dispersions, only: lo_phonon_dispersions_qpoint
  use konstanter, only: r8

  implicit none

  type(lo_crystalstructure) :: uc
  type(lo_mem_helper) :: mem
  type(lo_forceconstant_secondorder) :: fc
  type(lo_mpi_helper) :: mw
  type(lo_phonon_dispersions_qpoint) :: p
end module


! Read unit cell and force constants. Set verbosity > 0 only on head process.
subroutine tdep_initialize(verbosity)
  use globals
  implicit none
  integer verbosity
  call mem%init()
  call uc%readfromfile('infile.ucposcar', verbosity=verbosity)
  call fc%readfromfile(uc, 'infile.forceconstant', mem, verbosity=verbosity)
end subroutine


! Compute frequencies omega and eigenvectors U for Cartesian wavevector qcart.
subroutine tdep_compute(qcart, omega, U)
  use globals
  implicit none
  real(r8) qcart(3)
  real(r8) omega(*)
  complex(r8) U(*)
  integer N
  call p%generate(fc, uc, mem, qvec=qcart)
  N = uc%na * 3
  call dcopy(N, p%omega, 1, omega, 1)
  call zcopy(N*N, p%egv, 1, U, 1)
end subroutine
