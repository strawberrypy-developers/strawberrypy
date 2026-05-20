! comm: MPI communicator handle
! N : Global size of the matrix (N x N)
! NB: Block size for distribution
! nev: number of eigenvectors to compute (in ascending order of the corresponding eigenvalues)
! NPROW: Number of process rows in the grid
! NPCOL: Number of process columns in the grid
! A_local: Each process's local piece of the global matrix A
! W: Eigenvalues (real). Replicated on all processes.
! Z_local: Local portions of the eigenvector matrix Z.
SUBROUTINE diagonalize(comm, N, NB, nev, nprow, npcol, A_local, W, Z_local, INFO)
   USE ELPA
   IMPLICIT NONE

   ! ========= Argument Declaration =========
   INTEGER, intent(in) :: comm, N, NB, nev
   INTEGER, intent(in) ::   nprow, npcol
   COMPLEX*16, intent(in) :: A_local(:, :)
   DOUBLE PRECISION, intent(inout) :: W(N)
   COMPLEX*16, intent(inout) :: Z_local(:, :)
   INTEGER, intent(out) :: INFO

   INTEGER :: local_rows, local_cols
   INTEGER, EXTERNAL :: numroc

   INTEGER:: ctxt, myprow, mypcol

   class(elpa_t), pointer :: elpaInstance

   ! ============================================================================
   ! 1. INITIALIZATION
   ! ============================================================================
   call blacs_get(0, 0, ctxt)
   call blacs_gridinit(ctxt, 'Row-major', nprow, npcol)
   call blacs_gridinfo(ctxt, nprow, npcol, myprow, mypcol)

   ! Calculate local matrix sizes
   local_rows = numroc(N, NB, myprow, 0, nprow)
   local_cols = numroc(N, NB, mypcol, 0, npcol)

   INFO = elpa_init(20250131)
   elpaInstance => elpa_allocate(info)
   if (info /= ELPA_OK) then
      print *, "Could not allocate ELPA instance, info=", info
      return
   end if

   call elpaInstance%set("na", N, info)
   call elpaInstance%set("nev", nev, info)
   call elpaInstance%set("local_nrows", local_rows, info)
   call elpaInstance%set("local_ncols", local_cols, info)
   call elpaInstance%set("nblk", NB, info)
   call elpaInstance%set("mpi_comm_parent", comm, info)
   call elpaInstance%set("process_row", myprow, info)
   call elpaInstance%set("process_col", mypcol, info)
   info = elpaInstance%setup()

   if (info /= ELPA_OK) then
      print *, "Cannot setup ELPA, info=", info
      return
   end if

   ! ============================================================================
   ! 2. SOLVE THE EIGENVALUE PROBLEM
   ! ============================================================================
   call elpaInstance%set("solver", ELPA_SOLVER_2STAGE, info)
   call elpaInstance%eigenvectors_double_complex(A_local, W, Z_local, info)

   if (info /= ELPA_OK) then
      print *, "Cannot solve eigenvalue problem, info=", info
      return
   end if

   ! ============================================================================
   ! 3. CLEANUP
   ! ============================================================================
   call elpa_deallocate(elpaInstance, info)
   call blacs_gridexit(ctxt)

end SUBROUTINE diagonalize
