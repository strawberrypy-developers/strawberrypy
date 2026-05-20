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
   IMPLICIT NONE

   ! ========= Argument Declaration =========
   INTEGER, intent(in) :: comm, N, NB, nev
   INTEGER, intent(in) ::   nprow, npcol
   COMPLEX*16, intent(in) :: A_local(:, :)
   DOUBLE PRECISION, intent(inout) :: W(N)
   COMPLEX*16, intent(inout) :: Z_local(:, :)
   INTEGER, intent(out) :: INFO

   INTEGER :: local_rows, local_cols
   INTEGER :: desc_A(9), desc_Z(9)
   INTEGER, EXTERNAL :: numroc

   INTEGER:: ctxt, myprow, mypcol
   INTEGER :: lwork, lrwork
   COMPLEX*16, ALLOCATABLE :: work(:)
   DOUBLE PRECISION, ALLOCATABLE :: rwork(:)

   ! ============================================================================
   ! 1. INITIALIZATION
   ! ============================================================================
   call blacs_get(0, 0, ctxt)
   call blacs_gridinit(ctxt, 'Row-major', nprow, npcol)
   call blacs_gridinfo(ctxt, nprow, npcol, myprow, mypcol)

   ! Calculate local matrix sizes
   local_rows = numroc(N, NB, myprow, 0, nprow)
   local_cols = numroc(N, NB, mypcol, 0, npcol)

   call descinit(desc_A, N, N, NB, NB, 0, 0, ctxt, max(1, local_rows), INFO)
   if (INFO /= 0) then
      print *, "Error in descinit for A, info=", INFO
      return
   end if
   call descinit(desc_Z, N, N, NB, NB, 0, 0, ctxt, max(1, local_rows), INFO)
   if (INFO /= 0) then
      print *, "Error in descinit for Z, info=", INFO
      return
   end if

   ! A first call with LWORK=-1 and LRWORK=-1 makes PZHEEV calculate the optimal sizes for the WORK and RWORK arrays.
   lwork = -1
   lrwork = -1
   allocate (work(1), rwork(1))

   CALL PZHEEV('V', 'U', N, A_local, 1, 1, desc_A, W, Z_local, 1, 1, desc_Z, &
               WORK, LWORK, RWORK, LRWORK, INFO)

   if (INFO /= 0) then
      print *, "Error in workspace query for PZHEEV, info=", INFO
      return
   end if

   ! Extract the optimal sizes from the dummy variables
   LWORK = INT(WORK(1))
   LRWORK = INT(RWORK(1))
   DEALLOCATE (WORK, RWORK)

   ALLOCATE (WORK(LWORK))
   ALLOCATE (RWORK(LRWORK))

   !  Call PZHEEV to perform the actual diagonalization
   !  JOBZ='V': Compute both eigenvalues (W) and eigenvectors (Z)
   !  UPLO='U': The upper triangular part of the Hermitian matrix A is stored
   CALL PZHEEV('V', 'U', N, A_local, 1, 1, desc_A, W, Z_local, 1, 1, desc_Z, &
               WORK, LWORK, RWORK, LRWORK, INFO)
   if (INFO /= 0) then
      print *, "Error in PZHEEV diagonalization, info=", INFO
      return
   end if

   ! Deallocate workspace arrays
   DEALLOCATE (WORK)
   DEALLOCATE (RWORK)

   ! ============================================================================
   !  CLEANUP
   ! ============================================================================
   call blacs_gridexit(ctxt)

end SUBROUTINE diagonalize
