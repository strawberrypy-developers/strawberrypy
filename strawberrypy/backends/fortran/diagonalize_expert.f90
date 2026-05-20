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
   DOUBLE PRECISION, EXTERNAL :: pdlamch

   INTEGER:: ctxt, myprow, mypcol
   INTEGER :: lwork, lrwork, liwork
   COMPLEX*16, ALLOCATABLE :: work(:)
   DOUBLE PRECISION, ALLOCATABLE :: rwork(:), gap(:)
   INTEGER, ALLOCATABLE :: iwork(:), ifail(:), iclustr(:)

   DOUBLE PRECISION:: abstol, vl, vu
   INTEGER :: il, iu, m_out, nz
   DOUBLE PRECISION:: orfac

   ! ============================================================================
   ! 1. INITIALIZATION
   ! ============================================================================
   call blacs_get(0, 0, ctxt)
   call blacs_gridinit(ctxt, 'Row-major', nprow, npcol)
   call blacs_gridinfo(ctxt, nprow, npcol, myprow, mypcol)

   ! Calculate local matrix sizes
   local_rows = numroc(N, NB, myprow, 0, nprow)
   local_cols = numroc(N, NB, mypcol, 0, npcol)

   CALL descinit(desc_A, N, N, NB, NB, 0, 0, ctxt, max(1, local_rows), INFO)
   if (info /= 0) then
      print *, "Error in descinit for A, info=", info
      return
   end if
   CALL descinit(desc_Z, N, N, NB, NB, 0, 0, ctxt, max(1, local_rows), INFO)
   if (info /= 0) then
      print *, "Error in descinit for Z, info=", info
      return
   end if

   ! A first call with LWORK=-1 and LRWORK=-1 makes PZHEEV calculate the optimal sizes for the WORK and RWORK arrays.
   allocate (ifail(N))
   allocate (iclustr(2*nprow*npcol))
   allocate (gap(nprow*npcol))

   allocate (work(1), rwork(1), iwork(1))

   IL = 1
   IU = nev

   abstol = 2.0d0 * pdlamch(ctxt, 'U') 
   orfac = -1.0d0

   call pzheevx('V', 'I', 'U', N, a_local, 1, 1, desc_A, &
                vl, vu, IL, IU, ABSTOL, &
                m_out, nz, w, orfac, &
                z_local, 1, 1, desc_Z, &
                work, -1, rwork, -1, iwork, -1, &
                ifail, iclustr, gap, info)

   if (info /= 0) then
      print *, "Error in workspace query for PZHEEVX, info=", info
      return
   end if

   ! Extract the optimal sizes from the dummy variables
   LWORK = INT(WORK(1))
   LRWORK = INT(RWORK(1))
   LIWORK = INT(IWORK(1))

   DEALLOCATE (WORK, RWORK, IWORK)
   allocate (work(lwork), rwork(lrwork), iwork(liwork))

   !  Call PZHEEVX to perform the actual diagonalization
   call pzheevx('V', 'I', 'U', N, a_local, 1, 1, desc_A, &
                vl, vu, IL, IU, ABSTOL, &
                m_out, nz, w, orfac, &
                z_local, 1, 1, desc_Z, &
                work, lwork, rwork, lrwork, iwork, liwork, &
                ifail, iclustr, gap, info)

   if (info /= 0) then
      print *, "Error in PZHEEVX, info=", info
      return
   end if

   ! Deallocate workspace arrays
   deallocate (work, rwork, iwork)

   ! ============================================================================
   !  CLEANUP
   ! ============================================================================
   call blacs_gridexit(ctxt)

end SUBROUTINE diagonalize
