! N: Size of the coefficient matrix A taken into account slicing
! nrhs: number of columns of the solution matrix B taken into account slicing
! nb: block size
! a_local: Each process's local piece of the corresponding global matrix A
! gr_A : row length of the original global square matrix A without any operations on it and without slicing.
! b_local: Each process's local piece of the corresponding global matrix B
! gr_B, gc_B : row and column length of the original global matrix B without any operations on it and without slicing.
! nprow (npcol): Number of processor rows (columns) in the grid
! IA, JA: global row and column indices of the first element of global matrix A. When the array is sliced, IA, JA should be set to the starting slicing index w.r.t to the original matrix.
! IB, JB: global row and column indices of the first element of global matrix B. When the array is sliced, IB, JB should be set to the starting slicing index w.r.t to the original matrix.
subroutine linsolve(n, nrhs, nb, a_local, gr_A, b_local, gr_B, gc_B, nprow, npcol, IA, JA, IB, JB, info)
   implicit none

   ! --- Input/Output Arguments ---
   ! These are passed from the Python script
   integer, intent(in) :: n, nrhs, nb
   integer, intent(in) :: nprow, npcol
   integer, intent(in) :: gr_A, gr_B, gc_B, IA, JA, IB, JB
   complex*16, intent(inout) :: a_local(:, :) ! Local part of matrix A
   complex*16, intent(inout) :: b_local(:, :) ! Local part of matrix B, overwritten by solution X
   integer, intent(out) :: info

   ! --- ScaLAPACK and BLACS Local Variables ---
   integer :: desc_a(9), desc_b(9) ! Array descriptors
   integer :: ctxt
   integer :: myrow, mycol

   integer, allocatable :: ipiv(:) ! Pivot array

   ! Get the leading dimension of the local arrays from their shape
   integer :: lld_a, lld_b

   lld_a = size(a_local, 1)
   lld_b = size(b_local, 1)

   allocate (ipiv(lld_a + nb))

   ! ----------------------------------------------------------------------
   !  Initialize BLACS process grid
   ! ----------------------------------------------------------------------
   CALL blacs_get(0, 0, ctxt)
   CALL blacs_gridinit(ctxt, 'Row-major', nprow, npcol)
   CALL blacs_gridinfo(ctxt, nprow, npcol, myrow, mycol)

   ! ----------------------------------------------------------------------
   !  Create ScaLAPACK Array Descriptors
   ! ----------------------------------------------------------------------
   CALL descinit(desc_a, gr_A, gr_A, NB, NB, 0, 0, ctxt, max(1, lld_a), INFO)
   if (info /= 0) then
      print *, "Error in descinit for A, info=", info
      return
   end if
   CALL descinit(desc_b, gr_B, gc_B, NB, NB, 0, 0, ctxt, max(1, lld_b), INFO)
   if (info /= 0) then
      print *, "Error in descinit for B, info=", info
      return
   end if

   ! ============================================================================
   !  Use Linear Solver  (AX=B, solve for X given A and B)
   ! ============================================================================
   CALL pzgesv(n, nrhs, a_local, IA, JA, desc_a, ipiv, b_local, IB, JB, desc_b, info)

   if (info /= 0) then
      print *, "Error in pzgesv, info=", info
      return
   end if

   ! ----------------------------------------------------------------------
   !  Clean up BLACS
   ! ----------------------------------------------------------------------
   CALL blacs_gridexit(ctxt)

end subroutine linsolve
