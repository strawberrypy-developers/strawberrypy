! M: row length of the final global matrix C
! K: length of the inner dimension of one of the global matrix, taken into account the operations.
! N: column length of the final global matrix C
! NB: block size of the  block-cyclic distribution of data into the processor grid
! nprow: Number of process rows in the grid
! npcol: Number of process columns in the grid
! alpha, beta: scalar operations. See the description on pzgemm below
! A_loc, B_loc, C_loc: Each process's local piece of the corresponding global matrices
! gr_A, gc_A : row and column length of the original global matrix A without any operations on it and without slicing.
! gr_B, gc_B : row and column length of the original global matrix B without any operations on it and without slicing.
! op_A, op_B: operation on local matrix A and B. Can be 'N' (do nothing), 'T' (transpose), 'C' (conjugate transpose)
! IA, JA: global row and column indices of the first element of global matrix A without any operations on it. When the array is sliced, IA, JA should be set to the starting slicing index w.r.t to the original matrix.
! IB, JB: global row and column indices of the first element of global matrix B without any operations on it. When the array is sliced, IB, JB should be set to the starting slicing index w.r.t to the original matrix.

!! Note: M,K,N should be adjusted accordingly when doing slicing on the original matrices. They should refer to the new dimensions of the sliced matrices taken into account the operations on these sliced matrices.
!! Slicing: A(:,:) => A(21:,50:) => IA=21, JA=50

subroutine matmul(M, K, N, NB, nprow, npcol, alpha, beta, A_loc, B_loc, &
                  C_loc, gr_A, gc_A, gr_B, gc_B, op_A, op_B, &
                  IA, JA, IB, JB, info)
   implicit none

   ! Scalar arguments
   integer, intent(in) :: M, N, K, NB, nprow, npcol
   integer, intent(in) :: gr_A, gc_A, gr_B, gc_B
   integer, intent(in) :: IA, JA, IB, JB
   complex*16, intent(in) :: alpha, beta
   character, intent(in) :: op_A, op_B
   integer, intent(out) :: info

   ! Local data arrays passed from Python
   complex*16, intent(in) :: A_loc(:, :)
   complex*16, intent(in) :: B_loc(:, :)
   complex*16, intent(inout) :: C_loc(:, :)

   ! BLACS and ScaLAPACK variables
   integer :: myrow, mycol
   integer :: ctxt, descA(9), descB(9), descC(9)

   ! Get the leading dimension of the local arrays from their shape
   integer :: lld_a, lld_b, lld_c

   lld_a = size(A_loc, 1)
   lld_b = size(B_loc, 1)
   lld_c = size(C_loc, 1)

   ! ----------------------------------------------------------------------
   ! 1. Initialize BLACS process grid
   ! ----------------------------------------------------------------------
   ! Get a system context handle
   CALL blacs_get(0, 0, ctxt)
   ! Initialize an nprow x npcol grid
   CALL blacs_gridinit(ctxt, 'Row-major', nprow, npcol)
   ! Get my own coordinates (myrow, mycol) in the grid
   CALL blacs_gridinfo(ctxt, nprow, npcol, myrow, mycol)

   ! ----------------------------------------------------------------------
   ! 2. Create ScaLAPACK Array Descriptors
   ! ----------------------------------------------------------------------
   ! We need to calculate the number of rows/cols this process owns for each matrix
   CALL descinit(descA, gr_A, gc_A, NB, NB, 0, 0, ctxt, lld_a, info)
   if (info /= 0) then
      print *, "Error in descinit for A, info=", info
      return
   end if
   CALL descinit(descB, gr_B, gc_B, NB, NB, 0, 0, ctxt, lld_b, info)
   if (info /= 0) then
      print *, "Error in descinit for B, info=", info
      return
   end if
   CALL descinit(descC, M, N, NB, NB, 0, 0, ctxt, lld_c, info)
   if (info /= 0) then
      print *, "Error in descinit for C, info=", info
      return
   end if

   ! ----------------------------------------------------------------------
   ! 3. Call PZGEMM
   ! C(M,N) = alpha * op(A)[M,K] * op(B)[K,N] + beta * C[M,N]
   ! ----------------------------------------------------------------------
   CALL pzgemm(op_A, op_B, M, N, K, alpha, &
               A_loc, IA, JA, descA, &
               B_loc, IB, JB, descB, beta, &
               C_loc, 1, 1, descC)

   ! ----------------------------------------------------------------------
   ! 4. Clean up BLACS
   ! ----------------------------------------------------------------------
   CALL blacs_gridexit(ctxt)
   ! call blacs_exit(1)

end subroutine matmul
