#include "dofs.hpp"

DofInfo::DofInfo(ParFiniteElementSpace *fes_): 
    pmesh(fes_->GetParMesh()), fes(fes_), dim(fes_->GetParMesh()->Dimension()), nDofs(fes->GetNDofs()), normals(fes->GetNBE(), dim)
{
    ParBilinearForm M(fes);
    M.AddDomainIntegrator(new MassIntegrator());
    M.Assemble();
    M.Finalize(0);
    massmatrix_ld = M.SpMat();
    HYPRE_BigInt *cmap = NULL;

    //ParBilinearForm dummy(fes);
    hpr_mass = M.ParallelAssemble(&massmatrix_ld);
    hpr_mass->GetDiag(M_diag);
    hpr_mass->GetOffd(M_offdiag, cmap);

    I_ld = massmatrix_ld.GetI();
    J_ld = massmatrix_ld.GetJ();
    
    /*/
    for(int i = 0; i < massmatrix_ld.Height(); i++)
    {
        for(int k = I_ld[i]; k < I_ld[i+1]; k++)
        {
            int j = J_ld[k];

            MFEM_VERIFY(abs(massmatrix_ld(i,j) - M_diag(i,j)) < 1e-15, "werte nicht gleich i,j" )
            MFEM_VERIFY(abs(M_diag.GetData()[k] -M_diag(i,j)) < 1e-15, "werte nicht gleich k,ij_ld" )
            MFEM_VERIFY(abs(massmatrix_ld.GetData()[k] -massmatrix_ld(i,j)) < 1e-15, "werte nicht gleich k,ij_ld" )
        } 

        for(int k = I_ld[i]; k < I_ld[i+1]; k++)
        {
            int j = J_ld[k];

            MFEM_VERIFY(abs(massmatrix_ld.GetData()[k] - M_diag.GetData()[k]) < 1e-15, "werte nicht gleich k" )
            MFEM_VERIFY(abs(massmatrix_ld(i,j) - M_diag.GetData()[k]) < 1e-15, "werte nicht gleich ij k" )
        } 
    }
    //MFEM_ABORT("passt");
    cout << " passt " << endl;
    //*/

    BuildNormals();
}

DofInfo::~DofInfo() 
{   }

void DofInfo::BuildNormals()
{
    // build normals
    FaceElementTransformations *tr;
    Vector nor(dim);
    for(int b = 0; b < fes->GetNBE(); b++)
    {
        tr = pmesh->GetBdrFaceTransformations(b);
        MFEM_VERIFY(tr->Elem2No < 0, "no boundary element");

        const IntegrationRule *ir = &IntRules.Get(tr->GetGeometryType(), 1);

        const IntegrationPoint &dummy = ir->IntPoint(0);
        tr->SetAllIntPoints(&dummy);
        nor = 0.0;
        if (dim == 1)
        {   
            IntegrationPoint aux_ip;
            tr->Loc1.Transform(dummy, aux_ip);
            nor(0) = 2.0 * aux_ip.x - 1.0;
        }
        else
        {
            CalcOrtho(tr->Jacobian(), nor);
            nor /= nor.Norml2();
        }

        for(int d = 0; d < dim; d++)
        {
            normals(b, d) = nor(d);
        }
    }
}

double DofInfo::GetElem(const int i, const int j, const SparseMatrix& mat) const
{
    auto I = mat.GetI();
    auto J = mat.GetJ();
    auto A = mat.ReadData();

    for(int k = I[i]; k < I[i+1]; k++)
    {
        if(J[k] == j)
        {
            return A[k];
        }
    }

    return 0.0;
}

HYPRE_Int DofInfo::Extract_offdiagonals(hypre_ParCSRMatrix *A, hypre_ParVector *x, hypre_ParVector *y)
{
    hypre_ParCSRCommPkg *comm_pkg = hypre_ParCSRMatrixCommPkg(A);
    hypre_CSRMatrix *offd = hypre_ParCSRMatrixOffd(A);
    hypre_Vector *x_local = hypre_ParVectorLocalVector(x);
    hypre_Vector *y_local = hypre_ParVectorLocalVector(y);
    hypre_Vector *x_tmp;
    HYPRE_BigInt num_cols = hypre_ParCSRMatrixGlobalNumCols(A);
    HYPRE_BigInt x_size = hypre_ParVectorGlobalSize(x);
    HYPRE_Int num_cols_offd = hypre_CSRMatrixNumCols(offd);
    HYPRE_Int num_recvs, num_sends;
    HYPRE_Int ierr = 0;
    HYPRE_Int i;
    HYPRE_Int idxstride = hypre_VectorIndexStride(x_local);
    HYPRE_Int num_vectors = hypre_VectorNumVectors(x_local);
    HYPRE_Complex *x_local_data = hypre_VectorData(x_local);
    HYPRE_Complex *x_tmp_data;
    HYPRE_Complex *x_buf_data;
    HYPRE_ANNOTATE_FUNC_BEGIN;
    /*---------------------------------------------------------------------
     *  Check for size compatibility.  ParMatvec returns ierr = 11 if
     *  length of X doesn't equal the number of columns of A,
     *  ierr = 12 if the length of Y doesn't equal the number of rows
     *  of A, and ierr = 13 if both are true.
     *
     *  Because temporary vectors are often used in ParMatvec, none of
     *  these conditions terminates processing, and the ierr flag
     *  is informational only.
     *--------------------------------------------------------------------*/
    hypre_assert(idxstride > 0);
    if (num_cols != x_size)
    {
        ierr = 11;
    }
    if (num_cols != x_size)
    {
        ierr = 13;
    }
    hypre_assert(hypre_VectorNumVectors(y_local) == num_vectors);
    if (num_vectors == 1)
    {
        x_tmp = hypre_SeqVectorCreate(num_cols_offd);
    }
    else
    {
        hypre_assert(num_vectors > 1);
        x_tmp = hypre_SeqMultiVectorCreate(num_cols_offd, num_vectors);
        hypre_VectorMultiVecStorageMethod(x_tmp) = 1;
    }
    /*---------------------------------------------------------------------
     * If there exists no CommPkg for A, a CommPkg is generated using
     * equally load balanced partitionings
     *--------------------------------------------------------------------*/
    if (!comm_pkg)
    {
        hypre_MatvecCommPkgCreate(A);
        comm_pkg = hypre_ParCSRMatrixCommPkg(A);
    }
    /* Update send_map_starts, send_map_elmts, and recv_vec_starts when doing
       sparse matrix/multivector product  */
    hypre_ParCSRCommPkgUpdateVecStarts(comm_pkg, num_vectors, hypre_VectorVectorStride(hypre_ParVectorLocalVector(x)), hypre_VectorIndexStride(hypre_ParVectorLocalVector(x)));
    num_recvs = hypre_ParCSRCommPkgNumRecvs(comm_pkg);
    num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
    hypre_assert(num_cols_offd * num_vectors ==
                 hypre_ParCSRCommPkgRecvVecStart(comm_pkg, num_recvs));
    hypre_assert(hypre_ParCSRCommPkgRecvVecStart(comm_pkg, 0) == 0);
    hypre_assert(hypre_ParCSRCommPkgSendMapStart(comm_pkg, 0) == 0);
#ifdef HYPRE_PROFILE
    hypre_profile_times[HYPRE_TIMER_ID_PACK_UNPACK] -= hypre_MPI_Wtime();
#endif
#if defined(HYPRE_USING_PERSISTENT_COMM)
    hypre_ParCSRPersistentCommHandle *persistent_comm_handle =
        hypre_ParCSRCommPkgGetPersistentCommHandle(1, comm_pkg);
#else
    hypre_ParCSRCommHandle *comm_handle;
#endif
    /*---------------------------------------------------------------------
     * Allocate (during hypre_SeqVectorInitialize_v2) or retrieve
     * persistent receive data buffer for x_tmp (if persistent is enabled).
     *--------------------------------------------------------------------*/
#if defined(HYPRE_USING_PERSISTENT_COMM)
    hypre_VectorData(x_tmp) = (HYPRE_Complex *)
        hypre_ParCSRCommHandleRecvDataBuffer(persistent_comm_handle);
    hypre_SeqVectorSetDataOwner(x_tmp, 0);
#endif
    hypre_SeqVectorInitialize_v2(x_tmp, HYPRE_MEMORY_HOST);
    x_tmp_data = hypre_VectorData(x_tmp);
    /*---------------------------------------------------------------------
     * Allocate data send buffer
     *--------------------------------------------------------------------*/
#if defined(HYPRE_USING_PERSISTENT_COMM)
    x_buf_data = (HYPRE_Complex *)hypre_ParCSRCommHandleSendDataBuffer(persistent_comm_handle);
#else
    x_buf_data = hypre_TAlloc(HYPRE_Complex,
                              hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
                              HYPRE_MEMORY_HOST);
#endif
    /* The assert is because this code has been tested for column-wise vector storage only. */
    hypre_assert(idxstride == 1);
    /*---------------------------------------------------------------------
     * Pack send data
     *--------------------------------------------------------------------*/
#if defined(HYPRE_USING_OPENMP)
#pragma omp parallel for HYPRE_SMP_SCHEDULE
#endif
    for (i = hypre_ParCSRCommPkgSendMapStart(comm_pkg, 0);
         i < hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);
         i++)
    {
        x_buf_data[i] = x_local_data[hypre_ParCSRCommPkgSendMapElmt(comm_pkg, i)];
    }
#ifdef HYPRE_PROFILE
    hypre_profile_times[HYPRE_TIMER_ID_PACK_UNPACK] += hypre_MPI_Wtime();
    hypre_profile_times[HYPRE_TIMER_ID_HALO_EXCHANGE] -= hypre_MPI_Wtime();
#endif
    /* Non-blocking communication starts */
#ifdef HYPRE_USING_PERSISTENT_COMM
    hypre_ParCSRPersistentCommHandleStart(persistent_comm_handle,
                                          HYPRE_MEMORY_HOST, x_buf_data);
#else
    comm_handle = hypre_ParCSRCommHandleCreate_v2(1, comm_pkg,
                                                  HYPRE_MEMORY_HOST, x_buf_data,
                                                  HYPRE_MEMORY_HOST, x_tmp_data);
#endif
#ifdef HYPRE_PROFILE
    hypre_profile_times[HYPRE_TIMER_ID_HALO_EXCHANGE] += hypre_MPI_Wtime();
#endif
#ifdef HYPRE_PROFILE
    hypre_profile_times[HYPRE_TIMER_ID_HALO_EXCHANGE] -= hypre_MPI_Wtime();
#endif
    /* Non-blocking communication ends */
#ifdef HYPRE_USING_PERSISTENT_COMM
    hypre_ParCSRPersistentCommHandleWait(persistent_comm_handle, HYPRE_MEMORY_HOST, x_tmp_data);
#else
    hypre_ParCSRCommHandleDestroy(comm_handle);
#endif
#ifdef HYPRE_PROFILE
    hypre_profile_times[HYPRE_TIMER_ID_HALO_EXCHANGE] += hypre_MPI_Wtime();
#endif
    /* copy offd part */
    hypre_SeqVectorCopy(x_tmp, y_local);
    /*---------------------------------------------------------------------
     * Free memory
     *--------------------------------------------------------------------*/
    hypre_SeqVectorDestroy(x_tmp);
#if !defined(HYPRE_USING_PERSISTENT_COMM)
    hypre_TFree(x_buf_data, HYPRE_MEMORY_HOST);
#endif
    HYPRE_ANNOTATE_FUNC_END;
    return ierr;
}
void DofInfo::Extract_offd_hypre(mfem::HypreParMatrix *const mat, const mfem::Vector &x, mfem::Vector &y, const int offd_width)
{
    y.SetSize(offd_width);
    mfem::HypreParVector X(mat->GetComm(), mat->GetGlobalNumRows(), NULL, mat->ColPart(), false);
    mfem::HypreParVector Y(mat->GetComm(), mat->GetGlobalNumRows(), NULL, mat->ColPart(), false);
    Y.SetSize(offd_width);
    hypre_ParVectorLocalSize((hypre_ParVector *)Y) = offd_width;
    X.WrapMemoryRead(x.GetMemory());
    Y.WrapMemoryWrite(y.GetMemory());
    Extract_offdiagonals((hypre_ParCSRMatrix *)*mat, (hypre_ParVector *)X, (hypre_ParVector *)Y);
}


