#include "dofs.hpp"

DofInfo::DofInfo(ParFiniteElementSpace *fes_): 
    pmesh(fes_->GetParMesh()), fes(fes_), dim(fes_->GetParMesh()->Dimension()), nDofs(fes->GetNDofs()), normals(fes->GetNBE(), dim), ElementToBdrElementTable(NULL), DofToDofTable(NULL)
{
    ParBilinearForm M(fes);
    M.AddDomainIntegrator(new MassIntegrator());
    M.Assemble();
    M.Finalize();
    massmatrix_ld = M.SpMat();
    HypreParMatrix *MHP = M.ParallelAssemble();
    MHP->MergeDiagAndOffd(massmatrix);
    //massmatrix = M.SpMat();

    I = massmatrix.GetI();
    J = massmatrix.GetJ();

    I_ld = massmatrix_ld.GetI();
    J_ld = massmatrix_ld.GetJ();

    /*
    BuildDofToDofTable();
    BuildElementToBdrElementTable();
    //*/

    BuildNormals();
}

DofInfo::~DofInfo() 
{   
    if(DofToDofTable){delete DofToDofTable;}
    if(ElementToBdrElementTable){delete ElementToBdrElementTable;}
}

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


void DofInfo::BuildElementToBdrElementTable() const
{
    int nE = fes->GetNE();
    int nBe = fes->GetNBE();
    Table *ElementToBdrElementTable1 = new Table();

    ElementToBdrElementTable1->MakeI(nE);

    for(int be = 0; be < nBe; be++)
    {
        auto trans = pmesh->GetBdrFaceTransformations(be);
        ElementToBdrElementTable1->AddAColumnInRow(trans->Elem1No);
    } 

    ElementToBdrElementTable1->MakeJ();
    for(int be = 0; be < nBe; be++)
    {
        auto trans = pmesh->GetBdrFaceTransformations(be);
        ElementToBdrElementTable1->AddConnection(trans->Elem1No, be);
    } 

    ElementToBdrElementTable1->ShiftUpI();
    ElementToBdrElementTable1->Finalize();
    ElementToBdrElementTable = ElementToBdrElementTable1;
}

void DofInfo::BuildDofToDofTable() const
{
    Table *DofToDofTable1 = new Table();
    int nDofs = fes->GetNDofs();

    DofToDofTable1->MakeI(nDofs);

    const auto II = massmatrix.ReadI();
    const auto JJ = massmatrix.ReadJ();

    for(int i = 0; i < nDofs; i ++)
    {
        const int begin = II[i];
        const int end = II[i+1];

        for(int j = begin; j < end; j++)
        {
            DofToDofTable1->AddAColumnInRow(i);
        }
    }

    DofToDofTable1->MakeJ();
    for(int i = 0; i < nDofs; i ++)
    {
        const int begin = II[i];
        const int end = II[i+1];
        for(int j = begin; j < end; j++)
        {
            DofToDofTable1->AddConnection(i, JJ[j]);
        }
    }

    DofToDofTable1->ShiftUpI();
    DofToDofTable1->Finalize();
    DofToDofTable = DofToDofTable1;
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


