#include "feevol.hpp"

void zero_numVar(const Vector &x, Vector &zero)
{
    zero = 0.0;
}

FE_Evolution::FE_Evolution(ParFiniteElementSpace *fes_, ParFiniteElementSpace *vfes_, System *sys_, DofInfo &dofs_,Vector &lumpedMassMatrix_):
    TimeDependentOperator(vfes_->GetVSize()),
    fes(fes_), vfes(vfes_), sys(sys_), dim(fes_->GetParMesh()->Dimension()), lumpedMassMatrix(lumpedMassMatrix_), numVar(sys_->numVar), pmesh(fes_->GetParMesh()),
    nDofs(fes_->GetNDofs()), dofs(dofs_), nE(fes->GetNE()), inflow(vfes), res_gf(vfes), gcomm(fes->GroupComm()), vgcomm(vfes->GroupComm()), 
    ML_inv(numVar * nDofs, numVar * nDofs), ML_over_MLpdtMLs_m1(numVar * nDofs, numVar * nDofs), One_over_MLpdtMLs(numVar * nDofs, numVar * nDofs),
    GLnDofs(fes->GlobalTrueVSize()), TDnDofs(fes->GetTrueVSize()), aux_hpr(fes), C(dim), CT(dim), x_gl(numVar), updated(false),
    Mlumped_sigma_a(nDofs), Mlumped_sigma_aps(nDofs), uOld(vfes), Source_LF(vfes),

    C_diag(dim), C_diag_T(dim), C_offdiag(dim), C_offdiag_T(dim), hpr_con(dim), hpr_con_T(dim),
    offsets_diag(numVar +1), offsets_offdiag(numVar+1), rhs_td(TDnDofs * numVar), lumpedMassMatrix_td(TDnDofs), Mlumped_sigma_a_td(TDnDofs), Mlumped_sigma_aps_td(TDnDofs)
{
    const char* fecol = fes->FEColl()->Name();
    if (strncmp(fecol, "H1", 2))
    {
        MFEM_ABORT("FiniteElementSpace must be H1 conforming (CG).");
    }

    x_gl = NULL;


    // Just to make sure
    MFEM_ASSERT(nDofs == lumpedMassMatrix.Size(), "nDofs x VDim might be wrong!");
    SyncVector(lumpedMassMatrix);
    fes->GetRestrictionMatrix()->Mult(lumpedMassMatrix, lumpedMassMatrix_td);

    ui.SetSize(numVar);
    uj.SetSize(numVar);
    vi.SetSize(numVar);
    vj.SetSize(numVar);
    cij.SetSize(dim);
    cji.SetSize(dim);

    dbc.SetSize(numVar * nDofs);
    dbc.UseDevice(true);

    aux1.SetSize(numVar * nDofs);
    aux1.UseDevice(true);

    aux2.SetSize(numVar * nDofs);
    aux2.UseDevice(true);

    fluxEval_i.SetSize(numVar, dim);
    fluxEval_j.SetSize(numVar, dim);

    flux_i.SetSize(numVar);
    flux_j.SetSize(numVar);

    // build convection matrix
    const int btype = BasisType::ClosedUniform;
    H1_FECollection fec(fes->GetFE(0)->GetOrder(), dim, btype);
    ParFiniteElementSpace dfes(fes->GetParMesh(), &fec, dim, Ordering::byNODES);
    ParMixedBilinearForm Con(&dfes, fes);
    Con.AddDomainIntegrator(new VectorDivergenceIntegrator());
    Con.Assemble();
    Con.Finalize(0);
    Convection = Con.SpMat();
    //HypreParMatrix *C_hpm = Con.ParallelAssemble();
    //C_hpm->MergeDiagAndOffd(Convection);


    ParMixedBilinearForm Con_T(fes, &dfes);
    Con_T.AddDomainIntegrator(new TransposeIntegrator(new VectorDivergenceIntegrator()));
    Con_T.Assemble();
    Con_T.Finalize(0);
    Convection_T = Con_T.SpMat();
    //HypreParMatrix *CT_hpm = Con_T.ParallelAssemble();
    //CT_hpm->MergeDiagAndOffd(Convection_T);

    for(int d = 0; d < dim; d++)
    {
        C_diag[d] = new SparseMatrix(nDofs, nDofs);
        *C_diag[d] = dofs.massmatrix_ld;
        C_diag_T[d] = new SparseMatrix(nDofs, nDofs);
        *C_diag_T[d] = dofs.massmatrix_ld;
    }

    auto I_ld = dofs.I_ld;
    auto J_ld = dofs.J_ld;

    for(int i = 0; i < nDofs; i++)
    {
        for(int k = I_ld[i]; k < I_ld[i+1]; k++)
        {
            int j = J_ld[k];
            for(int d = 0; d < dim; d++)
            {
                C_diag[d]->Elem(i,j) = Convection(i,j + d * nDofs);
                C_diag_T[d]->Elem(i,j) = Convection_T(i + d * nDofs, j);
            }
        }
    }

    ParBilinearForm dummy(fes);
    for(int d = 0; d < dim; d++)
    {
        HYPRE_BigInt *cmap1 = NULL;
        HYPRE_BigInt *cmap2 = NULL;
        hpr_con[d] = dummy.ParallelAssemble(C_diag[d]);
        delete C_diag[d];

        C_diag[d] = new SparseMatrix;
        C_offdiag[d] = new SparseMatrix;
        hpr_con[d]->GetDiag(*C_diag[d]);
        hpr_con[d]->GetOffd(*C_offdiag[d], cmap1);

        hpr_con_T[d] = dummy.ParallelAssemble(C_diag_T[d]);
        delete C_diag_T[d];
        C_diag_T[d] = new SparseMatrix;
        C_offdiag_T[d] = new SparseMatrix;
        hpr_con_T[d]->GetDiag(*C_diag_T[d]);
        hpr_con_T[d]->GetOffd(*C_offdiag_T[d], cmap2);
    }

    offdiagsize = C_offdiag[0]->Width();

    for (int n = 0; n < numVar +1; n++) 
    { 
        offsets_diag[n] = n * TDnDofs; 
        offsets_offdiag[n] = n * offdiagsize;
    }
    x_td.Update(offsets_diag);
    x_td.UseDevice(true);
    x_od.Update(offsets_offdiag);
    x_od.UseDevice(true);

    sys->bdrCond.SetTime(0.0);
    inflow.ProjectCoefficient(sys->bdrCond);

    if(fes->GetNBE() > 0)
    {
        auto tr = pmesh->GetBdrFaceTransformations(0);
        intorder = tr->Elem1->OrderW() + 2 * fes->GetFE(tr->Elem1No)->GetOrder();
        if (fes->GetFE(tr->Elem1No)->Space() == FunctionSpace::Pk)
        {
            intorder++;
        }
    }
    else 
    {
        intorder = 0;
    }

    Vector ones(nDofs);
    ones = 1.0;

    HYPRE_BigInt *cmap1 = NULL;
    HYPRE_BigInt *cmap2 = NULL;

    ParBilinearForm sigma_a(fes);
    sigma_a.AddDomainIntegrator(new MassIntegrator(*sys->Sigma_0));
    sigma_a.Assemble(0);
    sigma_a.Finalize(0);
    M_sigma_a = sigma_a.SpMat();
    
    M_sigma_a.Mult(ones, Mlumped_sigma_a);
    SyncVector(Mlumped_sigma_a);

    M_a_HP = sigma_a.ParallelAssemble(&M_sigma_a);
    M_a_HP->GetDiag(M_sigma_a_diag);
    M_a_HP->GetOffd(M_sigma_a_offdiag, cmap1);

    ParBilinearForm sigma_aps(fes);
    sigma_aps.AddDomainIntegrator(new MassIntegrator(*sys->Sigma_1));
    sigma_aps.Assemble(0);
    sigma_aps.Finalize(0);
    M_sigma_aps = sigma_aps.SpMat();

    M_sigma_aps.Mult(ones, Mlumped_sigma_aps);
    SyncVector(Mlumped_sigma_aps);

    M_aps_HP = sigma_aps.ParallelAssemble(&M_sigma_aps);
    M_aps_HP->GetDiag(M_sigma_aps_diag);
    M_aps_HP->GetOffd(M_sigma_aps_offdiag, cmap2);

    /*
    cout << dofs.M_offdiag.Width() << ", " << offdiagsize << endl;
    cout << M_sigma_aps_offdiag.Width() << ", " << offdiagsize << endl;
    cout << M_sigma_a_offdiag.Width() << ", " << offdiagsize << endl;
    cout << endl;
    cout << M_sigma_aps_offdiag.Height() << ", " << TDnDofs << endl;
    cout << M_sigma_a_offdiag.Height() << ", " << TDnDofs << endl;
    cout << "----------" <<endl;
    //*/

    //ParLinearForm Source_LF(vfes);
    Source_LF.AddDomainIntegrator(new VectorDomainLFIntegrator(*sys->q));
    Source_LF.Assemble();
    Source = Source_LF;
    //VSyncVector(Source);

    fes->GetRestrictionMatrix()->Mult(Mlumped_sigma_a, Mlumped_sigma_a_td);
    fes->GetRestrictionMatrix()->Mult(Mlumped_sigma_aps, Mlumped_sigma_aps_td);

    for(int i = 0; i < nDofs; i++)
    {
        for(int n = 0; n < numVar; n++)
        {   
            ML_inv.Set(i + n * nDofs, i + n * nDofs, 1.0 / lumpedMassMatrix(i));
            ML_over_MLpdtMLs_m1.Set(i + n * nDofs, i + n * nDofs,  1.0); // gets reassembled casue dependent on dt
            One_over_MLpdtMLs.Set(i + n * nDofs, i + n * nDofs,  1.0); // gets reassembled casue dependent on dt
        }
    }
    ML_inv.Finalize(0);
    ML_over_MLpdtMLs_m1.Finalize(0);
    One_over_MLpdtMLs.Finalize(0);

}

void FE_Evolution::ComputeLOTimeDerivatives(const Vector &x, Vector &y) const
{
    
}

double FE_Evolution::ComputeSteadyStateResidual_quick(const ParGridFunction &uOld, const ParGridFunction &u, const double dt) const
{   
    /*
    res_gf = u;
    res_gf -= uOld;
    double odt = 1.0 / dt;
    res_gf *= odt;

    steadyStateResidual = res_gf.ComputeL2Error(zero);
    if(Mpi::Root())
    {
        cout << steadyStateResidual << endl; // res_gf.ComputeMaxError(zero);
    }
    //*/


    VectorFunctionCoefficient zero(numVar, zero_numVar);
    ComputeSteadyStateResidual_gf(u, res_gf);
    //VectorFunctionCoefficient zero(numVar, zero_numVar);
    steadyStateResidual = res_gf.ComputeL2Error(zero); // res_gf.ComputeMaxError(zero);

    return steadyStateResidual;
}


void FE_Evolution::Set_dt_Update_MLsigma(const double dt_)
{
    //*
    dt = dt_;

    for(int i = 0; i < nDofs; i++)
    {
        One_over_MLpdtMLs(i, i) =  1.0 / (lumpedMassMatrix(i) + dt * Mlumped_sigma_a(i) );
        ML_over_MLpdtMLs_m1(i, i) = -1.0 + lumpedMassMatrix(i) / (lumpedMassMatrix(i) + dt * Mlumped_sigma_a(i));
        
        for(int n = 1; n < numVar; n++)
        {
            One_over_MLpdtMLs(i + n * nDofs, i + n * nDofs) = 1.0 / (lumpedMassMatrix(i) + dt * Mlumped_sigma_aps(i));
            ML_over_MLpdtMLs_m1(i + n * nDofs, i + n * nDofs) = -1.0 + lumpedMassMatrix(i) / (lumpedMassMatrix(i) + dt * Mlumped_sigma_aps(i));
        }
    }
    //*/
}

double FE_Evolution::Compute_dt(const Vector &x, const double CFL) const
{
    //*
    double lambda_max = 0.0;
    GetDiagOffDiagNodes(x, x_td, x_od);

    auto I_diag = C_diag[0]->GetI();
    auto J_diag = C_diag[0]->GetJ();
    auto I_offdiag = C_offdiag[0]->GetI();
    auto J_offdiag = C_offdiag[0]->GetJ();

    for(int i = 0; i < TDnDofs; i++)
    {
        for(int n = 0; n < numVar; n++)
        {   
            ui(n) = x_td.GetBlock(n).Elem(i);
        }

        double  dij_sum = 0.0;

        for(int k = I_diag[i]; k < I_diag[i+1]; k++)
        {   
            int j = J_diag[k];
            if(i == j){continue;}

            for(int n = 0; n < numVar; n++)
            {   
                uj(n) = x_td.GetBlock(n).Elem(j);
            }

            for(int d = 0; d < dim; d++)
            {   
                cij(d) = C_diag[d]->GetData()[k];
                cji(d) = C_diag_T[d]->GetData()[k];
            }

            dij_sum += sys->ComputeDiffusion(cij, cji, ui, uj);
        } 

        if(offdiagsize > 0)
        {   
            for(int k = I_offdiag[i]; k < I_offdiag[i+1]; k++)
            {
                int j = J_offdiag[k];
                //if(i == j){continue;}
                //MFEM_VERIFY( i != j, "offdiag j = i!");

                for(int n = 0; n < numVar; n++)
                {   
                    uj(n) = x_od.GetBlock(n).Elem(j);
                }

                for(int d = 0; d < dim; d++)
                {   
                    cij(d) = C_offdiag[d]->GetData()[k];
                    cji(d) = C_offdiag_T[d]->GetData()[k];
                }

                dij_sum += sys->ComputeDiffusion(cij, cji, ui, uj);
            }
        }

        lambda_max = max(lambda_max, 2.0 * dij_sum / lumpedMassMatrix_td(i));
    }
    double glob_lambdamax;
    MPI_Allreduce(&lambda_max, &glob_lambdamax, 1, MPI_DOUBLE, MPI_MAX,
        MPI_COMM_WORLD);
        
    return CFL / glob_lambdamax;
}

void FE_Evolution::Expbc(const Vector &x, Vector &bc) const
{
    bc = 0.0;
    if(!sys->bdrcon_needed)
    {   
        return;
    }
    FaceElementTransformations *tr = NULL;
    Vector nor(dim);
    for(int b = 0; b < fes->GetNBE(); b++)
    { 
        tr = pmesh->GetBdrFaceTransformations(b);
        MFEM_ASSERT(tr->Elem2No < 0, "no boundary element");

        // if supersonic outlet then integral over boundary is zero so skip
        if(tr->Attribute == 2 || tr->Attribute == 4)
        {   
            continue;
        }
        //else 
        //{
        //    continue;
        //}
        //MFEM_ABORT(to_string(tr->Attribute))

        const IntegrationRule *ir = &IntRules.Get(tr->GetGeometryType(), intorder);

        const int dof = fes->GetFE(tr->Elem1No)->GetDof();
        Vector shape(dof);

        Array <int> vdofs;
        Array <int> edofs;
        vfes->GetElementVDofs(tr->Elem1No, vdofs);
        fes->GetElementDofs(tr->Elem1No, edofs);

        Vector u_n(vdofs.Size());
        x.GetSubVector(vdofs, u_n);
        Vector y_in_np1(vdofs.Size());
        inflow.GetSubVector(vdofs, y_in_np1);

        dofs.normals.GetRow(b, nor);

        DenseMatrix aux_mat(numVar, dof);

        for(int i = 0; i < dof; i++)
        {   
            for(int n = 0; n < numVar; n++)
            {
                ui(n) = u_n(i + n * dof);
                uj(n) = y_in_np1(i + n * dof);
            }

            sys->SetBoundaryConditions(ui, uj, nor, tr->Attribute);
            sys->ComputeLaxFriedrichsFlux(ui, uj, nor, flux_j);
            sys->EvaluateFlux(ui, fluxEval_i);
            fluxEval_i.Mult(nor, flux_i);
            flux_i -= flux_j;
            aux_mat.SetCol(i, flux_i);        
        }

        for(int q = 0; q < ir->GetNPoints(); q++)
        {
            const IntegrationPoint &ip = ir->IntPoint(q);
            tr->SetAllIntPoints(&ip);
            //shape = 0.0;
            fes->GetFE(tr->Elem1No)->CalcShape(tr->GetElement1IntPoint(), shape);
        
            shape *= ip.weight * tr->Weight();

            for(int i = 0; i < dof; i++)
            {   
                for(int n = 0; n < numVar; n++)
                {
                    // add to the vector orderring by nodes
                    bc(edofs[i] + n * nDofs) += shape(i) * aux_mat(n, i);
                }
            }
        }
    }
}

double FE_Evolution::ComputeSteadyStateResidual(const Vector &x, ParGridFunction &res_gf) const
{
    ComputeSteadyStateResidual_gf(x, res_gf);
    VectorFunctionCoefficient zero(numVar, zero_numVar);
    steadyStateResidual = res_gf.ComputeL2Error(zero); // res_gf.ComputeMaxError(zero);
    return steadyStateResidual;
}

double FE_Evolution::SteadyStateCheck(const Vector &u) const
{
   uOld = u;
   return steadyStateResidual;
}


void FE_Evolution::SyncVector(Vector &x) const
{
    MFEM_VERIFY(x.Size() == nDofs, "wrong size");
    Array<double> sync(x.GetData(), nDofs);
    gcomm.Reduce<double>(sync, GroupCommunicator::Sum);
    gcomm.Bcast(sync);
}

void FE_Evolution::SyncVector_Max(Vector &x) const
{
    MFEM_VERIFY(x.Size() == nDofs, "wrong size");
    Array<double> sync(x.GetData(), nDofs);
    gcomm.Reduce<double>(sync, GroupCommunicator::Max);
    gcomm.Bcast(sync);
}

void FE_Evolution::SyncVector_Min(Vector &x) const
{
    MFEM_VERIFY(x.Size() == nDofs, "wrong size");
    Array<double> sync(x.GetData(), nDofs);
    gcomm.Reduce<double>(sync, GroupCommunicator::Min);
    gcomm.Bcast(sync);
}


void FE_Evolution::VSyncVector(Vector &x) const
{
    MFEM_VERIFY(x.Size() == nDofs * numVar, "wrong size");
    Array<double> sync(x.GetData(), nDofs * numVar);
    vgcomm.Reduce<double>(sync, GroupCommunicator::Sum);
    vgcomm.Bcast(sync);
}


void FE_Evolution::GetDiagOffDiagNodes(const Vector &x, BlockVector &x_td, BlockVector &x_od) const
{
    vfes->GetRestrictionMatrix()->Mult(x, x_td);
    for(int n = 0; n < numVar; n++)
    {
        dofs.Extract_offd_hypre(hpr_con[0], x_td.GetBlock(n), x_od.GetBlock(n), offdiagsize);
    }
}

void FE_Evolution::GetOffDiagNodes(const BlockVector &x_td, BlockVector &x_od) const
{
    for(int n = 0; n < numVar; n++)
    {
        dofs.Extract_offd_hypre(hpr_con[0], x_td.GetBlock(n), x_od.GetBlock(n), offdiagsize);
    }
}

void FE_Evolution::UpdateGlobalVector(const Vector &x) const
{
    if(updated)
    {
        return;
    }

    MFEM_VERIFY(x.Size() == nDofs * numVar, "Wrong size!");

    for(int n = 0; n < numVar; n++)
    {
        if(x_gl[n])
        {
            delete x_gl[n];
        }
        
        for(int i = 0; i < nDofs; i++)
        {
            int i_td = fes->GetLocalTDofNumber(i);
            if(i_td != -1)
            {                
                aux_hpr(i_td) = x(i + n * nDofs);
            }
        }

        x_gl[n] = aux_hpr.GlobalVector();
        MFEM_VERIFY(x_gl[n]->Size() == GLnDofs, "wrong size!");
    }
    updated = true;
}
