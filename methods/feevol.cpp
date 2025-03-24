#include "feevol.hpp"

void zero_numVar(const Vector &x, Vector &zero)
{
    zero = 0.0;
}

FE_Evolution::FE_Evolution(ParFiniteElementSpace *fes_, ParFiniteElementSpace *vfes_, System *sys_, DofInfo &dofs_,Vector &lumpedMassMatrix_):
    TimeDependentOperator(vfes_->GetVSize()),
    fes(fes_), vfes(vfes_), sys(sys_), dim(fes_->GetParMesh()->Dimension()), lumpedMassMatrix(lumpedMassMatrix_), numVar(sys_->numVar), pmesh(fes_->GetParMesh()),
    nDofs(fes_->GetNDofs()), dofs(dofs_), nE(fes->GetNE()), inflow(vfes), res_gf(vfes), gcomm(fes->GroupComm()), vgcomm(vfes->GroupComm()), ML_inv(numVar * nDofs, numVar * nDofs), 
    GLnDofs(fes->GlobalTrueVSize()), TDnDofs(fes->GetTrueVSize()), aux_hpr(fes), C(dim), CT(dim), x_gl(numVar) //, GlD_to_TD(GLnDofs)
{
    const char* fecol = fes->FEColl()->Name();
    if (strncmp(fecol, "H1", 2))
    {
        MFEM_ABORT("FiniteElementSpace must be H1 conforming (CG).");
    }

    // Just to make sure
    MFEM_ASSERT(nDofs == lumpedMassMatrix.Size(), "nDofs x VDim might be wrong!");
    SyncVector(lumpedMassMatrix);

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
        C[d] = new SparseMatrix(nDofs, nDofs);
        CT[d] = new SparseMatrix(nDofs, nDofs);
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
                C[d]->Set(i,j, Convection(i,j + d * nDofs));
                CT[d]->Set(i,j, Convection_T(i + d * nDofs, j));
            }
        }
    }

    for(int d = 0; d < dim; d++)
    {
        C[d]->Finalize(0);
        CT[d]->Finalize(0);
    }

    ParBilinearForm dummy(fes);

    for(int d = 0; d < dim; d++)
    {
        HypreParMatrix *hpr = dummy.ParallelAssemble(C[d]);
        delete C[d];
        C[d] = new SparseMatrix;
        hpr->MergeDiagAndOffd(*C[d]);

        HypreParMatrix *hpr1 = dummy.ParallelAssemble(CT[d]);
        delete CT[d];
        CT[d] = new SparseMatrix;
        hpr1->MergeDiagAndOffd(*CT[d]);
    }

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

    for(int i = 0; i < nDofs; i++)
    {
        for(int n = 0; n < numVar; n++)
        {
            ML_inv.Set(i + n * nDofs, i + n * nDofs, 1.0 / lumpedMassMatrix(i));
        }
    }
    ML_inv.Finalize(0);

}

void FE_Evolution::ComputeLOTimeDerivatives(const Vector &x, Vector &y) const
{

}

double FE_Evolution::Compute_dt(const Vector &x, const double CFL) const
{
    /*
    double lambda_max = 0.0;

    auto I = dofs.I;
    auto J = dofs.J;

    for(int i = 0; i < nDofs; i++)
    {
        for(int n = 0; n < numVar; n++)
        {
            ui(n) = x(i + n * nDofs);
        }

        double  dij_sum = 0.0;

        for(int k = I[i]; k < I[i+1]; k++)
        {   
            int j = J[k];
            if(j == i)
            {
                continue;
            }

            for(int d = 0; d < dim; d++)
            {   
                // c_ij
                cij(d) = Convection(i, j + d * nDofs);
                cji(d) = Convection(j, i + d * nDofs);
            }

            for(int n = 0; n < numVar; n++)
            {
                uj(n) = x(j + n * nDofs);
            }

            dij_sum += sys->ComputeDiffusion(cij, cji, ui, uj);
        }
        lambda_max = max(lambda_max, 2.0 * dij_sum / lumpedMassMatrix(i));
    }
    return CFL / lambda_max;
    //*/
    return 0.001;
}

void FE_Evolution::Expbc(const Vector &x, Vector &bc) const
{
    FaceElementTransformations *tr;
    Vector nor(dim);
    bc = 0.0;
    for(int b = 0; b < fes->GetNBE(); b++)
    { 
        tr = pmesh->GetBdrFaceTransformations(b);
        MFEM_ASSERT(tr->Elem2No < 0, "no boundary element");

        // if supersonic outlet then integral over boundary is zero so skip
        if(tr->Attribute == 2 || tr->Attribute == 4)
        {   
            continue;
        }

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


void FE_Evolution::VSyncVector(Vector &x) const
{
    MFEM_VERIFY(x.Size() == nDofs * numVar, "wrong size");
    Array<double> sync(x.GetData(), nDofs * numVar);
    vgcomm.Reduce<double>(sync, GroupCommunicator::Sum);
    vgcomm.Bcast(sync);
}


void FE_Evolution::UpdateGlobalVector(const Vector &x) const
{
    MFEM_VERIFY(x.Size() == nDofs * numVar, "Wrong size!");

    for(int n = 0; n < numVar; n++)
    {
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
}