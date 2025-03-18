#include "feevol.hpp"

void zero_numVar(const Vector &x, Vector &zero)
{
    zero = 0.0;
}

FEM::FEM(ParFiniteElementSpace *fes_, ParFiniteElementSpace *vfes_, System *sys_, DofInfo &dofs_,Vector &lumpedMassMatrix_):
    fes(fes_), vfes(vfes_), sys(sys_), dim(fes_->GetParMesh()->Dimension()), numVar(sys_->numVar), lumpedMassMatrix(lumpedMassMatrix_), pmesh(fes_->GetParMesh()),
    nDofs(fes_->GetNDofs()), dofs(dofs_), targetScheme(true), nE(fes->GetNE()), inflow(vfes), res_gf(vfes), gcomm(fes->GroupComm()), vgcomm(vfes->GroupComm()), 
    T(numVar * nDofs, numVar * nDofs)
    //shared(nDofs, nDofs) // size is only temporary, shared will be overwritten by its diagoffdiagmerged
{
    const char* fecol = fes->FEColl()->Name();
    if (strncmp(fecol, "H1", 2))
    {
        MFEM_ABORT("FiniteElementSpace must be H1 conforming (CG).");
    }   

    // Just to make sure
    MFEM_ASSERT(nDofs == lumpedMassMatrix.Size(), "nDofs x VDim might be wrong!");

    lumpedMassMatrix_synced = lumpedMassMatrix;
    SyncVector(lumpedMassMatrix_synced);

    ui.SetSize(numVar);
    uj.SetSize(numVar);
    vi.SetSize(numVar);
    vj.SetSize(numVar);
    cij.SetSize(dim);
    cji.SetSize(dim);

    dbc.SetSize(numVar * nDofs);
    dbc.UseDevice(true);
    
    adf.SetSize(numVar * nDofs);
    adf.UseDevice(true);

    aux1.SetSize(numVar * nDofs);
    aux1.UseDevice(true);

    aux2.SetSize(numVar * nDofs);
    aux2.UseDevice(true);

    Block_ij.SetSize(numVar, numVar);
    Identity_numVar.SetSize(numVar, numVar);
    Identity_numVar = 0.0;
    for(int n = 0; n < numVar; n++)
    {
        Identity_numVar(n,n) = 1.0;
    }

    fluxJac_i.SetSize(numVar,numVar,dim);
    fluxJac_j.SetSize(numVar,numVar,dim);

    fluxEval_i.SetSize(numVar, dim);
    fluxEval_j.SetSize(numVar, dim);

    flux_i.SetSize(numVar);
    flux_j.SetSize(numVar);

    // build convection matrix
    const int btype = BasisType::Positive;
    H1_FECollection fec(fes->GetFE(0)->GetOrder(), dim, btype);
    FiniteElementSpace dfes(fes->GetMesh(), &fec, dim, Ordering::byNODES);
    MixedBilinearForm Con(&dfes, fes);
    Con.AddDomainIntegrator(new VectorDivergenceIntegrator());
    Con.Assemble();
    Con.Finalize(0);
    Convection = Con.SpMat();


    SparseMatrix ML_numVar(nDofs * numVar, nDofs * numVar);
    SparseMatrix ML_inv_numVar(nDofs * numVar, nDofs * numVar);;
    for(int i = 0; i < nDofs; i++)
    {
        for(int n = 0; n < numVar; n++)
        {
            ML_numVar.Set(i + n * nDofs, i + n * nDofs, lumpedMassMatrix(i));
            ML_inv_numVar.Set(i + n * nDofs, i + n * nDofs, 1.0 / lumpedMassMatrix(i));
        }
    }
    ML_numVar.Finalize(0);
    ML_inv_numVar.Finalize(0);
    ML_inv = ML_inv_numVar;
    ML = ML_numVar;

    inflow.ProjectCoefficient(sys->bdrCond);

    auto tr = pmesh->GetBdrFaceTransformations(0);
    intorder = tr->Elem1->OrderW() + 2 * fes->GetFE(tr->Elem1No)->GetOrder();
    if (fes->GetFE(tr->Elem1No)->Space() == FunctionSpace::Pk)
    {
        intorder++;
    }

    auto I = dofs.I;
    auto J = dofs.J;
    
    for(int i = 0; i < nDofs; i++)
    {   
        for(int k = I[i]; k < I[i+1]; k++)
        {
            int j = J[k];
            for(int n = 0; n < numVar; n++)
            {
                for(int m = 0; m < numVar; m++)
                {
                    T.Set(i + n * nDofs, j + m * nDofs, 1.0);
                }
            }
        }
    }
    T.Finalize();
    T = 0.0;
}


void FEM::Assemble_A(const Vector &x, SparseMatrix &A) const
{   
    MFEM_ASSERT(x.Size() == numVar * nDofs, "wrong dimensions!");
    MFEM_ASSERT(A.Size() == x.Size(), "wrong Matrix dimensions!");
    A = 0.0;

    auto I = dofs.I;
    auto J = dofs.J;

    for(int i = 0; i < nDofs; i++)
    {
        for(int k = I[i]; k < I[i+1]; k++)
        {   
            int j = J[k];

            for(int n = 0; n < numVar; n++)
            {
                uj(n) = x(j + n * nDofs);
            }
            sys->EvaluateFluxJacobian(uj, fluxJac_j);

            Block_ij = 0.0;
            for(int d = 0; d < dim; d++)
            {   
                cij(d) = Convection(i, j + d * nDofs);
                Block_ij.Add(cij(d), fluxJac_j(d)); 
            }

            for(int n = 0; n < numVar; n++)
            {
                for(int m = 0; m < numVar; m++)
                {
                    // put into the global matrix orderring by nodes
                    A(i + n * nDofs, j + m * nDofs) += Block_ij(n, m);
                }
            }
        }
    }
}


void FEM::Assemble_minusD(const Vector &x, SparseMatrix &D) const
{
    MFEM_ASSERT(x.Size() == numVar * nDofs, "wrong dimensions!");
    MFEM_ASSERT(D.Size() == x.Size(), "wrong Matrix dimensions!");

    D = 0.0;
    auto I = dofs.I;
    auto J = dofs.J;

    for(int i = 0; i < nDofs; i++)
    {
        for(int n = 0; n < numVar; n++)
        {
            ui(n) = x(i + n * nDofs);
        }

        double dij_sum = 0.0;

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

            double dij = sys->ComputeDiffusion(cij, cji, ui, uj);
            dij_sum += dij;

            for(int n = 0; n < numVar; n++)
            {
                D(i + n * nDofs, j + n * nDofs) = - dij;
            }
        }

        for(int n = 0; n < numVar; n++)
        {
            D(i + n * nDofs, i + n * nDofs) = dij_sum;
        }
    }
}

double FEM::Compute_dt(const Vector &x, const bool HO_lowCFL_init, const double CFL_init, const double CFL_HO) const
{
    double CFL = CFL_HO;
    if(HO_lowCFL_init) { CFL = CFL_init;}

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
}


void FEM::Impbc(const Vector &x, SparseMatrix &bc, Vector &dbc) const
{   
    MFEM_ASSERT(dbc.Size() == numVar * nDofs, "wrong dimensions!");

    dbc = 0.0;
    FaceElementTransformations *tr;
    Vector nor(dim);
    
    for(int b = 0; b < fes->GetNBE(); b++)
    {   
        tr = pmesh->GetBdrFaceTransformations(b);
        MFEM_ASSERT(tr->Elem2No < 0, "no boundary element");

        // if supersonic outlet then integral over boundary is zero so skip
        if(tr->Attribute == 2)
        {   
            continue;
        }

        const IntegrationRule *ir = &IntRules.Get(tr->GetGeometryType(), intorder);

        const int dof = fes->GetFE(tr->Elem1No)->GetDof();
        Vector shape(dof);
        Vector aux_vec(numVar);
        aux_vec = 0.0;
        DenseMatrix aux_mat(dof, numVar);

        Array <int> vdofs;
        Array <int> edofs;
        vfes->GetElementVDofs(tr->Elem1No, vdofs);
        fes->GetElementDofs(tr->Elem1No, edofs);

        Vector u_n(vdofs.Size());
        x.GetSubVector(vdofs, u_n);
        Vector y1(numVar), y2(numVar), y_in_np1(vdofs.Size());
        inflow.GetSubVector(vdofs, y_in_np1);

        DenseMatrix B(numVar, numVar);
        dofs.normals.GetRow(b, nor);
        B = 0.0;
        /*
        if(tr->Attribute == 1 && numVar != 1)
        {   
            for(int d = 0; d < dim; d++)
            {
                for(int dd = 0; dd < dim; dd++)
                {   
                    B( d + 1, dd + 1) = nor(d) * nor(dd);
                }
            }
            B *= -2.0;
            B += Identity_numVar;
        }
        //*/

        DenseTensor nAu_dof(numVar, numVar, dof);
        nAu_dof = 0.0;

        for(int i = 0; i < dof; i++)
        {   
            for(int n = 0; n < numVar; n++)
            {
                y1(n) = u_n(i + n * dof);
                y2(n) = y_in_np1(i + n * dof);
            }
            
            sys->SetBoundaryConditions(y1, y2, nor, tr->Attribute);
            sys->EvaluateFluxJacobian(y1, fluxJac_i);
            double lambda = sys->ComputeLambdaij(nor, y1, y2);
            nAu_dof(i) = Identity_numVar;
            nAu_dof(i) *= lambda;

            switch (tr->Attribute)
            {

            case 1: // reflecting wall
            {
                sys->EvaluateFluxJacobian(y2, fluxJac_j);
                nAu_dof(i).Add( - lambda, B);
                break;
            }
            case 3: // Supersonic inlet.
            case 4: // Subsonic outlet.
            case 5: // Subsonic inlet.
            case 6: // transsonic outlet
            {
                fluxJac_j = 0.0;
                sys->EvaluateFlux(y2, fluxEval_j);
                aux_vec = y2;
                aux_vec *= lambda;
                fluxEval_j.AddMult_a(-1.0, nor, aux_vec);
                aux_mat.SetRow(i, aux_vec);
                break;
            }            
            case 2: // supersonic outlet gets cought earlier
            default:
                MFEM_ABORT("Invalid boundary attribute.");
            }
            
            DenseMatrix nA(numVar, numVar), nA_IpB(numVar, numVar);
            nA = 0.0,
            nA_IpB = 0.0;

            for(int d = 0; d < dim; d++)
            {
                nAu_dof(i).Add( - nor(d), fluxJac_i(d));
                nA.Add(nor(d), fluxJac_j(d));
            }

            for(int n = 0; n < numVar; n++)
            {
                for(int m = 0; m < numVar; m++)
                {
                    for(int l = 0; l < numVar; l++)
                    {
                        nA_IpB(n, m) += nA(n,l) * B(l,m); 
                    }
                }
            }

            nAu_dof(i) += nA_IpB;
        }

        for(int q = 0; q < ir->GetNPoints(); q++)
        {
            const IntegrationPoint &ip = ir->IntPoint(q);
            tr->SetAllIntPoints(&ip);
            shape = 0.0;
            fes->GetFE(tr->Elem1No)->CalcShape(tr->GetElement1IntPoint(), shape);
        
            shape *= ip.weight * tr->Weight() * 0.5;

            for(int i = 0; i < dof; i++)
            {   
                for(int n = 0; n < numVar; n++)
                {
                    for(int m = 0; m < numVar; m++)
                    {
                        // add to the global matrix orderring by nodes
                        bc(edofs[i] + n * nDofs, edofs[i] + m * nDofs) += shape(i) * nAu_dof(n,m,i); 
                    }

                    // dirichlet bc
                    dbc(edofs[i] + n * nDofs) += aux_mat(i,n) * shape(i); 
                }
            }
        }
    }
}



void FEM::Expbc(const Vector &x, Vector &bc) const
{
    FaceElementTransformations *tr;
    Vector nor(dim);
    bc = 0.0;

    for(int b = 0; b < fes->GetNBE(); b++)
    { 
        tr = pmesh->GetBdrFaceTransformations(b);
        MFEM_ASSERT(tr->Elem2No < 0, "no boundary element");

        // if supersonic outlet then integral over boundary is zero so skip
        if(tr->Attribute == 2)
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

double FEM::ComputeSteadyStateResidual(const Vector &x, ParGridFunction &res_gf) const
{
    ComputeSteadyStateResidual_gf(x, res_gf);
    VectorFunctionCoefficient zero(numVar, zero_numVar);
    steadyStateResidual = res_gf.ComputeL2Error(zero); // res_gf.ComputeMaxError(zero);
    return steadyStateResidual;
}

double FEM::ComputeEntropySteadyStateResidual(const Vector &u, ParGridFunction &res_gf) const
{
    ComputeSteadyStateResidual_gf(u, res_gf);
    //ML_inv.Mult(res_gf, aux1);

    res_gf = 0.0;

    for(int i = 0; i < nDofs; i++)
    {
        for(int n = 0 ; n < numVar; n++)
        {
            ui(n) = u(i + n * nDofs);
        }
        sys->EntropyVariable(ui, uj);
        
        for(int n = 0 ; n < numVar; n++)
        {
            res_gf(i) += aux1(i + n * nDofs) * uj(n);
        }
    }
    MFEM_ABORT("NEEDS SCALAR GRIDFUNCTION!");
    VectorFunctionCoefficient zero(numVar, zero_numVar);
    return res_gf.ComputeL2Error(zero);
}


double FEM::SteadyStateCheck(const Vector &u) const
{
   uOld = u;
   return steadyStateResidual;
}


void FEM::SyncVector(Vector &x) const
{
    MFEM_VERIFY(x.Size() == nDofs, "wrong size");
    Array<double> sync(x.GetData(), nDofs);
    gcomm.Reduce<double>(sync, GroupCommunicator::Sum);
    gcomm.Bcast(sync);
}


void FEM::VSyncVector(Vector &x) const
{
    MFEM_VERIFY(x.Size() == nDofs * numVar, "wrong size");
    Array<double> sync(x.GetData(), nDofs * numVar);
    vgcomm.Reduce<double>(sync, GroupCommunicator::Sum);
    vgcomm.Bcast(sync);
}