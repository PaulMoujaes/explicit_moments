#include "loworder.hpp"

LowOrder::LowOrder(ParFiniteElementSpace *fes_, ParFiniteElementSpace *vfes_, System *sys_, DofInfo &dofs_,Vector &lumpedMassMatrix_) : 
        FE_Evolution(fes_, vfes_, sys_, dofs_, lumpedMassMatrix_)
{ }

void LowOrder::Mult(const Vector &x, Vector &y) const    
{   
    MFEM_VERIFY(sys->GloballyAdmissible(x), "not IDP!");
    GetDiagOffDiagNodes(x, x_td, x_od);
    rhs_td = 0.0;
    if(sys->timedependentSource)
    {
        sys->q->SetTime(t);
        Source_LF.LinearForm::operator=(0.0); 
        Source_LF.Assemble();
        Source = Source_LF;
        //VSyncVector(Source);
        //cout << Source.Norml2()<< endl;
    }
    
    if(sys->timedependentbdr)
    {
        sys->bdrCond.SetTime(t);
        inflow.ProjectCoefficient(sys->bdrCond);
    }

    //bdr condition with ldofs because bdr edges are not shared
    Expbc(x, aux1);
    aux1 += Source;
    VSyncVector(aux1);

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

            double dij = sys->ComputeDiffusion(cij, cji, ui, uj);

            sys->EvaluateFlux(ui, fluxEval_i);
            sys->EvaluateFlux(uj, fluxEval_j);

            fluxEval_j -= fluxEval_i;
            fluxEval_j.Mult(cij, flux_j);

            for(int n = 0; n < numVar; n++)
            {
                rhs_td(i + n * TDnDofs) += (dij * (uj(n) - ui(n)) - ( flux_j(n)));
            }
        }

        if(offdiagsize > 0)
        {
            for(int k = I_offdiag[i]; k < I_offdiag[i+1]; k++)
            {
                int j = J_offdiag[k];

                for(int n = 0; n < numVar; n++)
                {   
                    uj(n) = x_od.GetBlock(n).Elem(j);
                }

                for(int d = 0; d < dim; d++)
                {   
                    cij(d) = C_offdiag[d]->GetData()[k];
                    cji(d) = C_offdiag_T[d]->GetData()[k];
                }

                double dij = sys->ComputeDiffusion(cij, cji, ui, uj);

                sys->EvaluateFlux(ui, fluxEval_i);
                sys->EvaluateFlux(uj, fluxEval_j);

                fluxEval_j -= fluxEval_i;
                fluxEval_j.Mult(cij, flux_j);

                for(int n = 0; n < numVar; n++)
                {
                    rhs_td(i + n * TDnDofs) += (dij * (uj(n) - ui(n)) - ( flux_j(n)));
                }
            }
        }
    }
    
    updated = false;
    vfes->GetProlongationMatrix()->AddMult(rhs_td, aux1);
    One_over_MLpdtMLs.Mult(aux1, y);
    ML_over_MLpdtMLs_m1.AddMult(x, y, 1.0 / dt);
}



void LowOrder::ComputeSteadyStateResidual_gf(const Vector &x, ParGridFunction &res) const 
{  
    UpdateGlobalVector(x);
    aux1 = 0.0;
    Expbc(x, aux1);

    auto I = dofs.I;
    auto J = dofs.J;
    for(int i = 0; i < nDofs; i++)
    {
        int i_td = fes->GetLocalTDofNumber(i);
        if(i_td == -1) {continue;}
        int i_gl = fes->GetGlobalTDofNumber(i);

        for(int n = 0; n < numVar; n++)
        {   
            ui(n) = x(i + n * nDofs);
        }

        for(int k = I[i_td]; k < I[i_td+1]; k++)
        {
            int j_gl = J[k];
            if(j_gl == i_gl){continue;}

            for(int n = 0; n < numVar; n++)
            {   
                uj(n) = x_gl[n]->Elem(j_gl);
            }

            for(int d = 0; d < dim; d++)
            {   
                cij(d) = C[d]->Elem(i_td,j_gl);
                cji(d) = CT[d]->Elem(i_td,j_gl);
            }

            double dij = sys->ComputeDiffusion(cij, cji, ui, uj);

            sys->EvaluateFlux(ui, fluxEval_i);
            sys->EvaluateFlux(uj, fluxEval_j);

            fluxEval_j -= fluxEval_i;
            fluxEval_j.Mult(cij, flux_j);

            for(int n = 0; n < numVar; n++)
            {
                aux1(i + n * nDofs) += (dij * (uj(n) - ui(n)) - ( flux_j(n)));
            }
        }
        // add source term 
        for(int n = 0; n < numVar; n++)
        {
            aux1(i + n * nDofs) += Source(i + n * nDofs) - (n == 0) * Mlumped_sigma_a(i) * ui(n) - (n > 0) * Mlumped_sigma_aps(i) * ui(n);
        }
    }
    updated = false;

    VSyncVector(aux1);
    ML_inv.Mult(aux1, res);
}
