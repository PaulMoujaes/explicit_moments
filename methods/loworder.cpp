#include "loworder.hpp"

LowOrder::LowOrder(ParFiniteElementSpace *fes_, ParFiniteElementSpace *vfes_, System *sys_, DofInfo &dofs_,Vector &lumpedMassMatrix_) : 
        FE_Evolution(fes_, vfes_, sys_, dofs_, lumpedMassMatrix_)
{ }

void LowOrder::Mult(const Vector &x, Vector &y) const    
{   
    MFEM_VERIFY(sys->GloballyAdmissible(x), "not IDP!");
    Expbc(x, aux1);
    aux1 = 0.0;

    UpdateGlobalVector(x);

    auto I = dofs.I;
    auto J = dofs.J;
    for(int i = 0; i < nDofs; i++)
    {
        int i_td = fes->GetLocalTDofNumber(i);
        if(i_td == -1) {continue;}
        int i_gl = fes->GetGlobalTDofNumber(i);

        for(int n = 0; n < numVar; n++)
        {   
            ui(n) = x(i + n * nDofs);//x_gl->Elem(i_gl + n * GLnDofs);
            double xx = x(i + n * nDofs) - x_gl[n]->Elem(i_gl);
            MFEM_VERIFY(abs(xx) < 1e-15, "global and local not same");
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
                //cij(d) = Convection(i_td, j_gl +  d * nDofs);//C[d]->Elem(i_td,j_gl);
                //cji(d) = Convection(j_gl, i_td +  d * nDofs); //CT[d]->Elem(i_td,j_gl);

                cij(d) = C[d]->Elem(i_td,j_gl);
                cji(d) = CT[d]->Elem(i_td,j_gl);

            }

            double dij = sys->ComputeDiffusion(cij, cji, ui, uj);

            sys->EvaluateFlux(ui, fluxEval_i);
            sys->EvaluateFlux(uj, fluxEval_j);
            
            /*
            if(Mpi::Root())
            {
                ui.Print();
                uj.Print();
                cij.Print();
                cji.Print();
                fluxEval_i.Print();
                fluxEval_j.Print();
                cout << "----------------" << "\n\n";
            }
            //*/

            fluxEval_j -= fluxEval_i;
            fluxEval_j.Mult(cij, flux_j);

            for(int n = 0; n < numVar; n++)
            {
                aux1(i + n * nDofs) += (dij * (uj(n) - ui(n)) - ( flux_j(n)));
            }
        }
    }
    for(int n = 0; n < numVar; n++)
    {
        delete x_gl[n];
    }

    for(int i = 0; i < nDofs; i++)
    {
        if(fes->GetLocalTDofNumber(i) == -1)
        {
            for(int n = 0; n < numVar; n++)
            {   
                MFEM_VERIFY(abs(aux1(i + n * nDofs)) < 1e-15, "non td not zero");
            }
        }
    }

    VSyncVector(aux1);
    ML_inv.Mult(aux1, y);
    //MFEM_ABORT("")
}



// TODO parallel
void LowOrder::ComputeSteadyStateResidual_gf(const Vector &x, ParGridFunction &res) const 
{  
    /* 
    Expbc(x, aux1);

    auto I = dofs.I;
    auto J = dofs.J;
    for(int i = 0; i < nDofs; i++)
    {
        for(int n = 0; n < numVar; n++)
        {
            ui(n) = x(i + n * nDofs);
        }

        for(int k = I[i]; k < I[i+1]; k++)
        {
            int j = J[k];

            if(j == i)
            {
                continue;
            }

            for(int n = 0; n < numVar; n++)
            {
                uj(n) = x(j + n * nDofs);
            }

            for(int d = 0; d < dim; d++)
            {
                cij(d) = Convection(i, j + d * nDofs);
                cji(d) = Convection(j, i + d * nDofs);
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
    }

    ML_inv.Mult(aux1, res);
    //*/
}
