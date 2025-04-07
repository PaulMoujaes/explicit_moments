#include "mcl.hpp"

// PositivityFix = 1 -> true
#define PositivityFix 1

MCL::MCL(ParFiniteElementSpace *fes_, ParFiniteElementSpace *vfes_, System *sys_, DofInfo &dofs_,Vector &lumpedMassMatrix_) : 
        FE_Evolution(fes_, vfes_, sys_, dofs_, lumpedMassMatrix_),
        uDot(numVar * nDofs), adf(numVar * nDofs), umin_loc(TDnDofs, numVar), umax_loc(TDnDofs, numVar), min_scalar(TDnDofs), max_scalar(TDnDofs),
        fij_gl(numVar), uij(numVar), uji(numVar), umin(numVar), umax(numVar), aux1_hpr(fes), uDot_gl(numVar)
{
    uDot.UseDevice(true);
    adf.UseDevice(true);
    adf = 0.0;

    umin = NULL;
    umax = NULL;
    uDot_gl = NULL;

    auto I = dofs.I;
    auto J = dofs.J;

    // for matrices where we need the full stenci cause non-symmetric
    SparseMatrix dummy_mat(TDnDofs, GLnDofs);
    for(int i = 0; i < nDofs; i++)
    {
        int i_td = fes->GetLocalTDofNumber(i);
        if(i_td == -1) {continue;}
        int i_gl = fes->GetGlobalTDofNumber(i);

        for(int k = I[i_td]; k < I[i_td+1]; k++)
        {
            int j_gl = J[k];
            if(j_gl == i_gl){continue;}

            dummy_mat.Set(i_td, j_gl, 1.0);
        }
    }
    dummy_mat.Finalize(0);
    dummy_mat = 0.0;

    for(int n = 0; n < numVar; n++)
    {
        fij_gl[n] = new SparseMatrix(TDnDofs, GLnDofs);
        *fij_gl[n] = dummy_mat;
    }
}

void MCL::Mult(const Vector &x, Vector &y) const
{
    MFEM_VERIFY(sys->GloballyAdmissible(x), "not IDP!");
    UpdateGlobalVector(x);
    Expbc(x, aux1);
    ComputeAntiDiffusiveFluxes(x, aux1, adf);

    aux1 += adf;

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
    }

    VSyncVector(aux1);
    ML_inv.Mult(aux1, y);

    updated = false;
}

void MCL::CalcMinMax(const Vector &x) const
{   
    umin_loc = 0.0; umax_loc = 0.0;

    auto I = dofs.I;
    auto J = dofs.J;
    
    for(int i = 0; i < nDofs; i++)
    {
        int i_td = fes->GetLocalTDofNumber(i);
        if(i_td == -1 ){continue;}
        int i_gl = fes->GetGlobalTDofNumber(i);

        umin_loc(i_td,0) = x(i);
        umax_loc(i_td,0) = x(i);

        for(int n = 1; n < numVar; n++)
        {
            umin_loc(i_td, n) = x(i + n * nDofs) / x(i);
            umax_loc(i_td, n) = umin_loc(i_td, n);
        }

        for(int k = I[i_td]; k < I[i_td+1]; k++)
        {
            int j_gl = J[k];
            if(i_gl == j_gl){continue;}

            double dij = CalcBarState_returndij(i, j_gl , uij, uji);
            
            umin_loc(i_td, 0) = min(umin_loc(i_td, 0), x_gl[0]->Elem(j_gl));
            umax_loc(i_td, 0) = max(umax_loc(i_td, 0), x_gl[0]->Elem(j_gl));

            umin_loc(i_td, 0) = min(umin_loc(i_td, 0), uij(0));
            umax_loc(i_td, 0) = max(umax_loc(i_td, 0), uij(0));

            for(int n = 1; n < numVar; n++)
            {
                double vj = x_gl[n]->Elem(j_gl) / x_gl[0]->Elem(j_gl);
                umin_loc(i_td, n) = min(umin_loc(i_td, n), vj);
                umax_loc(i_td, n) = max(umax_loc(i_td, n), vj);          

                double vij = (uij(n) + uji(n)) / (uij(0) + uji(0) );
                umin_loc(i_td, n) = min(umin_loc(i_td, n), vij);
                umax_loc(i_td, n) = max(umax_loc(i_td, n), vij); 
            }
        }
    }


    for(int n = 0; n < numVar; n++)
    {
        if(umin[n])
        {
            delete umin[n];
        }
        if(umax[n])
        {
            delete umax[n];
        }

        min_scalar.SetData(umin_loc.GetColumn(n));
        max_scalar.SetData(umax_loc.GetColumn(n));

        //SyncVector_Min(min_scalar);
        //SyncVector_Max(max_scalar);

        
        for(int i_td = 0; i_td < TDnDofs; i_td++)
        {        
            aux_hpr(i_td) = min_scalar(i_td);
            aux1_hpr(i_td) = max_scalar(i_td);
         
        }

        umin[n] = aux_hpr.GlobalVector();
        umax[n] = aux1_hpr.GlobalVector();
    }   
}



double MCL::CalcBarState_returndij(const int i, const int j_gl, Vector &uij, Vector &uji) const
{
    int i_td = fes->GetLocalTDofNumber(i);
    if(i_td == -1)
    {
        MFEM_ABORT("i_td = -1");
    }

    int i_gl = fes->GetGlobalTDofNumber(i);
    if(j_gl == i_gl )
    {
        MFEM_ABORT("i = j in barstates");
    }

    for(int n = 0; n < numVar; n++)
    {
        ui(n) = x_gl[n]->Elem(i_gl);
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
    fluxEval_i.Mult(cij, flux_i);
    fluxEval_j.Mult(cij, flux_j);
    for(int n = 0; n < numVar; n++)
    {   
        uij(n) = 0.5 * ((ui(n) + uj(n)) - (flux_j(n) - flux_i(n)) / dij ) ;
    }

    fluxEval_i.Mult(cji, flux_i);
    fluxEval_j.Mult(cji, flux_j);

    for(int n = 0; n < numVar; n++)
    {   
        uji(n) = 0.5 * ((ui(n) + uj(n)) - (flux_i(n) - flux_j(n)) / dij ) ;
    }

    return dij;
}



void MCL::ComputeAntiDiffusiveFluxes(const Vector &x, const Vector &dbc, Vector &AntiDiffFluxes) const
{   
    UpdateGlobalVector(x);
    CalcUdot(x, dbc);
    CalcMinMax(x);

    auto I = dofs.I;
    auto J = dofs.J;

    for(int i = 0; i < nDofs; i++)
    {   
        int i_td = fes->GetLocalTDofNumber(i);
        if(i_td == -1) {continue;}
        int i_gl = fes->GetGlobalTDofNumber(i);

        for(int k = I[i_td]; k < I[i_td+1]; k++)
        {
            int j_gl = J[k];
            if(j_gl == i_gl){continue;}

            double dij = CalcBarState_returndij(i, j_gl, uij, uji);
             
            const double fij_ = dofs.massmatrix(i_td, j_gl) * (uDot_gl[0]->Elem(i_gl) - uDot_gl[0]->Elem(j_gl)) + dij * (x_gl[0]->Elem(i_gl) - x_gl[0]->Elem(j_gl));
            double fij_star;
            
            if( fij_> 0)
            {
                double fij_min = 2.0 * dij * (umax[0]->Elem(i_gl) - uij(0));
                double fij_max = 2.0 * dij* (uji(0) - umin[0]->Elem(j_gl));
                double fij_bound = min(fij_max, fij_min);
                
                //fij_max = 2.0 * dij* (uji - 1e-15);
                //fij_bound = fij_max;

                fij_star = min(fij_, fij_bound);
                fij_star = max(0.0, fij_star);
            }
            else
            {
                double fij_min = 2.0 * dij * (umin[0]->Elem(i_gl) - uij(0));
                double fij_max = 2.0 * dij* (uji(0) - umax[0]->Elem(j_gl));
                double fij_bound = max(fij_max, fij_min);

                //fij_min = 2.0 * dij * (1e-15 - uij);
                //fij_bound = fij_min;

                fij_star = max(fij_, fij_bound); 
                fij_star= min(0.0 , fij_star);
            }

            //*
            if(abs(x_gl[0]->Elem(i_gl) - x_gl[0]->Elem(j_gl)) < 1e-14)
            {
                fij_star = 0.0;
            }
            //*/

            MFEM_ASSERT(abs(fij_star) <= abs(fij_) + 1e-15, "a_ij density wrong!");
            
            fij_gl[0]->Elem(i_td, j_gl) = fij_star;
        }
    }

     
    // limiting the conserved products

    //*

    for(int i = 0; i < nDofs; i++)
    {
        int i_td = fes->GetLocalTDofNumber(i);
        if(i_td == -1) {continue;}
        int i_gl = fes->GetGlobalTDofNumber(i);

        for(int k = I[i_td]; k < I[i_td+1]; k++)
        {
            int j_gl = J[k];
            if(j_gl == i_gl){continue;}

            double dij = CalcBarState_returndij(i, j_gl, uij, uji);

            for(int n = 1; n < numVar; n++)
            {
                const double phi_ij = (uij(n) + uji(n)) / (uij(0) + uji(0) );
                const double phi_ji = phi_ij;

                const double rij = uij(0) + 0.5 * fij_gl[0]->Elem(i_td, j_gl) / dij;
                const double rji = uji(0) - 0.5 * fij_gl[0]->Elem(i_td, j_gl) / dij;

                const double fij_ = dofs.massmatrix(i_td, j_gl) * (uDot_gl[n]->Elem(i_gl) - uDot_gl[n]->Elem(j_gl))  + dij * (x_gl[n]->Elem(i_gl) - x_gl[n]->Elem(j_gl));  
                const double gij_ = fij_ + 2.0 * dij * (uij(n) - rij * phi_ij) ;

                double gij_star;
                if(gij_ > 0.0)
                {   
                    const double gij_max = 2.0 * dij * min(rij * (umax[n]->Elem(i_gl) - phi_ij), rji * (phi_ji - umin[n]->Elem(j_gl)));
                    gij_star = max(0.0, min(gij_, gij_max));
                }
                else 
                {
                    const double gij_min =  2.0 * dij * max(rij * (umin[n]->Elem(i_gl) - phi_ij) , rji * (phi_ji - umax[n]->Elem(j_gl)));
                    gij_star = min(0.0, max(gij_, gij_min));
                }

                MFEM_ASSERT(abs(gij_star) < abs(gij_) + 1e-15, "a_ij rhophi wrong!");                
                double fij_star = gij_star - 2.0 * dij * (uij(n) - rij * phi_ij);

                fij_gl[n]->Elem(i_td, j_gl) = fij_star;
            }
        }
    }
    //*/

    // positivity fix
    #if PositivityFix == 1
        for(int i = 0; i < nDofs; i++)
        {   
            int i_td = fes->GetLocalTDofNumber(i);
            if(i_td == -1) {continue;}
            int i_gl = fes->GetGlobalTDofNumber(i);

            for(int k = I[i_td]; k < I[i_td+1]; k++)
            {
                int j_gl = J[k];
                if(j_gl == i_gl){continue;}

                double dij = CalcBarState_returndij(i, j_gl, uij, uji);

                for(int n = 0; n < numVar; n++)
                {
                    uij(n) += 0.5 * fij_gl[n]->Elem(i_td, j_gl) / dij;  
                    uji(n) -= 0.5 * fij_gl[n]->Elem(i_td, j_gl) / dij;
                }
                
                if(sys->ComputePressureLikeVariable(uij) < 0.0 || sys->ComputePressureLikeVariable(uji) < 0.0)
                {
                    for(int n = 0; n < numVar; n++)
                    {
                        fij_gl[n]->Elem(i_td, j_gl) = 0.0;
                    }
                }
            }
        }
    #endif

    AntiDiffFluxes = 0.0;
    for(int n = 0; n < numVar; n++)
    {
        auto II = fij_gl[n]->ReadI();
        auto Fij = fij_gl[n]->ReadData();

        for(int i = 0; i < nDofs; i++)
        {
            int i_td = fes->GetLocalTDofNumber(i);
            if(i_td == -1) {continue;}

            for(int k = II[i_td]; k < II[i_td+1]; k++)
            {
                AntiDiffFluxes(i + n * nDofs) += Fij[k];
            }
        }
    }


    /*
    const auto II = fij.ReadI();
    //const auto JJ = fij.ReadJ();
    const auto AA = fij.ReadData();
    double *adf = AntiDiffFluxes.ReadWrite();
    mfem::forall(fij.Height(), [=] MFEM_HOST_DEVICE (int i)
    {
        const int begin = II[i];
        const int end = II[i+1];

        double sum = 0.0;
        for(int j = begin; j < end; j++)
        {
            sum += AA[j];
        }
        adf[i] = sum;
    });
    //*/
}

void MCL::CalcUdot(const Vector &x, const Vector &dbc) const
{   
    // TBD: We need the global Vector of uDot

    aux2 = dbc;

    // just to make sure
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
                aux2(i + n * nDofs) += (dij * (uj(n) - ui(n)) - ( flux_j(n)));
            }
        }
    }

    VSyncVector(aux2);
    ML_inv.Mult(aux2, uDot);

    for(int n = 0; n < numVar; n++)
    {
        if(uDot_gl[n])
        {
            delete uDot_gl[n];
        }

        for(int i = 0; i < nDofs; i++)
        {
            int i_td = fes->GetLocalTDofNumber(i);
            if(i_td != -1)
            {                
                aux_hpr(i_td) = uDot(i + n * nDofs);
            }
        }

        uDot_gl[n] = aux_hpr.GlobalVector();
        MFEM_VERIFY(uDot_gl[n]->Size() == GLnDofs, "wrong size!");
    }
}


void MCL::ComputeSteadyStateResidual_gf(const Vector &x, ParGridFunction &res) const
{   
    Mult(x, res);
    /*
    Expbc(x, aux1);
    ComputeAntiDiffusiveFluxes(x, aux1, adf);
    aux1 += adf;
    
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

            fluxEval_i.Mult(cij, flux_i);
            fluxEval_j.Mult(cij, flux_j);

            for(int n = 0; n < numVar; n++)
            {
                aux1(i + n * nDofs) += dij * (uj(n) - ui(n)) - ( flux_j(n) - flux_i(n) );
            }            
        }

        /*
        for(int n = 1; n < numVar; n++)
        {
            aux1(i + n * nDofs) -= sys->collision_coeff * ui(n) * lumpedMassMatrix(i);
        }
    }
    ML_inv.Mult(aux1, res);
    //*/
}