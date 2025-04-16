#include "mcl_comp.hpp"

// PositivityFix = 1 -> true
#define PositivityFix 1

MCL_Comp::MCL_Comp(ParFiniteElementSpace *fes_, ParFiniteElementSpace *vfes_, System *sys_, DofInfo &dofs_,Vector &lumpedMassMatrix_) : 
        MCL(fes_, vfes_, sys_, dofs_, lumpedMassMatrix_)
{ }


void MCL_Comp::CalcMinMax(const Vector &x) const
{   
    umin_loc = 0.0; umax_loc = 0.0;

    auto I = dofs.I;
    auto J = dofs.J;
    
    for(int i = 0; i < nDofs; i++)
    {
        int i_td = fes->GetLocalTDofNumber(i);
        if(i_td == -1 ){continue;}
        int i_gl = fes->GetGlobalTDofNumber(i);

        for(int n = 0; n < numVar; n++)
        {
            umin_loc(i_td, n) = x(i + n * nDofs);
            umax_loc(i_td, n) = umin_loc(i_td, n);
        }

        for(int k = I[i_td]; k < I[i_td+1]; k++)
        {
            int j_gl = J[k];
            if(i_gl == j_gl){continue;}

            double dij = CalcBarState_returndij(i, j_gl , uij, uji);

            for(int n = 0; n < numVar; n++)
            {
                umin_loc(i_td, n) = min(umin_loc(i_td, n), x_gl[n]->Elem(j_gl));
                umax_loc(i_td, n) = max(umax_loc(i_td, n), x_gl[n]->Elem(j_gl));          

                umin_loc(i_td, n) = min(umin_loc(i_td, n), uij(n));
                umax_loc(i_td, n) = max(umax_loc(i_td, n), uij(n)); 
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
        
        for(int i_td = 0; i_td < TDnDofs; i_td++)
        {        
            aux_hpr(i_td) = min_scalar(i_td);
            aux1_hpr(i_td) = max_scalar(i_td);
         
        }

        umin[n] = aux_hpr.GlobalVector();
        umax[n] = aux1_hpr.GlobalVector();
    }   
}




void MCL_Comp::ComputeAntiDiffusiveFluxes(const Vector &x, const Vector &dbc, Vector &AntiDiffFluxes) const
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
            
            for(int n = 0; n < numVar; n++)
            {
                double mij_sigma = (n == 0) * M_sigma_a(i_td, j_gl) + (n > 0) * M_sigma_aps(i_td, j_gl);
                const double fij_ = dofs.massmatrix(i_td, j_gl) * (uDot_gl[n]->Elem(i_gl) - uDot_gl[n]->Elem(j_gl)) + (dij + mij_sigma) * (x_gl[n]->Elem(i_gl) - x_gl[n]->Elem(j_gl));
                double fij_star;
            
                if( fij_> 0)
                {
                    double fij_min = 2.0 * dij * (umax[n]->Elem(i_gl) - uij(n));
                    double fij_max = 2.0 * dij* (uji(n) - umin[n]->Elem(j_gl));
                    double fij_bound = min(fij_max, fij_min);
                
                    fij_star = min(fij_, fij_bound);
                    fij_star = max(0.0, fij_star);
                }
                else
                {
                    double fij_min = 2.0 * dij * (umin[n]->Elem(i_gl) - uij(n));
                    double fij_max = 2.0 * dij* (uji(n) - umax[n]->Elem(j_gl));
                    double fij_bound = max(fij_max, fij_min);

                    fij_star = max(fij_, fij_bound); 
                    fij_star= min(0.0 , fij_star);
                }

                MFEM_ASSERT(abs(fij_star) <= abs(fij_) + 1e-15, "a_ij density wrong!");
            
                fij_gl[n]->Elem(i_td, j_gl) = fij_star;
            }
        }
    }

    // positivity fix
    #if PositivityFix == 1
        int counter = 0;

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
                
                double psi1_ij_sq = 0.0;
                double psi1_ji_sq = 0.0;

                double f1_ij_sq = 0.0;

                double f1p1_ij = 0.0;
                double f1p1_ji = 0.0;

                for(int d = 0; d < dim; d++)
                {
                    f1_ij_sq += fij_gl[d+1]->Elem(i_td, j_gl) * fij_gl[d+1]->Elem(i_td, j_gl);

                    psi1_ij_sq += uij(d+1) * uij(d+1);
                    psi1_ji_sq += uji(d+1) * uji(d+1);
                    
                    f1p1_ij += uij(d+1) * fij_gl[d+1]->Elem(i_td, j_gl);
                    f1p1_ji += - fij_gl[d+1]->Elem(i_td, j_gl) * uji(d+1);

                }

                double f1_ji_sq = f1_ij_sq; // fij = - fji => fij^2 = fji^2
                double f0_ij = fij_gl[0]->Elem(i_td, j_gl);
                double f0_ji = - f0_ij;

                double Qij = 4.0 * dij * dij * (uij(0) * uij(0) - psi1_ij_sq);
                double Qji = 4.0 * dij * dij * (uji(0) * uji(0) - psi1_ji_sq);

                double Rij = max(0.0 , f1_ij_sq - f0_ij * f0_ij) + 4.0 * dij * ( f1p1_ij - f0_ij * uij(0));
                double Rji = max(0.0 , f1_ji_sq - f0_ji * f0_ji) + 4.0 * dij * ( f1p1_ji - f0_ji * uji(0));

                double aij = Qij / Rij;
                double aji = Qji / Rji;
                
                double alpha = 1.0;
                if(Rij > Qij && Rji > Qji)
                {
                    alpha = min(aij, aji);
                }
                else if( Rij > Qij && Rji <= Qji)
                {
                    alpha = aij;
                }
                else if( Rij <= Qij && Rji > Qji)
                {
                    alpha = aji;
                }
                
                if(alpha < 1.0 - 1e-12)
                {
                    //cout << alpha << endl;
                    counter++;
                }


                for(int n = 0; n < numVar; n++)
                {
                    fij_gl[n]->Elem(i_td, j_gl) *= alpha;

                    uij(n) += 0.5 * fij_gl[n]->Elem(i_td, j_gl) / dij;  
                    uji(n) -= 0.5 * fij_gl[n]->Elem(i_td, j_gl) / dij;
                }
                
                MFEM_VERIFY(sys->Admissible(uij) && sys->Admissible(uji), "PA barstates not PA!");

            }
        }
        //*/

        //cout << counter << " / " << nDofs << endl;
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
}
