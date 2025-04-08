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
                const double fij_ = dofs.massmatrix(i_td, j_gl) * (uDot_gl[n]->Elem(i_gl) - uDot_gl[n]->Elem(j_gl)) + dij * (x_gl[n]->Elem(i_gl) - x_gl[n]->Elem(j_gl));
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
}
