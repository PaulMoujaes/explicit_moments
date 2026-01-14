#include "mcl.hpp"

// PositivityFix = 1 -> true
#define PositivityFix 1

MCL::MCL(ParFiniteElementSpace *fes_, ParFiniteElementSpace *vfes_, System *sys_, DofInfo &dofs_,Vector &lumpedMassMatrix_) : 
        FE_Evolution(fes_, vfes_, sys_, dofs_, lumpedMassMatrix_),
        umin_loc(TDnDofs, numVar), umax_loc(TDnDofs, numVar), min_scalar(TDnDofs), max_scalar(TDnDofs),
        fij_gl(numVar), uij(numVar), uji(numVar), umin(numVar), umax(numVar), aux1_hpr(fes), uDot_gl(numVar), adf_td(TDnDofs * numVar), fij_vec(numVar)
{
    //uDot.UseDevice(true);
    adf_td.UseDevice(true);
    adf_td = 0.0;

    umin = NULL;
    umax = NULL;
    uDot_gl = NULL;

    uDot_td.Update(offsets_diag);
    uDot_td.UseDevice(true);
    uDot_od.Update(offsets_offdiag);
    uDot_od.UseDevice(true);

    umin_td.Update(offsets_diag);
    umin_td.UseDevice(true);
    umin_od.Update(offsets_offdiag);
    umin_od.UseDevice(true);

    umax_td.Update(offsets_diag);
    umax_td.UseDevice(true);
    umax_od.Update(offsets_offdiag);
    umax_od.UseDevice(true);


    /*
    auto I = dofs.I;
    auto J = dofs.J;

    // for matrices where we need the full stencil cause non-symmetric
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
    //*/
}

void MCL::Mult(const Vector &x, Vector &y) const
{
    //MFEM_VERIFY(sys->GloballyAdmissible(x), "not IDP!");
    GetDiagOffDiagNodes(x, x_td, x_od);
    //rhs_td = 0.0;
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

    ComputeAntiDiffusiveFluxes(x_td, x_od, aux1, Source, adf_td);
    rhs_td = adf_td;
    //rhs_td = 0.0;

    // still ldofs, synced later
    aux1 += Source;

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
    
    VSyncVector(aux1);
    vfes->GetProlongationMatrix()->AddMult(rhs_td, aux1);
    One_over_MLpdtMLs.Mult(aux1, y);
    ML_over_MLpdtMLs_m1.AddMult(x, y, 1.0 / dt);
}


void MCL::CalcMinMax(const BlockVector &x_td, const BlockVector &x_od) const
{   
    MFEM_VERIFY(umin_td.Size() == numVar * TDnDofs, "wrong size min")
    MFEM_VERIFY(umax_td.Size() == numVar * TDnDofs, "wrong size max")

    for(int n = 0; n < numVar; n++)
    {
        MFEM_VERIFY(umin_td.GetBlock(n).Size() == TDnDofs, "wrong size min n")
        MFEM_VERIFY(umax_td.GetBlock(n).Size() == TDnDofs, "wrong size max n")
    }

    umin_td = x_td; 
    umax_td = x_td;

    auto I_diag = C_diag[0]->GetI();
    auto J_diag = C_diag[0]->GetJ();
    auto I_offdiag = C_offdiag[0]->GetI();
    auto J_offdiag = C_offdiag[0]->GetJ();
    
    for(int i = 0; i < TDnDofs; i++)
    {
        for(int n = 0; n < numVar; n++)
        {   
            ui(n) = x_td.GetBlock(n).Elem(i);
            //umin_td.GetBlock(n).Elem(i) = ui(n);
            //umax_td.GetBlock(n).Elem(i) = ui(n);
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

            double dij = sys->CalcBarState_returndij(ui, uj, cij, cji, uij, uji);

            for(int n = 0; n < numVar; n++)
            {      
                umin_td.GetBlock(n).Elem(i) = min(umin_td.GetBlock(n).Elem(i), uj(n));
                umax_td.GetBlock(n).Elem(i) = max(umax_td.GetBlock(n).Elem(i), uj(n));
                
                umin_td.GetBlock(n).Elem(i) = min(umin_td.GetBlock(n).Elem(i), uij(n));
                umax_td.GetBlock(n).Elem(i) = max(umax_td.GetBlock(n).Elem(i), uij(n));
            }
        }

        /*
        umin_td.GetBlock(0).Print();
        cout <<"--------------------------------__" << endl;
        umax_td.GetBlock(0).Print();
        MFEM_ABORT("")
        //*/

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

                double dij = sys->CalcBarState_returndij(ui, uj, cij, cji, uij, uji);

                for(int n = 0; n < numVar; n++)
                {       
                    umin_td.GetBlock(n).Elem(i) = min(umin_td.GetBlock(n).Elem(i), uj(n));
                    umax_td.GetBlock(n).Elem(i) = max(umax_td.GetBlock(n).Elem(i), uj(n));
                
                    umin_td.GetBlock(n).Elem(i) = min(umin_td.GetBlock(n).Elem(i), uij(n));
                    umax_td.GetBlock(n).Elem(i) = max(umax_td.GetBlock(n).Elem(i), uij(n)); 
                }
            }
        }
    } 

    GetOffDiagNodes(umin_td, umin_od);
    GetOffDiagNodes(umax_td, umax_od);
}




void MCL::ComputeAntiDiffusiveFluxes(const BlockVector &x_td, const BlockVector &x_od, const Vector &dbc, const Vector &Source, Vector &AntiDiffFluxes) const
{   
    MFEM_VERIFY(AntiDiffFluxes.Size() == TDnDofs * numVar, "wrong size adf")
    AntiDiffFluxes = 0.0;
    CalcUdot(x_td, x_od, dbc, Source);
    CalcMinMax(x_td, x_od);

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

            fij_vec = 0.0;

            for(int n = 0; n < numVar; n++)
            {   
                uj(n) = x_td.GetBlock(n).Elem(j);
            }

            for(int d = 0; d < dim; d++)
            {   
                cij(d) = C_diag[d]->GetData()[k];
                cji(d) = C_diag_T[d]->GetData()[k];
            }

            double dij = sys->CalcBarState_returndij(ui, uj, cij, cji, uij, uji);

            for(int n = 0; n < numVar; n++)
            {
                double mij_sigma = (n == 0) * M_sigma_a_diag.GetData()[k] + (n > 0) * M_sigma_aps_diag.GetData()[k];
                double mij = dofs.M_diag.GetData()[k];

                const double fij_ = mij * (uDot_td.GetBlock(n).Elem(i) - uDot_td.GetBlock(n).Elem(j)) + (dij + mij_sigma) * (ui(n) - uj(n));
                double fij_star;
            
                if( fij_> 0)
                {
                    double fij_min = umax_td.GetBlock(n).Elem(i) - uij(n);
                    double fij_max = uji(n) - umin_td.GetBlock(n).Elem(j);
                    double fij_bound = 2.0 * dij *  min(fij_max, fij_min);
                
                    fij_star = min(fij_, fij_bound);
                    fij_star = max(0.0, fij_star);
                }
                else
                {
                    double fij_min = umin_td.GetBlock(n).Elem(i) - uij(n);
                    double fij_max = uji(n) - umax_td.GetBlock(n).Elem(j);
                    double fij_bound = 2.0 * dij * max(fij_max, fij_min);

                    fij_star = max(fij_, fij_bound); 
                    fij_star= min(0.0 , fij_star);
                }

                MFEM_ASSERT(abs(fij_star) <= abs(fij_) + 1e-15, "a_ij density wrong!");
            
                fij_vec(n) = fij_star;
            }
            
            #if PositivityFix == 1
                IDPfix(uij, uji, dij, fij_vec);
            #endif

            for(int n = 0; n < numVar; n++)
            {
                AntiDiffFluxes(i + n * TDnDofs) += fij_vec(n);
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

                double dij = sys->CalcBarState_returndij(ui, uj, cij, cji, uij, uji);

                for(int n = 0; n < numVar; n++)
                {
                    double mij_sigma = (n == 0) * M_sigma_a_offdiag.GetData()[k] + (n > 0) * M_sigma_aps_offdiag.GetData()[k];
                    double mij = dofs.M_offdiag.GetData()[k];

                    const double fij_ = mij * (uDot_td.GetBlock(n).Elem(i) - uDot_od.GetBlock(n).Elem(j)) + (dij + mij_sigma) * (ui(n) - uj(n));
                    double fij_star;
            
                    if( fij_> 0)
                    {
                        double fij_min = umax_td.GetBlock(n).Elem(i) - uij(n);
                        double fij_max = uji(n) - umin_od.GetBlock(n).Elem(j);
                        double fij_bound = 2.0 * dij * min(fij_max, fij_min);
                
                        fij_star = min(fij_, fij_bound);
                        fij_star = max(0.0, fij_star);
                    }
                    else
                    {
                        double fij_min = umin_td.GetBlock(n).Elem(i) - uij(n);
                        double fij_max = uji(n) - umax_od.GetBlock(n).Elem(j);
                        double fij_bound = 2.0 * dij * max(fij_max, fij_min);

                        fij_star = max(fij_, fij_bound); 
                        fij_star = min(0.0 , fij_star);
                    }

                    MFEM_ASSERT(abs(fij_star) <= abs(fij_) + 1e-15, "a_ij density wrong!");
                    
                    fij_vec(n) = fij_star;
                }
            
                #if PositivityFix == 1
                    IDPfix(uij, uji, dij, fij_vec);
                #endif

                for(int n = 0; n < numVar; n++)
                {
                    AntiDiffFluxes(i + n * TDnDofs) += fij_vec(n);
                }
            }
        }
    }
}

void MCL::IDPfix(const Vector &uij, const Vector &uji, const double dij, Vector &fij_vec) const 
{
    double psi1_ij_sq = 0.0;
    double psi1_ji_sq = 0.0;

    double f1_ij_sq = 0.0;

    double f1p1_ij = 0.0;
    double f1p1_ji = 0.0;

    for(int d = 0; d < dim; d++)
    {
        f1_ij_sq += fij_vec(d+1) * fij_vec(d+1);

        psi1_ij_sq += uij(d+1) * uij(d+1);
        psi1_ji_sq += uji(d+1) * uji(d+1);
                    
        f1p1_ij += uij(d+1) * fij_vec(d+1);
        f1p1_ji += - fij_vec(d+1) * uji(d+1);
    }

    double f1_ji_sq = f1_ij_sq; // fij = - fji => fij^2 = fji^2
    double f0_ij = fij_vec(0);
    double f0_ji = - f0_ij;

    double Qij = 4.0 * dij * dij * (uij(0) * uij(0) - psi1_ij_sq);
    double Qji = 4.0 * dij * dij * (uji(0) * uji(0) - psi1_ji_sq);

    MFEM_VERIFY(Qij > 0.0, "Qij not positive! " + to_string(log(abs(Qij))));
    MFEM_VERIFY(Qji > 0.0, "Qji not positive! " + to_string(log(abs(Qji))));
                
    Qij = max(Qij, 0.0);
    Qji = max(Qji, 0.0);
                
    Qij *= 1.0 - 1e-13;
    Qji *= 1.0 - 1e-13;

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

    MFEM_VERIFY(alpha >= -1e-15 && alpha <= 1.0 +1e-15, "alpha_pa not in [0,1]!");
                
    alpha = max(0.0, alpha);
    alpha = min(1.0, alpha);
                
    fij_vec *= alpha;
    //fij_vec *= 0.9; // pfusch
    /*

    MFEM_VERIFY(sys->Admissible(uij), "fucked uij");
    MFEM_VERIFY(sys->Admissible(uji), "fucked uji");

    Vector uij_ = uij;
    Vector uji_ = uji;
    Vector fij_ = fij_vec;
    fij_ *= 0.5;
    fij_ /= dij;
    uij_ += fij_;
    uji_ -= fij_;
    MFEM_VERIFY(sys->Admissible(uij_), "IDP fix fucked uij");
    MFEM_VERIFY(sys->Admissible(uji_), "IDP fix fucked uji");
    //cout << "passt" << endl;
    //*/
}


void MCL::CalcUdot(const BlockVector &x_td, const BlockVector &x_od, const Vector &dbc, const Vector &Source) const
{   
    vfes->GetProlongationMatrix()->MultTranspose(dbc, uDot_td);
    vfes->GetProlongationMatrix()->AddMultTranspose(Source, uDot_td);

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
                uDot_td.GetBlock(n).Elem(i) += (dij * (uj(n) - ui(n)) - ( flux_j(n)));
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
                    uDot_td.GetBlock(n).Elem(i) += (dij * (uj(n) - ui(n)) - ( flux_j(n)));
                }
            }
        }


        for(int n = 0; n < numVar; n++)
        {
            uDot_td.GetBlock(n).Elem(i) -= (n == 0) * Mlumped_sigma_a_td(i) * ui(n) + (n > 0) * Mlumped_sigma_aps_td(i) * ui(n);
            uDot_td.GetBlock(n).Elem(i) /= lumpedMassMatrix_td(i);
        }
    }

    GetOffDiagNodes(uDot_td, uDot_od);
}


void MCL::ComputeSteadyStateResidual_gf(const Vector &x, ParGridFunction &res) const
{   
    //res = 0.0;
    //*
   //MFEM_VERIFY(sys->GloballyAdmissible(x), "not IDP!");
    GetDiagOffDiagNodes(x, x_td, x_od);
    //rhs_td = 0.0;


    //bdr condition with ldofs because bdr edges are not shared
    Expbc(x, aux1);

    ComputeAntiDiffusiveFluxes(x_td, x_od, aux1, Source, adf_td);
    rhs_td = adf_td;
    //rhs_td = 0.0;

    // still ldofs, synced later
    aux1 += Source;

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

        for(int n = 0; n < numVar; n++)
        {
            rhs_td(i + n * TDnDofs) -= (n == 0) * Mlumped_sigma_a_td(i) * ui(n) + (n > 0) * Mlumped_sigma_aps_td(i) * ui(n);
            //rhs_td(i + n * TDnDofs) /= lumpedMassMatrix_td(i);
        }
    }
    
    VSyncVector(aux1);
    vfes->GetProlongationMatrix()->AddMult(rhs_td, aux1);
    ML_inv.Mult(aux1, res);
    //*/
}