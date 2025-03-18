#include "mcl.hpp"

// PositivityFix = 1 -> true
#define PositivityFix 1

MCL::MCL(ParFiniteElementSpace *fes_, ParFiniteElementSpace *vfes_, System *sys_, DofInfo &dofs_,Vector &lumpedMassMatrix_) : 
        FEM(fes_, vfes_, sys_, dofs_, lumpedMassMatrix_), fij(numVar * nDofs, numVar * nDofs), BarStates(numVar * nDofs, numVar * nDofs), dij_mat(nDofs, nDofs), 
        Phi_states((numVar - 1) * nDofs, (numVar - 1) * nDofs), rho_ij_star(nDofs, nDofs), umin(nDofs, numVar), umax(nDofs, numVar)
{
    /*
    targetScheme = false; 

    if(!sys->steadyState || numVar == 1 || sys->solutionKnown)
    {
        targetScheme = true;
    }

    if(!targetScheme)
    {   
        cout << "\n" << "-----------------------------------------------------------------" << endl;
        cout << "Loworder initialization on!" << "\n";
        cout << "-----------------------------------------------------------------" << "\n\n";
    }  
    //*/

    uDot.SetSize(numVar * nDofs);
    uDot.UseDevice(true); 

    uDof3.SetSize(numVar);
    uDof4.SetSize(numVar);
    uDof5.SetSize(numVar);
    uDof6.SetSize(numVar);
    uDiff.SetSize(numVar);

    BilinearForm Mass(fes);
    Mass.AddDomainIntegrator(new MassIntegrator());
    Mass.Assemble();
    Mass.Finalize(0);
    MassMatrix = Mass.SpMat();

    auto I = dofs.I;
    auto J = dofs.J;
    //*
    for(int i = 0; i < nDofs; i++)
    {
        for(int k = I[i]; k < I[i+1]; k++)
        {
            int j = J[k];
            for(int n = 0; n < numVar; n++)
            {
                for(int m = 0; m < numVar; m++)
                {
                    fij.Set(i + n * nDofs, j + m * nDofs, 1.0);
                }
            }
        }
    }
    fij.Finalize(0);
    fij = 0.0;

    for(int i = 0; i < nDofs; i++)
    {
        for(int k = I[i]; k < I[i+1]; k++)
        {
            int j = J[k];
            if( i == j){continue;}
            dij_mat.Set(i, j, 1.0);
            rho_ij_star.Set(i, j, 1.0);

            for(int n = 0; n < numVar; n++)
            {
                for(int m = 0; m < numVar; m++)
                {
                    BarStates.Set(i + n * nDofs, j + m * nDofs, 1.0);
                }
            }
        }
    }

    BarStates.Finalize(0);
    BarStates = 0.0;
    dij_mat.Finalize(0);
    dij_mat = 0.0;
    rho_ij_star.Finalize(0);
    rho_ij_star = 0.0;

    for(int i = 0; i < nDofs; i++)
    {   
        for(int k = I[i]; k < I[i+1]; k++)
        {   
            int j = J[k];
            if(j >= i)
            {
                continue;
            }

            for(int n = 1; n < numVar; n++)
            { 
                Phi_states.Set( i + (n-1) * nDofs, j +(n-1) * nDofs, 1.0);
                Phi_states.Set( j + (n-1) * nDofs, i +(n-1) * nDofs, 1.0);
            }
        }
    }  
    Phi_states.Finalize(0);

    //*/ 
}

void MCL::CalcMinMax(const Vector &x, const SparseMatrix &BarStates, const SparseMatrix &Phi_states, DenseMatrix &umin, DenseMatrix &umax) const
{
    umin = 0.0; umax = 0.0;

    auto I = dofs.I;
    auto J = dofs.J;
    
    for(int i = 0; i < nDofs; i++)
    {
        umin(i,0) = x(i);
        umax(i,0) = x(i);

        for(int n = 1; n < numVar; n++)
        {
            umin(i, n) = x(i+ n * nDofs) / x(i);
            umax(i, n) = umin(i, n);
        }

        for(int k = I[i]; k < I[i+1]; k++)
        {
            int j = J[k];
            if(i == j){continue;}
            
            umin(i, 0) = min(umin(i, 0), x(j));
            umax(i, 0) = max(umax(i, 0), x(j));

            umin(i, 0) = min(umin(i, 0), BarStates(i,j));
            umax(i, 0) = max(umax(i, 0), BarStates(i,j));

            for(int n = 1; n < numVar; n++)
            {
                double vj = x(j + n * nDofs) / x(j);
                umin(i, n) = min(umin(i, n), vj);
                umax(i, n) = max(umax(i, n), vj);          

                double vij = Phi_states(i + (n-1) * nDofs, j + (n-1) * nDofs);
                umin(i, n) = min(umin(i, n), vij);
                umax(i, n) = max(umax(i, n), vij); 
            }
        }
    }
}


void MCL::AssembleSystem(const Vector &Mx_n, const Vector &x, SparseMatrix &S, Vector &b, const double dt) const
{   
    MFEM_ASSERT(S.Finalized(), "Matrix not finalized");
    S = 0.0;
    mD = S;
    b = Mx_n;
    
    Assemble_A(x, S);
    Assemble_minusD(x, mD);
    S += mD;
    
    Impbc(x, S, dbc);
    ComputeAntiDiffusiveFluxes(x, &S, dbc, adf);
    b.Add(dt, dbc);
    b.Add(dt, adf);
    
    S *= dt;
    S += ML;

    //*
    for(int i = 0; i < nDofs; i++)
    {
        for(int d = 0; d < dim; d++)
        {
            S(i + (d+1) * nDofs, i + (d+1) * nDofs) += sys->collision_coeff * dt * lumpedMassMatrix(i);
        }
    }
    //*/
}


void MCL::CalcBarState(const Vector &x, SparseMatrix &BarStates, SparseMatrix &Phi_states, SparseMatrix &dij_mat) const
{
    BarStates = 0.0;
    dij_mat = 0.0;
    Phi_states = 0.0;
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
            if(j >= i )
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
            dij_mat(i, j) = dij;
            dij_mat(j, i) = dij;

            sys->EvaluateFlux(ui, fluxEval_i);
            sys->EvaluateFlux(uj, fluxEval_j);

            fluxEval_i.Mult(cij, flux_i);
            fluxEval_j.Mult(cij, flux_j);

            for(int n = 0; n < numVar; n++)
            {   
                BarStates(i + n * nDofs, j + n * nDofs) = 0.5* ((ui(n) + uj(n)) - (flux_j(n) - flux_i(n)) / dij ) ;
            }

            fluxEval_i.Mult(cji, flux_i);
            fluxEval_j.Mult(cji, flux_j);

            for(int n = 0; n < numVar; n++)
            {   
                BarStates(j + n * nDofs, i + n * nDofs) = 0.5* ((ui(n) + uj(n)) - (flux_i(n) - flux_j(n)) / dij ) ;
            }
            
        }
    }

    for(int i = 0; i < nDofs; i++)
    {   
        for(int k = I[i]; k < I[i+1]; k++)
        {   
            int j = J[k];
            if(j >= i)
            {
                continue;
            }

            for(int n = 1; n < numVar; n++)
            { 
                double phi_ij = (BarStates(i + n * nDofs, j + n * nDofs) + BarStates(j + n * nDofs, i + n * nDofs)) /  (BarStates(i, j) + BarStates(j, i) );
                Phi_states( i + (n-1) * nDofs, j +(n-1) * nDofs) =  phi_ij;
                Phi_states( j + (n-1) * nDofs, i +(n-1) * nDofs) =  phi_ij;
            }
        }
    }  
}



void MCL::ComputeAntiDiffusiveFluxes(const Vector &x, const SparseMatrix *A, const Vector &dbc, Vector &AntiDiffFluxes) const
{   
    AntiDiffFluxes = 0.0;
    if(!targetScheme)
    {
        //AntiDiffFluxes = 0.0;
        return;
    }

    if(A)
    {
        MFEM_ASSERT(A->Size() == numVar * nDofs, "Matrix dimensions wrong!");
        A->Mult(x, uDot);
        uDot *= -1.0;
        uDot += dbc;
        aux1 = uDot;
        ML_inv.Mult(aux1, uDot);
    }
    else
    {
        CalcUdot(x, dbc, uDot);
    }
    
    CalcBarState(x, BarStates, Phi_states, dij_mat);
    CalcMinMax(x, BarStates, Phi_states, umin, umax);
    auto I = dofs.I;
    auto J = dofs.J;

    fij = 0.0;
    rho_ij_star = 0.0;
    for(int i = 0; i < nDofs; i++)
    {           
        for(int k = I[i]; k < I[i+1]; k++)
        {
            int j = J[k];
            if(j == i)
            {
                continue;
            }

            double uij = BarStates(i, j) ; 
            double uji = BarStates(j, i ); 

            const double dij = dij_mat(i, j);
             
            const double fij_ = MassMatrix(i, j) * (uDot(i) - uDot(j)) + dij * (x(i) - x(j));
            double fij_star;
            
            if( fij_> 0)
            {
                double fij_min = 2.0 * dij * (umax(i,0) - uij);
                double fij_max = 2.0 * dij* (uji - umin(j,0));
                double fij_bound = min(fij_max, fij_min);

                fij_star = min(fij_, fij_bound);
                fij_star = max(0.0, fij_star);
            }
            else
            {
                double fij_min = 2.0 * dij * (umin(i,0) - uij);
                double fij_max = 2.0 * dij* (uji - umax(j,0));
                double fij_bound = max(fij_max, fij_min);

                fij_star = max(fij_, fij_bound); 
                fij_star= min(0.0 , fij_star);
            }

            //*
            if(abs(x(i) - x(j)) < 1e-14)
            {
                fij_star = 0.0;
            }
            //*/

            MFEM_ASSERT(abs(fij_star) <= abs(fij_) + 1e-15, "a_ij density wrong!");
            
            fij(i, j) = fij_star;
            rho_ij_star(i, j) =  uij + 0.5 * fij_star / dij;
        }
    }

     
    // limiting the conserved products

    //*

    for(int i = 0; i < nDofs; i++)
    {

        for(int n = 1; n < numVar; n++)
        {
            for(int k = I[i]; k < I[i+1]; k++)
            {   
                int j = J[k];
                if(j == i){continue;}

                const double uij = BarStates(i + n * nDofs, j + n * nDofs); 
                const double uji = BarStates(j + n * nDofs, i + n * nDofs); 

                const double phi_ij = Phi_states(i + (n-1) * nDofs, j + (n-1) * nDofs);
                const double phi_ji = Phi_states(j + (n-1) * nDofs, i + (n-1) * nDofs);

                const double dij = dij_mat(i, j);
                const double rij = rho_ij_star(i, j);
                const double rji = rho_ij_star(j, i);

                const double fij_ = MassMatrix(i, j) * (  (uDot(i + n * nDofs) - uDot(j + n * nDofs)) +  sys->collision_coeff * (x(i + n * nDofs) - x(j + n * nDofs))) + dij * (x(i + n * nDofs) - x(j + n * nDofs));  
                const double gij_ = fij_ + 2.0 * dij * (uij - rij * phi_ij) ;

                double gij_star;
                if(gij_ > 0.0)
                {   
                    const double gij_max = 2.0 * dij * min(rij * (umax(i, n) - phi_ij), rji * (phi_ji - umin(j, n)));
                    gij_star = max(0.0, min(gij_, gij_max));
                }
                else 
                {
                    const double gij_min =  2.0 * dij * max(rij * (umin(i, n) - phi_ij) , rji * (phi_ji - umax(j, n)));
                    gij_star = min(0.0, max(gij_, gij_min));
                }

                MFEM_ASSERT(abs(gij_star) < abs(gij_) + 1e-15, "a_ij rhophi wrong!");                
                fij(i + n * nDofs, j +  n * nDofs) =  gij_star - 2.0 * dij * (uij - rij * phi_ij);
            }
        }
    }
    //*/

    // positivity fix
    #if PositivityFix == 1
        for(int i = 0; i < nDofs; i++)
        {
            for(int k = I[i]; k < I[i+1]; k++)
            {
                int j = J[k];
                if(j >= i){continue;}

                for(int n = 0; n < numVar; n++)
                {
                    ui(n) = 0.5 * ( BarStates(i + n * nDofs, j + n * nDofs) + fij(i + n * nDofs, j +  n * nDofs) ) / dij_mat(i,j);
                    uj(n) = 0.5 * ( BarStates(j + n * nDofs, i + n * nDofs) + fij(j + n * nDofs, i +  n * nDofs) ) / dij_mat(j,i);
                }
                
                if(sys->ComputePressureLikeVariable(ui) < 0.0 || sys->ComputePressureLikeVariable(uj) < 0.0)
                {
                    for(int n = 0; n < numVar; n++)
                    {
                        fij(i + n * nDofs, j +  n * nDofs) = 0.0;
                        fij(j + n * nDofs, i +  n * nDofs) = 0.0;
                    }
                }
            }
        }
    #endif  

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
}

void MCL::CalcUdot(const Vector &x, const Vector &dbc, Vector &uDot) const
{   
    aux1 = dbc;

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
            int j = J[i];
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

            Vector cijfi(numVar), cijfj(numVar);

            fluxEval_i.Mult(cij, flux_i);
            fluxEval_j.Mult(cij, flux_j);

            for(int n = 0; n < numVar; n++)
            {
                aux1(i + n * nDofs) +=  (dij * (uj(n) - ui(n)) - ( flux_j(n) - flux_i(n) ));
            }            
        }

        for(int n = 1; n < numVar; n++)
        {
            aux1(i + n * nDofs) -= sys->collision_coeff * ui(n) * lumpedMassMatrix(i);
        }
    }

    ML_inv.Mult(aux1, uDot);
}


void MCL::ComputeSteadyStateResidual_gf(const Vector &x, ParGridFunction &res) const
{   
    Expbc(x, aux1);
    ComputeAntiDiffusiveFluxes(x, NULL, aux1, adf);
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

        for(int n = 1; n < numVar; n++)
        {
            aux1(i + n * nDofs) -= sys->collision_coeff * ui(n) * lumpedMassMatrix(i);
        }
    }
    ML_inv.Mult(aux1, res);
}