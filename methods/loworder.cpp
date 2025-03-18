#include "loworder.hpp"

LowOrder::LowOrder(ParFiniteElementSpace *fes_, ParFiniteElementSpace *vfes_, System *sys_, DofInfo &dofs_,Vector &lumpedMassMatrix_) : 
        FE_Evolution(fes_, vfes_, sys_, dofs_, lumpedMassMatrix_)
{ }

/*
void LowOrder::CalcResidual(const Vector &Mx_n, const Vector &x, const double dt, Vector &res) const
{
    res = Mx_n;
    ML.AddMult(x, res, -1.0);

    Expbc(x, dbc);
    res.Add(dt, dbc);

    for(int i = 0; i < nDofs; i++)
    {
        for(int n = 0; n < numVar; n++)
        {
            ui(n) = x(i + n * nDofs);
        }

        Array <int> row;
        dofs.DofToDofTable->GetRow(i,row);

        for(int j = 0; j < row.Size(); j++)
        {
            if(row[j] == i)
            {
                continue;
            }

            for(int n = 0; n < numVar; n++)
            {
                uj(n) = x(row[j] + n * nDofs);
            }

            for(int d = 0; d < dim; d++)
            {
                cij(d) = Convection(i, row[j] + d * nDofs);
                cji(d) = Convection(row[j], i + d * nDofs);
            }

            double dij = sys->ComputeDiffusion(cij, cji, ui, uj, i, row[j]);

            sys->EvaluateFlux(ui, fluxEval_i, i);
            sys->EvaluateFlux(uj, fluxEval_j, row[j]);

            fluxEval_j -= fluxEval_i;
            fluxEval_j.Mult(cij, flux_j);

            for(int n = 0; n < numVar; n++)
            {
                res(i + n * nDofs) += dt * (dij * (uj(n) - ui(n)) - ( flux_j(n)) );
            }            
        }
    }
}
//*/

void LowOrder::Mult(const Vector &x, Vector &y) const
{
    ComputeLOTimeDerivatives(x, y);
}



void LowOrder::ComputeSteadyStateResidual_gf(const Vector &x, ParGridFunction &res) const 
{   
    Expbc(x, res);

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
                res(i + n * nDofs) += (dij * (uj(n) - ui(n)) - ( flux_j(n)));
            }
        }

        /*
        for(int d = 0; d < dim; d++)
        {
            res(i + (d+1) * nDofs) -= sys->collision_coeff * lumpedMassMatrix(i) * ui(d+1);
        }
        //*/ 
    }

    for(int i = 0; i < nDofs; i++)
    {
        for(int n = 0; n < numVar; n++)
        {
            res(i + n * nDofs) /= lumpedMassMatrix(i);
        }
    }
}
