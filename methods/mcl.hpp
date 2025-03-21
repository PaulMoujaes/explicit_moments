#ifndef MOMENTS_METHODS_MCL
#define MOMENTS_METHODS_MCL

#include "feevol.hpp"

class MCL : public FE_Evolution
{
    public:
        //mutable Vector uDof3, uDof4, uDof5, uDof6, uDiff;
        SparseMatrix MassMatrix;
        mutable DenseMatrix umin, umax;
        mutable Vector uDot, adf;
        mutable SparseMatrix BarStates, fij, dij_mat, Phi_states, rho_ij_star;

        explicit MCL(ParFiniteElementSpace *fes_, ParFiniteElementSpace *vfes_, System *sys_, DofInfo &dofs_,Vector &lumpedMassMatrix_);

        virtual ~MCL() { };

        virtual void ComputeAntiDiffusiveFluxes(const Vector &x, const Vector &dbc, Vector &AntiDiffFluxes) const;
        virtual void CalcBarState(const Vector &x, SparseMatrix &BarStates, SparseMatrix &Phi_states, SparseMatrix &dij_mat) const;
        virtual void ComputeSteadyStateResidual_gf(const Vector &x, ParGridFunction &res) const;
        virtual void CalcUdot(const Vector &x, const Vector &dbc, Vector &uDot) const;
        virtual void Mult(const Vector &x, Vector &y) const;
        virtual void CalcMinMax(const Vector &x, const SparseMatrix &BarStates, const SparseMatrix &Phi_states, DenseMatrix &umin, DenseMatrix &umax) const;
};

#endif