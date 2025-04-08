#ifndef MOMENTS_METHODS_MCLCOMP
#define MOMENTS_METHODS_MCLCOMP

#include "mcl.hpp"

class MCL_Comp : public MCL
{
    public:
        //mutable Vector uDof3, uDof4, uDof5, uDof6, uDiff;
        //mutable DenseMatrix umin_loc, umax_loc;
        //mutable Vector uDot, adf, uij, uji, max_scalar, min_scalar;
        //mutable Array <SparseMatrix*> fij_gl;
        //mutable Array <Vector*> umin, umax, uDot_gl;
        //mutable HypreParVector aux1_hpr;

        explicit MCL_Comp(ParFiniteElementSpace *fes_, ParFiniteElementSpace *vfes_, System *sys_, DofInfo &dofs_,Vector &lumpedMassMatrix_);

        virtual ~MCL_Comp() { };

        virtual void ComputeAntiDiffusiveFluxes(const Vector &x, const Vector &dbc, Vector &AntiDiffFluxes) const override;
        //virtual double CalcBarState_returndij(const int i, const int j_gl, Vector &uij, Vector &uji) const;
        //virtual void ComputeSteadyStateResidual_gf(const Vector &x, ParGridFunction &res) const;
        //virtual void CalcUdot(const Vector &x, const Vector &dbc) const;
        //virtual void Mult(const Vector &x, Vector &y) const;
        virtual void CalcMinMax(const Vector &x) const override;
};

#endif