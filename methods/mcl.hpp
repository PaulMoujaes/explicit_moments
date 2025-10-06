#ifndef MOMENTS_METHODS_MCL
#define MOMENTS_METHODS_MCL

#include "feevol.hpp"

class MCL : public FE_Evolution
{
    public:
        //mutable Vector uDof3, uDof4, uDof5, uDof6, uDiff;
        mutable DenseMatrix umin_loc, umax_loc;
        mutable Vector uij, uji, max_scalar, min_scalar;
        mutable Array <SparseMatrix*> fij_gl;
        mutable Array <Vector*> umin, umax, uDot_gl;
        mutable HypreParVector aux1_hpr;
        mutable BlockVector uDot_td, uDot_od;
        mutable BlockVector umin_td, umin_od;
        mutable BlockVector umax_td, umax_od;
        mutable Vector adf_td, fij_vec;


        explicit MCL(ParFiniteElementSpace *fes_, ParFiniteElementSpace *vfes_, System *sys_, DofInfo &dofs_,Vector &lumpedMassMatrix_);

        virtual ~MCL() { };

        virtual void ComputeAntiDiffusiveFluxes(const BlockVector &x_td, const BlockVector &x_od, const Vector &dbc, const Vector &Source, Vector &AntiDiffFluxes_td) const;
        virtual void ComputeSteadyStateResidual_gf(const Vector &x, ParGridFunction &res) const;
        virtual void CalcUdot(const BlockVector &x_td, const BlockVector &x_od, const Vector &dbc, const Vector &Source) const;
        virtual void Mult(const Vector &x, Vector &y) const;
        virtual void CalcMinMax(const BlockVector &x_td, const BlockVector &x_od) const;
};

#endif