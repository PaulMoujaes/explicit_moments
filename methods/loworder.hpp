#ifndef MOMENTS_METHODS_LOWORDER
#define MOMENTS_METHODS_LOWORDER

#include "feevol.hpp"

class LowOrder : public feevol
{
    public:
        explicit LowOrder(ParFiniteElementSpace *fes_, ParFiniteElementSpace *vfes_, System *sys_, DofInfo &dofs_,Vector &lumpedMassMatrix_);
        virtual ~LowOrder() { };
        virtual void AssembleSystem(const Vector &Mx_n, const Vector &x, SparseMatrix &S, Vector &b, const double dt) const;
        virtual void ComputeSteadyStateResidual_gf(const Vector &x, ParGridFunction &res) const;
};

#endif