#ifndef MOMENTS_METHODS_LOWORDER
#define MOMENTS_METHODS_LOWORDER

#include "feevol.hpp"

class LowOrder : public FE_Evolution
{
    public:
        explicit LowOrder(ParFiniteElementSpace *fes_, ParFiniteElementSpace *vfes_, System *sys_, DofInfo &dofs_,Vector &lumpedMassMatrix_);
        virtual ~LowOrder() { };

        virtual void Mult(const Vector &x, Vector &y) const override;
        virtual void ComputeSteadyStateResidual_gf(const Vector &x, ParGridFunction &res) const;
};

#endif