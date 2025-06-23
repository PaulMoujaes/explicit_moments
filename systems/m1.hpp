#ifndef MOMENTS_M1
#define MOMENTS_M1

#include "system.hpp"

class M1 : public System
{
public:


   explicit M1(ParFiniteElementSpace *vfes_, BlockVector &ublock, SystemConfiguration &config_);
   ~M1() { };

   virtual void EvaluateFlux(const Vector &u, DenseMatrix &fluxEval) const;
   virtual void EvaluateFluxJacobian(const Vector &u, DenseTensor &fluxJac) const;
   virtual double GetWaveSpeed(const Vector &u, const Vector n) const;
   virtual void EntropyVariable(const Vector &u, Vector &v) const override;
   virtual void EntropyPotential(const Vector &u, Vector &psi) const override;
   virtual void SetBoundaryConditions(const Vector &y1, Vector &y2, const Vector &normal, int attr) const;
   virtual void ComputeErrors(Array<double> & errors, const ParGridFunction &u, double domainSize, const double t) const override;
   virtual bool Admissible(const Vector &u) const override;
   //virtual bool GloballyAdmissible(const Vector &x) const override;
   virtual void ComputeDerivedQuantities(const Vector &u, ParGridFunction &f) const override;
   virtual void EvaluatePsi2(const Vector &u, DenseMatrix &psi2) const;
   virtual double EvaluateEddingtonFactor(const Vector &u) const;
   virtual double EvaluateDerivativeEddingtonFactor(const Vector &u) const;
   virtual double Entropy(const Vector &u) const override;
   virtual void EntropyFlux(const Vector &u, Vector &q) const override;
   virtual double Evaluate_f(const Vector &u) const;
   virtual void Evaluate_f_vec(const Vector &u, Vector &v) const;
   virtual void Adjust(Vector &u) const;
   virtual double ComputePressureLikeVariable(Vector &u) const;
   //double ComputeLambdaij(const Vector &n, const Vector &u1, const Vector &u2) const;

   mutable DenseMatrix psi2;
};

#endif