#ifndef MOMENTS_SYSTEM
#define MOMENTS_SYSTEM

#include "../auxiliaries/auxiliarieslib.hpp"

struct SystemConfiguration
{
   int benchmark;
};

class System
{
    public: 
        explicit System(ParFiniteElementSpace *fes_, BlockVector &ublock, int numVar_, SystemConfiguration &config_, VectorFunctionCoefficient bdrCond_);

        virtual ~System();

        virtual void EvaluateFlux(const Vector &u, DenseMatrix &fluxEval) const = 0;
        virtual void EvaluateFluxJacobian(const Vector &u, DenseTensor &fluxJac) const = 0;
        virtual void EntropyVariable(const Vector &u, Vector &v) const {v = u; }
        virtual void EntropyPotential(const Vector &u, Vector &psi) const  { }
        virtual void EntropyFlux(const Vector &u, Vector &q) const  { };
        virtual double GetWaveSpeed(const Vector &u, const Vector n) const = 0;
        virtual bool Admissible(const Vector &u) const {MFEM_ABORT(""); return false;};
        virtual bool GloballyAdmissible(const Vector &x) const;
        virtual void ComputeDerivedQuantities(const Vector &u, ParGridFunction &d1) const {};
        virtual void SetBoundaryConditions(const Vector &y1, Vector &y2, const Vector &normal, int attr) const = 0;
        virtual double ComputeLambdaij(const Vector &n, const Vector &u1, const Vector &u2) const;
        virtual double ComputeDiffusion(const Vector &cij, const Vector &cji, const Vector &u1, const Vector &u2) const;
        virtual void ComputeLaxFriedrichsFlux(const Vector &x1, const Vector &x2, const Vector &normal, Vector &y) const;
        virtual double GetIntWaveSpeed(const Vector &u, const Vector &x) const {MFEM_ABORT("Not implemented for this system!") return 0.0;};
        virtual void ComputeErrors(Array<double> &errors, const ParGridFunction &u, double domainSize) const {};
        virtual void WriteErrors(const Array<double> &errors) const;
        virtual double Entropy(const Vector &u) const {MFEM_ABORT("No Entropy for this system"); return 0.0; };
        virtual void CutOff(Vector &x) const;
        virtual void Adjust(Vector &u) const = 0;
        virtual double ComputePressureLikeVariable(Vector &u) const = 0;

        const int numVar, dim;
        ParFiniteElementSpace *vfes;
        ParGridFunction u0;
        bool solutionKnown;

        FunctionCoefficient *Sigma_0, *Sigma_1;
        VectorFunctionCoefficient *q;
        mutable VectorFunctionCoefficient bdrCond;
        mutable DenseMatrix flux1, flux2;
        //mutable DenseTensor fluxJac1, fluxJac2;
        mutable Vector ui;
        const char* problemName;
        double collision_coeff = 0.0;
};

#endif