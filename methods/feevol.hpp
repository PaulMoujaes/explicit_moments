#ifndef MOMENTS_FEEVOL
#define MOMENTS_FEEVOL

#include "../systems/systemslib.hpp"
#include "../auxiliaries/auxiliarieslib.hpp"

class FE_Evolution : public TimeDependentOperator
{
    public: 
        // General member variables
        ParFiniteElementSpace *fes;
        ParFiniteElementSpace *vfes;
        GroupCommunicator &gcomm;
        GroupCommunicator &vgcomm;
        ParMesh *pmesh;
        ParGridFunction inflow;
        System *sys;
        //mutable double t;
        int intorder;
        //mutable bool targetScheme;

        //mutable Array<double> sync, vsync;

        // Parameters that are needed repeatedly
        const int dim, numVar, nDofs, nE;

        SparseMatrix shared;

        // Stuff for implicit fixed-point iteration
        //mutable SparseMatrix mD, T; // mD;
        //mutable DenseMatrix Block_ij, Identity_numVar;
        mutable Vector dbc, adf, aux1, aux2;
        mutable Vector ui, uj, vi, vj;
        //mutable DenseTensor fluxJac_i, fluxJac_j;
        mutable DenseMatrix fluxEval_i, fluxEval_j;
        mutable Vector flux_i, flux_j;
        mutable Vector cij, cji;
        mutable Vector uOld;
        mutable double steadyStateResidual;
        mutable ParGridFunction res_gf;
        DofInfo &dofs;
        mutable Array <int> vdofs;

        //SparseMatrix ML, ML_inv;

        const Vector lumpedMassMatrix;
        Vector lumpedMassMatrix_synced;

        Vector lumpedM_oa, lumpedM_os;
        SparseMatrix Convection, M_oa, M_os;

        FE_Evolution(ParFiniteElementSpace *fes_, ParFiniteElementSpace *vfes, System *sys_, DofInfo &dofs_,Vector &lumpedMassMatrix_);
        virtual ~FE_Evolution() { } 

        virtual void ComputeLOTimeDerivatives(const Vector &x, Vector &y) const;
        virtual void Mult(const Vector &x, Vector &y) const = 0;
        virtual double ComputeSteadyStateResidual(const Vector &x, ParGridFunction &res_gf) const;
        virtual void ComputeSteadyStateResidual_gf(const Vector &x, ParGridFunction &res_gf) const = 0;
        virtual double Compute_dt(const Vector &x, const double CFL) const;
        virtual void Expbc(const Vector &x, Vector &bc) const;
        virtual double SteadyStateCheck(const Vector &u) const;
        virtual void SyncVector(Vector &x) const;
        virtual void VSyncVector(Vector &x) const;
        
        /*
        virtual double ComputeSteadyStateResidual(const Vector &x, ParGridFunction &res_gf) const;
        virtual void ComputeSteadyStateResidual_gf(const Vector &x, ParGridFunction &res_gf) const = 0;
        virtual double ComputeEntropySteadyStateResidual(const Vector &x, ParGridFunction &res_gf) const;
        virtual void Expbc(const Vector &x, Vector &bc) const;
        virtual void Impbc(const Vector &x, SparseMatrix &bc, Vector &dbc) const;
        virtual double SteadyStateCheck(const Vector &u) const;
        virtual void SyncVector(Vector &x) const;
        virtual void VSyncVector(Vector &x) const;
        //*/
};

#endif
