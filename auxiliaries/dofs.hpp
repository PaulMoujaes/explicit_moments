#ifndef IMPCG_DOFS
#define IMPCG_DOFS

#include "mfem.hpp"
#include "mfem/general/forall.hpp"

using namespace std;
using namespace mfem;

// NOTE: The mesh is assumed to consist of segments, triangles, quads or hexes.
class DofInfo
{
    public: 
        ParMesh *pmesh;
        ParFiniteElementSpace *fes;
        int nDofs;
        const int dim;
        mutable Table *DofToDofTable, *ElementToBdrElementTable;
        SparseMatrix massmatrix;
        DenseMatrix normals;
        int* I;
        int* J;

        explicit DofInfo(ParFiniteElementSpace *fes_);
        virtual ~DofInfo();

    private:
        virtual void BuildDofToDofTable() const;
        virtual void BuildElementToBdrElementTable() const;
        virtual void BuildNormals();
};

#endif