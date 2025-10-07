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
        SparseMatrix massmatrix_ld;
        DenseMatrix normals;
        //int* I, *J;
        int *I_ld, *J_ld;
        HypreParMatrix* hpr_mass;
        SparseMatrix M_diag, M_offdiag;


        virtual void Extract_offd_hypre(mfem::HypreParMatrix *const mat, const mfem::Vector &x, mfem::Vector &y, const int offd_width);
        virtual HYPRE_Int Extract_offdiagonals(hypre_ParCSRMatrix *A, hypre_ParVector *x, hypre_ParVector *y);
        double GetElem(const int i, const int j, const SparseMatrix& mat) const;
        explicit DofInfo(ParFiniteElementSpace *fes_);
        virtual ~DofInfo();

    private:
        virtual void BuildNormals();
};

#endif