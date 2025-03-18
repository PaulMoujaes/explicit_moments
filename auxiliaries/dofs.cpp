#include "dofs.hpp"

DofInfo::DofInfo(ParFiniteElementSpace *fes_): 
    pmesh(fes_->GetParMesh()), fes(fes_), dim(fes_->GetParMesh()->Dimension()), nDofs(fes->GetNDofs()), normals(fes->GetNBE(), dim), ElementToBdrElementTable(NULL), DofToDofTable(NULL)
{
    ParBilinearForm M(fes);
    M.AddDomainIntegrator(new MassIntegrator());
    M.Assemble();
    M.Finalize();
    massmatrix = M.SpMat();

    I = massmatrix.GetI();
    J = massmatrix.GetJ();

    //*
    BuildDofToDofTable();
    BuildElementToBdrElementTable();
    //*/

    BuildNormals();
}

DofInfo::~DofInfo() 
{   
    if(DofToDofTable){delete DofToDofTable;}
    if(ElementToBdrElementTable){delete ElementToBdrElementTable;}
}

void DofInfo::BuildNormals()
{
    // build normals
    FaceElementTransformations *tr;
    Vector nor(dim);
    for(int b = 0; b < fes->GetNBE(); b++)
    {
        tr = pmesh->GetBdrFaceTransformations(b);
        MFEM_VERIFY(tr->Elem2No < 0, "no boundary element");

        const IntegrationRule *ir = &IntRules.Get(tr->GetGeometryType(), 1);

        const IntegrationPoint &dummy = ir->IntPoint(0);
        tr->SetAllIntPoints(&dummy);
        nor = 0.0;
        if (dim == 1)
        {   
            IntegrationPoint aux_ip;
            tr->Loc1.Transform(dummy, aux_ip);
            nor(0) = 2.0 * aux_ip.x - 1.0;
        }
        else
        {
            CalcOrtho(tr->Jacobian(), nor);
            nor /= nor.Norml2();
        }

        for(int d = 0; d < dim; d++)
        {
            normals(b, d) = nor(d);
        }
    }
}


void DofInfo::BuildElementToBdrElementTable() const
{
    int nE = fes->GetNE();
    int nBe = fes->GetNBE();
    Table *ElementToBdrElementTable1 = new Table();

    ElementToBdrElementTable1->MakeI(nE);

    for(int be = 0; be < nBe; be++)
    {
        auto trans = pmesh->GetBdrFaceTransformations(be);
        ElementToBdrElementTable1->AddAColumnInRow(trans->Elem1No);
    } 

    ElementToBdrElementTable1->MakeJ();
    for(int be = 0; be < nBe; be++)
    {
        auto trans = pmesh->GetBdrFaceTransformations(be);
        ElementToBdrElementTable1->AddConnection(trans->Elem1No, be);
    } 

    ElementToBdrElementTable1->ShiftUpI();
    ElementToBdrElementTable1->Finalize();
    ElementToBdrElementTable = ElementToBdrElementTable1;
}

void DofInfo::BuildDofToDofTable() const
{
    Table *DofToDofTable1 = new Table();
    int nDofs = fes->GetNDofs();

    DofToDofTable1->MakeI(nDofs);

    const auto II = massmatrix.ReadI();
    const auto JJ = massmatrix.ReadJ();

    for(int i = 0; i < nDofs; i ++)
    {
        const int begin = II[i];
        const int end = II[i+1];

        for(int j = begin; j < end; j++)
        {
            DofToDofTable1->AddAColumnInRow(i);
        }
    }

    DofToDofTable1->MakeJ();
    for(int i = 0; i < nDofs; i ++)
    {
        const int begin = II[i];
        const int end = II[i+1];
        for(int j = begin; j < end; j++)
        {
            DofToDofTable1->AddConnection(i, JJ[j]);
        }
    }

    DofToDofTable1->ShiftUpI();
    DofToDofTable1->Finalize();
    DofToDofTable = DofToDofTable1;
}


