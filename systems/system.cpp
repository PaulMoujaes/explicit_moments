#include "system.hpp"

System::System(ParFiniteElementSpace *vfes_, BlockVector &ublock, int numVar_, SystemConfiguration &config_, VectorFunctionCoefficient bdrCond_):
      vfes(vfes_), u0(vfes_, ublock), numVar(numVar_), bdrCond(bdrCond_), dim(vfes_->GetMesh()->Dimension()), ui(numVar), flux1_c(numVar), flux2_c(numVar)
{

   flux1.SetSize(numVar, dim);
   flux2.SetSize(numVar, dim);
}

System::~System()
{ 
   if(Sigma_0)
   {
      delete Sigma_0;
   }

   if(Sigma_1)
   {
      delete Sigma_1;
   }
   
   if(q) 
   {
      delete q;
   }
}

void System::WriteErrors(const Array<double> &errors) const
{
   ofstream File("errors.txt", ios_base::app);

   if (!File)
   {
      MFEM_ABORT("Error opening file.");
   }
   else
   {
      ostringstream Strs;
      for (int i = 0; i < errors.Size(); i++)
      {
         Strs << errors[i] << " ";
      }
      Strs << "\n";
      string Str = Strs.str();
      File << Str;
      File.close();
   }
}

double System::ComputeLambdaij(const Vector &n, const Vector &u1, const Vector &u2) const
{
   Vector n1 = n;
   n1 /= n.Norml2();

   return max(abs(GetWaveSpeed(u1, n1)), abs(GetWaveSpeed(u2, n1)));
}

double System::CalcBarState_returndij(const Vector &ui, const Vector &uj, const Vector &cij, const Vector &cji, Vector &uij, Vector &uji) const
{
   double dij = ComputeDiffusion(cij, cji, ui, uj);
   EvaluateFlux(ui, flux1);
   EvaluateFlux(uj, flux2);
   flux1.Mult(cij, flux1_c);
   flux2.Mult(cij, flux2_c);

   for(int n = 0; n < numVar; n++)
   {   
      uij(n) = 0.5 * ((ui(n) + uj(n)) - (flux2_c(n) - flux1_c(n)) / dij ) ;
   }
   flux1.Mult(cji, flux1_c);
   flux2.Mult(cji, flux2_c);

   for(int n = 0; n < numVar; n++)
   {   
      uji(n) = 0.5 * ((ui(n) + uj(n)) - (flux1_c(n) - flux2_c(n)) / dij ) ;
   }

   return dij;
}


double System::ComputeDiffusion(const Vector &cij, const Vector &cji, const Vector &u1, const Vector &u2) const
{
   if(cij.Norml2() < 1e-15 || cji.Norml2() < 1e-15)
   {
      return 0.0;
   }

   Vector n1 = cij;
   double cij_abs = cij.Norml2();
   Vector n2 = cji;
   double cji_abs = cji.Norml2();

   n1 /= cij_abs;
   n2 /= cji_abs;

   double wavespeed1 = max(abs(GetWaveSpeed(u1, n1)), abs(GetWaveSpeed(u2, n1)));
   double wavespeed2 = max(abs(GetWaveSpeed(u1, n2)), abs(GetWaveSpeed(u2, n2)));

   return max(wavespeed1 * cij_abs, wavespeed2 * cji_abs);
}


void System::ComputeLaxFriedrichsFlux(const Vector &x1, const Vector &x2, const Vector &normal, Vector &y) const
{
   EvaluateFlux(x1, flux1);
   EvaluateFlux(x2, flux2);
   flux1 += flux2;
   double wavespeed = max(abs(GetWaveSpeed(x1, normal)), abs(GetWaveSpeed(x2, normal)));
   subtract(wavespeed, x1, x2, y);
   flux1.AddMult(normal, y);
   y *= 0.5;
}


bool System::GloballyAdmissible(const Vector &x) const
{   
    int nDofs = vfes->GetNDofs();

    for(int i = 0; i < nDofs; i++)
    {   
        for(int n = 0; n < numVar; n++)
        {
            ui(n) = x(i + n * nDofs);
        }
        if(!Admissible(ui))
        {   
            cout << i << endl;
            //ui.Print();
            cout << "NICHT IDP " << endl;
            return false;
        }
        
    }
    return true;
}

void System::CutOff(Vector &x) const
{
   int nDofs = vfes->GetNDofs();

   for(int i = 0; i < nDofs; i++)
   {   
      for(int n = 0; n < numVar; n++)
      {
         ui(n) = x(i + n * nDofs);
      }
        
      if(!Admissible(ui))
      {
         Adjust(ui);
         for(int n = 0; n < numVar; n++)
         {
            x(i + n * nDofs) = ui(n);
         }
      }  
   }
}
