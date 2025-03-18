#include "m1.hpp"

SystemConfiguration configM1;

void AnalyticalSolutionM1(const Vector &x, Vector &u);
void InitialConditionM1(const Vector &x, Vector &u);
void InflowFunctionM1(const Vector &x, Vector &u);
bool Admissible_outside(Vector &u);
bool solutionKnown_glob;

M1::M1(ParFiniteElementSpace *vfes_, BlockVector &ublock, SystemConfiguration &config_): 
                System(vfes_, ublock, vfes_->GetMesh()->Dimension() + 1, config_, VectorFunctionCoefficient(vfes_->GetMesh()->Dimension() + 1, InflowFunctionM1)), psi2(dim, dim)
{
    configM1 = config_;
    VectorFunctionCoefficient ic(numVar, InitialConditionM1);

    /*
    ui(0) = 0.5;
    ui(1) = 0.1;
    ui(2) = 0.1;

    cout << Evaluate_f(ui) << endl;
    cout << EvaluateEddingtonFactor(ui) << endl;
    Vector v(dim);
    Evaluate_f_vec(ui, v);
    v.Print();
    cout << v.Norml2()<<  "\n\n";

    EvaluatePsi2(ui, psi2);
    psi2.Print();
    cout << endl;

    MFEM_VERIFY(Admissible(ui), "u not admissible!")


    EvaluateFluxJacobian(ui, fluxJac1);
    cout << fluxJac1(0).Det() << ", "<< fluxJac1(1).Det()<< endl;
    EvaluateFlux(ui, flux1);

    Vector uj = ui;
    fluxJac1(0).Mult(ui, uj);
    flux2.SetCol(0, uj);
    fluxJac1(1).Mult(ui, uj);
    flux2.SetCol(1,uj);

    flux1.Print();
    cout << endl;
    flux2.Print();

    MFEM_ABORT("")
    //*/


    switch (configM1.benchmark)
    {   
        case 0:
        {
            problemName = "M1-Smooth-Beam-from-left";
            solutionKnown = false;
            collision_coeff = 0.1;
            u0.ProjectCoefficient(ic);
            break;
        }

        case 1:
        {
            problemName = "M1-Smooth-Beam-from-left-and-bottom";
            solutionKnown = false;
            collision_coeff = 0.1;
            u0.ProjectCoefficient(ic);
            break;
        }

        case 2:
        {
            problemName = "M1-Disc-Beam-from-left-and-bottom";
            solutionKnown = false;
            collision_coeff = 0.1;
            u0.ProjectCoefficient(ic);
            break;
        }

        default:
            MFEM_ABORT("No such test case implemented.");
    }

    solutionKnown_glob = solutionKnown;
    //collision_coeff = 0.0;
}


double M1::ComputePressureLikeVariable(Vector &u) const
{
    // such that idp is equivalent to (1-f) > 0
    return 1.0 - Evaluate_f(u);
}

void M1::EvaluateFluxJacobian(const Vector &u, DenseTensor &fluxJac) const
{    
    fluxJac.SetSize(numVar, numVar, dim);
    double chi = EvaluateEddingtonFactor(u);
    Vector v(dim);
    Evaluate_f_vec(u, v);
    if(v.Norml2() > 1.0)
    {
        v /= v.Norml2();
    }
    double f_sq = v.Norml2();
    f_sq *= f_sq;

    switch(dim)
    {
        case 2:
        {
            // x direction
            fluxJac(0,0,0) = 0.0;
            fluxJac(0,1,0) = 1.0;
            fluxJac(0,2,0) = 0.0;

            fluxJac(1, 0, 0) = 0.5 * (1.0 - chi);
            fluxJac(1, 1, 0) = 0.5 * (3.0 * chi - 1.0) * v(0) / (f_sq + 1e-15);
            fluxJac(1, 2, 0) = 0.0;
            
            fluxJac(2,0,0) = 0.0;
            fluxJac(2,1,0) = 0.0;
            fluxJac(2,2,0) = 0.5 * (3.0 * chi - 1.0) * v(0) / (f_sq + 1e-15);

            // y direction
            fluxJac(0,0,1) = 0.0;
            fluxJac(0,1,1) = 0.0;
            fluxJac(0,2,1) = 1.0;

            fluxJac(1, 0, 1) = 0.0;
            fluxJac(1, 1, 1) = 0.5 * (3.0 * chi - 1.0) * v(1) / (f_sq + 1e-15);
            fluxJac(1, 2 ,1) = 0.0;
            
            fluxJac(2,0,1) = 0.5 * (1.0 - chi);
            fluxJac(2,1,1) = 0.0;
            fluxJac(2,2,1) = 0.5 * (3.0 * chi - 1.0) * v(1) / (f_sq + 1e-15);
            break;
        }
        default: MFEM_ABORT("Flux Jacobian not implemented for this dimension");
    }
}

void M1::EvaluatePsi2(const Vector &u, DenseMatrix &psi2) const
{
    psi2.SetSize(dim, dim);
    psi2 = 0.0;

    double chi = EvaluateEddingtonFactor(u);
    Vector v(dim);
    Evaluate_f_vec(u, v);
    for(int d = 0; d < dim; d++)
    {
        psi2(d,d) = 0.5 * (1.0 - chi);
    }

    if(v.Norml2() > 1.0)
    {
        v /= v.Norml2();
    }
    double f_sq = v.Norml2();
    f_sq *= f_sq; 
    for(int i = 0; i < dim; i++)
    {
        for(int j = 0; j < dim; j++)
        {
            psi2(i,j) += 0.5 * (3.0 * chi - 1.0) * v(i) * v(j) / (f_sq + 1e-15);
        }
    }

    psi2 *= max(u(0), 1e-15);
}

void M1::Evaluate_f_vec(const Vector &u, Vector &v) const
{
    v.SetSize(dim);
    for(int d = 0; d < dim; d++)
    {
        v(d) = u(d+1) / max(u(0), 1e-15);
    }
}

double M1::Evaluate_f(const Vector &u) const
{
    double f = 0.0;
    for(int d = 0; d < dim; d++)
    {
        f += u(d+1) * u(d+1);
    }

    return sqrt(f) / max(u(0), 1e-15);
}

double M1::EvaluateEddingtonFactor(const Vector &u) const
{
    double f = min(1.0 ,Evaluate_f(u));

    //return (3.0 + 4.0 * f * f) / (5.0 + 2.0 * sqrt(4.0 - 3.0 * f * f));
    return (5.0 - 2.0 * sqrt(max(4.0 - 3.0 * f * f, 0.0))) / 3.0;
}

double M1::EvaluateDerivativeEddingtonFactor(const Vector &u) const
{
    double f = min(1.0 ,Evaluate_f(u));

    return 2.0 * f /  sqrt((max(4.0 - 3.0 * f * f, 1e-15)));
}

void M1::EvaluateFlux(const Vector &u, DenseMatrix &fluxEval) const
{
    fluxEval.SetSize(u.Size(), dim);

    EvaluatePsi2(u, psi2);

    for(int d = 0; d < dim; d++)
    {
        fluxEval(0, d) = u(d+1);

        for(int dd = 0; dd < dim; dd++)
        {
            fluxEval(d +1, dd) = psi2(d, dd);
        }
    }
}

void M1::EntropyFlux(const Vector &u, Vector &q) const
{
    q.SetSize(dim);

    MFEM_ABORT("tbd")
}


void M1::EntropyVariable(const Vector &u, Vector &v) const
{
    MFEM_ABORT("tbd")
}


void M1::EntropyPotential(const Vector &u, Vector &psi) const
{
    MFEM_ABORT("tbd")
}


double M1::GetWaveSpeed(const Vector &u, const Vector n) const
{   

    return 1.0;
}


bool M1::Admissible(const Vector &u) const
{
    MFEM_VERIFY(!isnan(u(0)), "psi0 is nan!");
    if(u(0) < 1e-15)
    {
        cout << "psi0 = " << u(0) << ", ";
        return false; 
    }


    Vector v(dim);
    for(int d = 0; d < dim; d++)
    {
        v(d) = u(d+1) / u(0);
        MFEM_VERIFY(!isnan(u(d+1)), "psi1 is nan!");
    }

    if(! (v.Norml2() < 1.0 + 1e-15))
    {
        cout << "f = " << v.Norml2() << ", ";
        return false; 
    }

    return true;
}

void M1::Adjust(Vector &u) const
{
    u(0) = max(u(0), 1e-15);

    Vector v(dim);
    for(int d = 0; d < dim; d++)
    {
        v(d) = u(d+1) / u(0);
    }

    if(v.Norml2() > 1.0)
    {
        v /= v.Norml2();
    }

    for(int d = 0; d < dim; d++)
    {
        u(d+1) = v(d) * u(0);
    }
}

void M1::SetBoundaryConditions(const Vector &y1, Vector &y2, const Vector &normal, int attr) const
{
   switch (attr)
   {
        case 1: // Solid wall boundary.
        {
            if (dim == 1)
            {
                y2(0) = y1(0);
                y2(1) = -y1(1);
            }
            else if (dim == 2)
            {
                double momTimesNorm = y1(1) * normal(0) + y1(2) * normal(1);
                y2(0) = y1(0);
                y2(1) = y1(1) - 2.0 * momTimesNorm * normal(0);
                y2(2) = y1(2) - 2.0 * momTimesNorm * normal(1);
            }
            else
            {
                double momTimesNorm = y1(1) * normal(0) + y1(2) * normal(1) + y1(3) * normal(2);
                y2(0) = y1(0);
                y2(1) = y1(1) - 2.0 * momTimesNorm * normal(0);
                y2(2) = y1(2) - 2.0 * momTimesNorm * normal(1);
                y2(3) = y1(3) - 2.0 * momTimesNorm * normal(2);
            }
            break;
        }
        case 2: // Supersonic outlet.
        {
            y2 = y1;
            break;
        }
        case 3: // Supersonic inlet.
        {
            break;
        }
        default:
            MFEM_ABORT("Invalid boundary attribute.");
    }
}


double M1::Entropy(const Vector &u) const
{
    MFEM_ABORT("tbd")
} 


void AnalyticalSolutionM1(const Vector &x, Vector &u)
{
   const int dim = x.Size();
    switch (configM1.benchmark)
    {
        case 0:
        case 1:
        case 2:
        {
            MFEM_ABORT("tbd")
            break;
        }
        default:
            MFEM_ABORT("Analytical solution not known.");
    }

    MFEM_VERIFY(Admissible_outside(u), "Analytical solution not IDP");
}



void InitialConditionM1(const Vector &x, Vector &u)
{
    if(solutionKnown_glob) 
    {
        AnalyticalSolutionM1(x, u);
    }
    else
    {
        switch (configM1.benchmark)
        {
            case 0: 
            case 1:
            case 2:
            {
                u = 0.0;
                //u = 0.01;
                u(0) = 0.1;
                
                //InflowFunctionM1(x,u);
                break;
            }

            default: 
            MFEM_ABORT("No initial condition for this benchmark implemented!");
        }
    }

    MFEM_VERIFY(Admissible_outside(u), "Initial condition not IDP");
}

void InflowFunctionM1(const Vector &x, Vector &u)
{
    double f = 0.9;
    switch (configM1.benchmark)
    {
        case 0:
        { 
            u = 0.0;
            u(0) = 0.1;
            
            double ymax = 0.55;
            double ymin = 0.45;
            double middle = 0.5 * (ymax + ymin);
            double alpha = 0.0; //- 0.2 * M_PI;

            if((x(1) <= ymax && x(1) >= ymin))
            {
                u(0) = 0.9 * cos(10.0 * M_PI * (x(1) - middle ) / 3.0) * cos(10.0 * M_PI * (x(1) - middle ) / 3.0) + 0.1;
                u(1) = cos(alpha) * f * u(0);
                u(2) = sin(alpha) * f * u(0);
            }

            break; 
        }
        case 1:
        { 
            u = 0.0;
            u(0) = 0.1;
            
            double xmax = 0.4;
            double xmin = 0.3;
            double middle = 0.5 * (xmax + xmin);
            if((x(1) <= xmax && x(1) >= xmin))
            {
                u(0) = 0.5 * cos(10.0 * M_PI * (x(1) - middle ) / 3.0) * cos(10.0 * M_PI * (x(1) - middle ) / 3.0) + 0.1 ;
                u(1) =  f * u(0);
                //u(2) =  0.0;
            }
            else if((x(0) <= xmax && x(0) >= xmin))
            {
                double u_0 = 0.5 * cos(10.0 * M_PI * (x(0) - middle ) / 3.0) * cos(10.0 * M_PI * (x(0) - middle ) / 3.0) + 0.1;
                u(0) = u_0; // max(u(0), u_0 );

                //u(1) =  max(0.0,  u(1));
                u(2) =  f * u(0);
            }
            break; 
        }
        case 2:
        { 
            u = 0.0;
            u(0) = 0.1;
            
            double xmax = 0.1;
            double xmin = 0.05;
            double middle = 0.5 * (xmax + xmin);
            if((x(1) <= xmax && x(1) >= xmin))
            {
                u(0) = 0.6;
                u(1) =  f * u(0);
                //u(2) =  0.0;
            }
            else if((x(0) <= xmax && x(0) >= xmin))
            {
                u(0) = 0.6; 
                u(2) =  f * u(0);
            }
            break; 
        }

        default:
        MFEM_ABORT("No Inflow for this benchmark implemented!");
    }

    MFEM_VERIFY(Admissible_outside(u), "Inflow not IDP");
}

bool Admissible_outside(Vector &u)
{
    int dim = u.Size()-1;
    if(u(0) < 1e-15)
    {
        return false;
    }

    Vector v(dim);
    for(int d = 0; d < dim; d++)
    {
        v(d) = u(d+1) / u(0);
    }

    return v.Norml2() < 1.0+1e-15;

}


void M1::ComputeErrors(Array<double> & errors, const ParGridFunction &u, double domainSize) const
{
   errors.SetSize(3);
   Vector component(dim + 2);
   VectorFunctionCoefficient uAnalytic(numVar, AnalyticalSolutionM1);

   component = 0.0;
   component(0) = 1.0;
   VectorConstantCoefficient weight1(component);
   errors[0] = u.ComputeLpError(1.0, uAnalytic, NULL, &weight1) / domainSize;
   errors[1] = pow(u.ComputeLpError(2.0, uAnalytic, NULL, &weight1) / domainSize, 2.0);
   errors[2] = u.ComputeLpError(numeric_limits<double>::infinity(), uAnalytic, NULL, &weight1);
   
   /*
   component = 0.0;
   component(1) = 1.0;
   VectorConstantCoefficient weight2(component);
   errors[0] += u.ComputeLpError(1.0, uAnalytic, NULL, &weight2) / domainSize;
   errors[1] += pow(u.ComputeLpError(2.0, uAnalytic, NULL, &weight2) / domainSize, 2.0);
   errors[2] = max(errors[2], u.ComputeLpError(numeric_limits<double>::infinity(), uAnalytic, NULL, &weight2));

   component = 0.0;
   component(2) = 1.0;
   VectorConstantCoefficient weight3(component);
   errors[0] += u.ComputeLpError(1.0, uAnalytic, NULL, &weight3) / domainSize;
   errors[1] += pow(u.ComputeLpError(2.0, uAnalytic, NULL, &weight3) / domainSize, 2.0);
   errors[2] = max(errors[2], u.ComputeLpError(numeric_limits<double>::infinity(), uAnalytic, NULL, &weight3));

   if (dim > 1)
   {
      component = 0.0;
      component(3) = 1.0;
      VectorConstantCoefficient weight4(component);
      errors[0] += u.ComputeLpError(1.0, uAnalytic, NULL, &weight4) / domainSize;
      errors[1] += pow(u.ComputeLpError(2.0, uAnalytic, NULL, &weight4) / domainSize, 2.0);
      errors[2] = max(errors[2], u.ComputeLpError(numeric_limits<double>::infinity(), uAnalytic, NULL, &weight4));
   }

   if (dim > 2)
   {
      component = 0.0;
      component(4) = 1.0;
      VectorConstantCoefficient weight5(component);
      errors[0] += u.ComputeLpError(1.0, uAnalytic, NULL, &weight5) / domainSize;
      errors[1] += pow(u.ComputeLpError(2.0, uAnalytic, NULL, &weight5) / domainSize, 2.0);
      errors[2] = max(errors[2], u.ComputeLpError(numeric_limits<double>::infinity(), uAnalytic, NULL, &weight5));
   }

   //*/
   errors[1] = sqrt(errors[1]);
}




