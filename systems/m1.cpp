#include "m1.hpp"

SystemConfiguration configM1;

void AnalyticalSolutionM1(const Vector &x, Vector &u);
void InitialConditionM1(const Vector &x, Vector &u);
void InflowFunctionM1(const Vector &x, Vector &u);

double sigma_a(const Vector &x);
double sigma_aps(const Vector &x);

bool Admissible_outside(Vector &u);
bool solutionKnown_glob;

M1::M1(ParFiniteElementSpace *vfes_, BlockVector &ublock, SystemConfiguration &config_): 
                System(vfes_, ublock, vfes_->GetMesh()->Dimension() + 1, config_, VectorFunctionCoefficient(vfes_->GetMesh()->Dimension() + 1, InflowFunctionM1)), psi2(dim, dim)
{
    configM1 = config_;
    VectorFunctionCoefficient ic(numVar, InitialConditionM1);

    Sigma_0 = new FunctionCoefficient(sigma_a);
    Sigma_1 = new FunctionCoefficient(sigma_aps);

    switch (configM1.benchmark)
    {   
        case 0:
        {
            problemName = "M1-Particle-Pulse";
            solutionKnown = false;
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
        {
            MFEM_ABORT("tbd")
            break;
        }
        default:
            MFEM_ABORT("Analytical solution not known.");
    }
}



void InitialConditionM1(const Vector &x, Vector &u)
{
    switch (configM1.benchmark)
    {
        case 0: 
        {
            u = 0.0;
            double omega_sq = 0.02;
            omega_sq *= omega_sq;
            Vector X = x;
            X -= 0.5;
            double ex = exp(-10.0 * (X(0) * X(0) + X(1) * X(1)) / omega_sq );
            u(0) = max(1e-4, ex);
            break;
        }

        default: 
        MFEM_ABORT("No initial condition for this benchmark implemented!");        
    }
}

void InflowFunctionM1(const Vector &x, Vector &u)
{
    double f = 0.9;
    switch (configM1.benchmark)
    {
        case 0:
        { 
            InitialConditionM1(x,u);  
            break;      
        }

        default:
        MFEM_ABORT("No Inflow for this benchmark implemented!");
    }
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



double sigma_a(const Vector &x)
{ 
    switch (configM1.benchmark)
    {
        case 0: 
        {
            return 0.0;
        }

        default: 
        MFEM_ABORT("No sigma_a for this benchmark!");
    }
    return 0.0;
}

double sigma_aps(const Vector &x)
{
    double sigma_s = 0.0;
    switch (configM1.benchmark)
    {
        case 0: 
        {
            sigma_s = 0.0; break;
        }

        default: 
        MFEM_ABORT("No sigma_aps for this benchmark!");
    }
    return sigma_s + sigma_a(x);
}




