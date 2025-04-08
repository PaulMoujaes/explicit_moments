// This code solves convection dominated systems implicitly via linearized fixed-point-iterations that converge to implicit solutions to the nonlinear implicit problem

#include "auxiliaries/auxiliarieslib.hpp"
#include "methods/methodslib.hpp"
#include "systems/systemslib.hpp"
//#include "nonlinearsolvers/fpiterlib.hpp"
#include <chrono>

using namespace std::chrono;
using namespace mfem;

int main(int argc, char *argv[])
{
    MPI_Session mpi(argc, argv);
    int num_procs = Mpi::WorldSize();
    int myid = Mpi::WorldRank();
    Hypre::Init();

    // 1. Choose simulation setup parameters
    SystemConfiguration sysconfig;
    int system = 0;
    sysconfig.benchmark = 0;

    FEScheme FEscheme = loworder;
    int order = 1;
    int refinements = 0;
    bool paraview;
    bool important = false;
    string vtk_name = "Test";

    const char *MeshFile = "meshes/periodic-square.mesh";
    const char *OutputDirectory = "output/not_important";

    double dt = 0.001;
    double CFL = -std::numeric_limits<double>::infinity();;
    double t_final = 1.0;
    int ode_solver_type = 1; // explicit Euler

    int visualizationSteps = 1;
    int paraviewSteps = 0;

    int precision = 8;
    cout.precision(precision);

    OptionsParser args(argc, argv);

    args.AddOption(&sysconfig.benchmark, "-b", "--benchmark", "Benchmark problem to solve.");
    args.AddOption(&ode_solver_type, "-ode","--ode-solver", "Ode Solver!");
    args.AddOption(&system, "-sys", "--system", "System to solve.");
    args.AddOption(&order, "-o", "--order", "Order (polynomial degree) of the finite element space.");
    args.AddOption(&refinements, "-r", "--refine", "Number of times to refine the mesh uniformly.");
    args.AddOption((int*)(&FEscheme), "-fe", "--fescheme", 
                    "Scheme: 0 - LowOrder, \n\t"
                    "        1 - MCL).");
    args.AddOption(&MeshFile, "-m", "--mesh", "Mesh file to use.");
    args.AddOption(&OutputDirectory, "-out", "--output-directory", "Output directory.");
    args.AddOption(&dt, "-dt", "--time-step", "Time step size.");
    args.AddOption(&t_final, "-ft", "--final-time", "Final Time.");
    args.AddOption(&CFL, "-cfl", "--cfl", "CFL number.");
    args.AddOption(&vtk_name, "-vn", "--vtk-name", "Give the vtk files a name.");
    args.AddOption(&visualizationSteps, "-vis", "--visualization-steps", "Visualize every n-th time step");
    args.AddOption(&paraviewSteps, "-ps", "--paraview-visualization-steps", "Export VTK every n-th time step. If no ps > 0 is given, no VTK output will be generated");
    args.AddOption(&important, "-i", "--important", "-ni", "--not-important", "Write out vtk files into another folder for easier find.");
    args.Parse();

    if (!args.Good())
    {
        if(Mpi::Root())
        {
            args.PrintUsage(cout);
        }
        return 1;
    }
    if(Mpi::Root())
    {
        args.PrintOptions(cout);
    }

    bool addaptive_ts = CFL > 0.0;
    paraview = !vtk_name.std::string::empty() && paraviewSteps != 0;

    auto lastSlash = std::string(MeshFile).find_last_of("/");
    string meshfile;

    if (lastSlash != std::string::npos) 
    {
        // Use substr to get the string after '/'
        meshfile = std::string(MeshFile + lastSlash + 1);

    } 
    else
    {
        // No '/' found, use the original string
        meshfile = MeshFile;
    }

    if(important)
    {
        OutputDirectory = "output/Important";
    }


    // 2. Read mesh
    Mesh *mesh = new Mesh(MeshFile, 1, 1);
    const int dim = mesh->Dimension();
    for (int level = 0; level < refinements; level++)
    {
        mesh->UniformRefinement();
    }
    if (mesh->NURBSext)
    {
        mesh->SetCurvature(max(order, 1));
    }

    ParMesh pmesh(MPI_COMM_WORLD, *mesh);
    delete mesh;

    // 3. Set number of solution components 

    int numVar;
    switch (system)
    {
        case m1: numVar = 1 + dim; break; //M1
        case m2: //numVar = 1 + dim + dim * dim; break; //M2
        default:
        {
            if(Mpi::Root())
            {
                cout << "Unknown System: " << system  << endl;
            }
            return 1; 
        }
    }

    // 4. Choose CG FE space for specific polynomial order and refined mesh.
    const int btype = BasisType::ClosedUniform; 
    H1_FECollection fec(order, dim, btype);
    ParFiniteElementSpace fes(&pmesh, &fec);
    ParFiniteElementSpace vfes(&pmesh, &fec, numVar, Ordering::byNODES);
    ParFiniteElementSpace dfes(&pmesh, &fec, dim, Ordering::byNODES);

    Array<int> offsets(numVar + 1);
    for (int k = 0; k <= numVar; k++) 
    { 
        offsets[k] = k * fes.GetNDofs(); 
    }
    BlockVector ublock(offsets);
    ublock.UseDevice(true);

    const int problemSize = vfes.GlobalTrueVSize ();
    const int component_problemSize = fes.GlobalTrueVSize();
    int loc_elN = fes.GetNE();
    int glob_elN = 0;
    MPI_Allreduce(&loc_elN, &glob_elN, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    double h = sqrt((double) glob_elN);
    
    if(Mpi::Root())
    {
        cout << endl;
        cout << "Number of elements:               " << glob_elN << "\n";
        cout << "Number of unknowns:               " << problemSize << "\n";
        cout << "Number of unknowns per component: " << component_problemSize << "\n";
        cout << "1 / h =                           " << h << "\n";
        cout << "h =                               " << 1.0 / h << "\n\n";

    }
        

    // 5. Choose a system
    if(Mpi::Root())
    {
        cout << "Building System..." << "\n";
    }
    System *sys;
    switch (system)
    {
        case m1: sys = new M1(&vfes, ublock, sysconfig); break;
        case m2: //sys = new M2(&vfes, ublock, sysconfig); break;
        default: 
            if(Mpi::Root())
            {
                cout << "Unknown System: " << system  << endl;
            }    
            return 2; 
    }
    string prob = sys->problemName;
    if(Mpi::Root())
    {
        cout << "System built! " << "\n\n" << "Simulating " << prob << "\n\n";
    }

    ParGridFunction u(&vfes, ublock);
    u = sys->u0;
    ParGridFunction Du = u;
    Du = 0.0;
    ParGridFunction res_gf = Du;
    ParGridFunction main(&fes, ublock.GetBlock(0));
    ParGridFunction f(&fes);
    ParGridFunction psi1(&dfes, ublock.GetData() + fes.GetNDofs());
    sys->ComputeDerivedQuantities(u, f);

    ParBilinearForm *mL = new ParBilinearForm(&fes);
    Vector lumpedMassMatrix(mL->Height());
    mL->AddDomainIntegrator(new LumpedIntegrator(new MassIntegrator));
    mL->Assemble();
    mL->Finalize();
    mL->SpMat().GetDiag(lumpedMassMatrix);
    delete mL;

    double loc_mass = main * lumpedMassMatrix;
    double glob_init_mass = 0.0;
    MPI_Allreduce(&loc_mass, &glob_init_mass, 1, MPI_DOUBLE, MPI_SUM,
              MPI_COMM_WORLD);

    if(Mpi::Root())
    {
        cout << "Building DofInfo..."<< endl;
    }
    DofInfo dofs(&fes);
    if(Mpi::Root())
    {
        cout << "DofInfo built!" << "\n\n";
    }

    if(Mpi::Root())
    {
        cout << "Building FE Method..." << "\n";
    }
    FE_Evolution *met = NULL;
    switch(FEscheme)
    {
        case loworder: met = new LowOrder(&fes, &vfes, sys, dofs, lumpedMassMatrix); break;
        case mcl: met = new MCL(&fes, &vfes, sys, dofs, lumpedMassMatrix); break;
        case mcl_comp: met = new MCL_Comp(&fes, &vfes, sys, dofs, lumpedMassMatrix); break;
        default: 
        {
            if(Mpi::Root())
            {
                cout <<"Unknown FE Method!" << endl;
            }
            return 3;
        }
    }
    if(Mpi::Root())
    {
        cout << "FE Method built!" << "\n\n";
    }
    
    // 9. Choose the scheme via the fixed-point iteration
    ODESolver *odesolver = NULL;;
    switch (ode_solver_type)
    {
        // Explicit methods
        case 1: odesolver = new ForwardEulerSolver; break;
        case 2: odesolver = new RK2Solver(1.0); break;
        case 3: odesolver = new RK3SSPSolver; break;
        default:
        if (Mpi::Root())
        {
           cout << "Unknown ODE solver type: " << ode_solver_type << '\n';
        }
        return 5;
    }

    // 10. Prepare time stepping.
    mfem::ParaViewDataCollection *pd = NULL;
    string para_loc;
    if (paraview)
    {
        // string para_loc = "test";
        para_loc = vtk_name + "_" + prob + "_" + meshfile + "_fe-" + to_string(FEscheme) + "_sys-" + to_string(system) + "_bm-"+ to_string(sysconfig.benchmark) 
            + "_o-" + to_string(order) + "_CFL-" + to_string(CFL) + "_ode-" +to_string(ode_solver_type) + "_r-" + to_string(refinements);

        pd = new mfem::ParaViewDataCollection(para_loc, &pmesh);
        pd->SetPrefixPath(OutputDirectory);
        pd->SetHighOrderOutput(true);
        pd->SetLevelsOfDetail(order);
        pd->RegisterField("psi0", &main);
        pd->RegisterField("psi1", &psi1);
        pd->RegisterField("f", &f);
        //pd->RegisterField("inflow", &met->inflow);

        //pd->RegisterField("analytical solution", &ana_sol);
        pd->SetCycle(0);
        pd->SetTime(0.0);
        pd->Save();
        if(Mpi::Root())
        {
            cout << "Initial condition saved as " << para_loc << "\n\n";
        }  
    } 
    if(Mpi::Root())
    {
        cout << "Preprocessing done. Entering time stepping loop!\n";
    }

    double t = 0.0;
    odesolver->Init(*met);
    met->SetTime(t);
    bool done = false;

    
    tic_toc.Clear();
    tic_toc.Start();
    auto start = high_resolution_clock::now();


    // 11. Perform time integration.
    for (int ti = 0; !done;)
    {
        ti++;
        if(addaptive_ts)
        {
            dt = met->Compute_dt(u, CFL);
        }
        double dt_real = min(dt, t_final - t);
        odesolver->Step(u, t, dt_real);

        done = (t >= t_final - 1e-8*dt);

        double min_loc = main.Min();
        double max_loc = main.Max();
        double min_glob, max_glob;
        MPI_Allreduce(&min_loc, &min_glob, 1, MPI_DOUBLE, MPI_MIN,
                  MPI_COMM_WORLD);
        MPI_Allreduce(&max_loc, &max_glob, 1, MPI_DOUBLE, MPI_MAX,
                  MPI_COMM_WORLD);

        if ((done || ti % visualizationSteps == 0) && Mpi::Root())
        {               
            // visualize on console
            auto stop = high_resolution_clock::now();
            auto msecs = duration_cast<milliseconds>(stop - start);
            double remainder_msecs = msecs.count() * (t_final - t) / (visualizationSteps * dt);
            int remainder_secs = std::round(remainder_msecs / 1.0E+3);
            int mins = remainder_secs / 60;
            int secs = remainder_secs % 60;
            int hrs = mins / 60;
            mins = mins % 60;


            cout << "Time step: " << ti << ", Time: " << t << ", main in [" << min_glob << ", "<< max_glob << "], Remaining: " << hrs << "hrs " << mins << "mins " << secs << "secs." << '\n';

            start = high_resolution_clock::now();
        }

        if (paraview && (ti % paraviewSteps == 0 || done))
        {
            sys->ComputeDerivedQuantities(u, f);
            pd->SetCycle(ti);
            pd->SetTime(t);
            pd->Save();
            if(Mpi::Root())
            {
                cout << "VTK exported: " << para_loc << "\n\n";
            }
        }
    }

    tic_toc.Stop();
    double realtime = tic_toc.RealTime();
    if(Mpi::Root())
    {    
        int hrsmins = (int(realtime) / 3600) * 3600 + ((int(realtime) % 3600) / 60) * 60;
        cout << "Time stepping loop done in " << int(realtime) / 3600 << "hrs " <<  (int(realtime) % 3600) / 60 << "mins "<< realtime - double(hrsmins) << "secs.\n\n";

        // 12. Save the computation time.
        ofstream File("runtimes.txt", ios_base::app);

        if (!File)
        {
            MFEM_ABORT("Error opening file.");
        }
        else
        {
            ostringstream Strs;
            Strs << realtime << " seconds needed for the run: ";
            for (int i = 0; i < argc; i++)
            { 
                Strs << argv[i] << " "; 
            }
            Strs << '\n';
            string Str = Strs.str();
            File << Str;
            File.close();
        }
    }

    // 13. Compute solution errors and additional values.
    double domainSize = lumpedMassMatrix.Sum();

    if (sys->solutionKnown)
    {
        Array<double> errors;
        sys->ComputeErrors(errors, u, domainSize);

        if(Mpi::Root())
        {
            cout << "L1 error:                           " << errors[0] << '\n';
            cout << "L2 error:                           " << errors[1] << '\n';
            cout << "Linf error                          " << errors[2] << "\n\n";
            sys->WriteErrors(errors);
        }
    }

    loc_mass = main * lumpedMassMatrix;
    double glob_end_mass = 0.0;
    MPI_Allreduce(&loc_mass, &glob_end_mass, 1, MPI_DOUBLE, MPI_SUM,
              MPI_COMM_WORLD);
    double min_loc = main.Min();
    double max_loc = main.Max();
    double min_glob, max_glob;
    MPI_Allreduce(&min_loc, &min_glob, 1, MPI_DOUBLE, MPI_MIN,
                  MPI_COMM_WORLD);
    MPI_Allreduce(&max_loc, &max_glob, 1, MPI_DOUBLE, MPI_MAX,
                  MPI_COMM_WORLD);

    if(Mpi::Root())
    {
        cout  << "Minimum of primary values:          " << min_glob << '\n'
            << "Maximum of primary values:          " << max_glob << '\n'
            << "Difference in solution mass:        " << abs(glob_init_mass - glob_end_mass) / domainSize << "\n\n";

        if (paraview)
        {
            cout << "Results are saved as " << para_loc << endl;
        }
    }
    

    delete odesolver;
    delete met;
    delete sys;
    delete pd;
}
