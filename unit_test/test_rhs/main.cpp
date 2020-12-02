#include <AMReX_PlotFileUtil.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_BCRec.H>


using namespace amrex;

#include <test_react.H>
#include <test_react_F.H>
#include <extern_parameters.H>
#include <network.H>
#include <rhs_zones.H>
#include <AMReX_buildInfo.H>
#include <unit_test.H>

int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
        main_main();
    }
    amrex::Finalize();
    return 0;
}

void main_main ()
{
    BoxArray ba;
    Geometry geom;

    network_init();

    int Ncomp = 100;
    int Nghost = 0;

    DistributionMapping dm(ba);
    MultiFab state(ba, dm, Ncomp, Nghost);

    for (MFIter mfi(state); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.tilebox();

        auto s = state.array(mfi);

        AMREX_PARALLEL_FOR_3D(bx, i, j, k,
        {
            do_rhs(i, j, k, s);
        });

    }

}
