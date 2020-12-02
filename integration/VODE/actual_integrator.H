#ifndef actual_integrator_H
#define actual_integrator_H

// Common variables and routines for burners
// that use VODE for their integration.

#include <network.H>
#include <burn_type.H>
#include <temperature_integration.H>
#include <extern_parameters.H>
#include <vode_type.H>
#include <vode_dvode.H>

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void actual_integrator (burn_t& state, Real dt)
{

    dvode_t vode_state;

    // Set the tolerances.  We will be more relaxed on the temperature
    // since it is only used in evaluating the rates.
    //
    // **NOTE** if you reduce these tolerances, you probably will need
    // to (a) decrease dT_crit, (b) increase the maximum number of
    // steps allowed.

    for (int n = 1; n <= NumSpec; ++n) {
        vode_state.atol(n) = atol_spec; // mass fractions
    }
    vode_state.atol(net_itemp) = atol_temp; // temperature
    vode_state.atol(net_ienuc) = atol_enuc; // energy generated

    for (int n = 1; n <= NumSpec; ++n) {
        vode_state.rtol(n) = rtol_spec; // mass fractions
    }
    vode_state.rtol(net_itemp) = rtol_temp; // temperature
    vode_state.rtol(net_ienuc) = rtol_enuc; // energy generated

    // Start off by assuming a successful burn.

    state.success = true;

    // Initialize the integration time.

    vode_state.t = 0.0_rt;
    vode_state.tout = dt;

    // Fill in the initial integration state.

    burn_to_vode(state, vode_state);

    // Save the initial energy for our later diagnostics.

    Real e_in = state.e;

    state.T_old = state.T;
    state.cv_old = state.cv;
    state.cp_old = state.cp;

    state.self_heat = true;

    // Call the integration routine.

    int istate = dvode(state, vode_state);

    // Subtract the energy offset.

    vode_state.y(net_ienuc) -= e_in;

    // Copy the integration data back to the burn state.

    vode_to_burn(vode_state, state);

    // Normalize the final abundances.

    normalize_abundances_burn(state);

    // Get the number of RHS and Jacobian evaluations.

    state.n_rhs = vode_state.NFE;
    state.n_jac = vode_state.NJE;

    // VODE does not always fail even though it can lead to unphysical states.
    // Add some checks that indicate a burn fail even if VODE thinks the
    // integration was successful.

    if (istate < 0) {
        state.success = false;
    }

    if (vode_state.y(net_itemp) < 0.0_rt) {
        state.success = false;
    }

    for (int n = 1; n <= NumSpec; ++n) {
        if (vode_state.y(n) < -vode_failure_tolerance) {
            state.success = false;
        }

        if (vode_state.y(n) > 1.0_rt + vode_failure_tolerance) {
            state.success = false;
        }
    }


}

#endif
