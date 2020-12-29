#ifndef actual_integrator_H
#define actual_integrator_H

#include <network.H>
#include <actual_network.H>
#include <actual_rhs.H>
#include <burn_type.H>
#include <rate_type.H>
#include <temperature_integration.H>
#include <eos_type.H>
#include <eos.H>
#include <extern_parameters.H>

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void clean_state (burn_t& state)
{
    // Renormalize the abundances.

    normalize_abundances_burn(state);

    // Call the EOS to get T from e.

    eos_t eos_state;

    burn_to_eos(state, eos_state);

    eos(eos_input_re, eos_state);

    eos_to_burn(eos_state, state);

    // Ensure that the temperature always stays within reasonable limits.

    state.T = amrex::min(MAX_TEMP, amrex::max(state.T, EOSData::mintemp));

}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void initialize_state (burn_t& state)
{
    // We assume that (rho, T) coming in are valid, do an EOS call
    // to get T from e.

    eos_t eos_state;

    burn_to_eos(state, eos_state);

    eos(eos_input_rt, eos_state);

    eos_to_burn(eos_state, state);

    state.self_heat = true;

    state.success = true;

    state.n_rhs = 0;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void evaluate_rhs (burn_t& state, Array1D<Real, 1, NumSpec>& f_minus, Array1D<Real, 1, NumSpec>& f_plus,
                   Real& dedt)
{
    rate_t rr;
    evaluate_rates(state, rr);

    Array1D<Real, 1, NumSpec> y;

    for (int spec = 1; spec <= NumSpec; ++spec) {
        y(spec) = state.xn[spec-1] * aion_inv[spec-1];
    }

    // For each species calculate its positive and negative RHS contributions.

    for (int spec = 1; spec <= NumSpec; ++spec) {

        f_plus(spec) = 0.0;
        f_minus(spec) = 0.0;

        // Loop through rates. All terms that have a positive contribution
        // to the RHS for this species get the full term added to f_plus.
        // All terms that deplete this species are added to f_minus.

        for (int rate = 1; rate <= Rates::NumRates; ++rate) {
            rhs_t rhs_data = RHS::rhs_data(spec, rate);

            if (std::abs(rhs_data.prefactor) > 0.0) {

                Real term = rhs_data.prefactor;

                if (rhs_data.specindex1 >= 0) {
                    term *= y(rhs_data.specindex1);
                }

                if (rhs_data.specindex2 >= 0) {
                    term *= y(rhs_data.specindex2);
                }

                if (rhs_data.specindex3 >= 0) {
                    term *= y(rhs_data.specindex3);
                }

                term *= rr.rates(rate);

                // Scale by aion to get dX/dt instead of dY/dt.

                term *= aion[spec - 1];

                if (rhs_data.prefactor > 0.0) {
                    f_plus(spec) += term;
                } else {
                    f_minus(spec) -= term;
                }
            }
        }
    }
 
    // Determine the corresponding energy and temperature RHS.

    Array1D<Real, 1, NumSpec> dYdt;
    for (int n = 1; n <= NumSpec; ++n) {
        dYdt(n) = (f_plus(n) - f_minus(n)) * aion_inv[n - 1];
    }

    dedt = ener_rhs(state, dYdt);

    state.n_rhs += 1;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void actual_integrator (burn_t& state, Real dt)
{
    initialize_state(state);

    Real xn_in[NumSpec];
    for (int n = 0; n < NumSpec; ++n) {
        xn_in[n] = state.xn[n];
    }
    Real T_in = state.T;
    Real e_in = state.e;

    Real t = 0.0_rt;
    int num_timesteps = 0;

    Real dt_sub = dt * dt_init_fraction;

    // When checking the integration time to see if we're done,
    // be careful with roundoff issues.

    const Real timestep_safety_factor = 1.0e-12_rt;

    while (t < (1.0_rt - timestep_safety_factor) * dt && num_timesteps < ode_max_steps) {

        // Allow the timestep to attempt to grow.

        dt_sub *= dt_max_change_factor;

        // Make a copy of the initial state since we may iterate on the timestep.

        burn_t state_0 = state;

        // Evaluate the RHS.

        Array1D<Real, 1, NumSpec> f_minus, f_plus;
        Real dedt;

        evaluate_rhs(state_0, f_minus, f_plus, dedt);

        bool mass_constraint_satisfied = false;

        while (!mass_constraint_satisfied) {

            // Update the state with the exact QSS solution.

            Real sum_X = 0.0_rt;

            for (int n = 1; n <= NumSpec; ++n) {
                Real c = f_plus(n);
                Real d = f_minus(n);
                Real tau = state_0.xn[n-1] / amrex::max(d, 1.0e-50_rt);
                state.xn[n-1] = state_0.xn[n-1] * std::exp(-dt_sub / tau) + c * tau * (1.0_rt - std::exp(-dt_sub / tau));

                sum_X += state.xn[n-1];
            }

            // Update the energy with simple forward Euler.

            state.e = state_0.e + dt_sub * dedt;

            if (std::abs(1.0_rt - sum_X) / sum_X <= sum_X_tolerance) {
                mass_constraint_satisfied = true;
            }
            else {
                dt_sub *= dt_cut_factor;
            }

        }

        clean_state(state);

        t += dt_sub;
        ++num_timesteps;

        std::cout << "t = " << t << " dt = " << dt_sub << " timestep = " << num_timesteps << " ";
        for (int n = 0; n < NumSpec; ++n) {
            std::cout << state.xn[n] << " ";
        }
        std::cout << "\n";

    }

    if (num_timesteps >= ode_max_steps) {
        state.success = false;
    }

#ifndef AMREX_USE_CUDA
    if (burner_verbose) {
        // Print out some integration statistics, if desired.
        std::cout <<  "integration summary: " << std::endl;
        std::cout <<  "dens: " << state.rho << " temp: " << state.T << std::endl;
        std::cout <<  "energy released: " << state.e - e_in << std::endl;
        std::cout <<  "number of steps taken: " << num_timesteps << std::endl;
        std::cout <<  "number of f evaluations: " << state.n_rhs << std::endl;
    }
#endif

    // If we failed, print out the current state of the integration.

    if (!state.success) {
#ifndef AMREX_USE_CUDA
        std::cout << "ERROR: integration failed in net" << std::endl;
        std::cout << "time = " << t << std::endl;
        std::cout << "dens = " << state.rho << std::endl;
        std::cout << "temp start = " << T_in << std::endl;
        std::cout << "xn start = ";
        for (int n = 0; n < NumSpec; ++n) {
            std::cout << xn_in[n] << " ";
        }
        std::cout << std::endl;
        std::cout << "temp current = " << state.T << std::endl;
        std::cout << "xn current = ";
        for (int n = 0; n < NumSpec; ++n) {
            std::cout << state.xn[n] << " ";
        }
        std::cout << std::endl;
        std::cout << "energy generated = " << state.e - e_in << std::endl;
#endif
    }

}

#endif
