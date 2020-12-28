#ifndef actual_integrator_H
#define actual_integrator_H

#include <network.H>
#include <actual_network.H>
#include <actual_rhs.H>
#include <burn_type.H>
#include <rate_type.H>
#include <eos_type.H>
#include <eos.H>
#include <extern_parameters.H>

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void clean_state (burn_t& state)
{
    // Renormalize the abundances.

    normalize_abundances_burn(state);

    // Evaluate the EOS to get T from e.

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
    // to fill the rest of the thermodynamic variables.

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

    Real T_in = state.T;
    Real e_in = state.e;
    Real xn_in[NumSpec];
    for (int n = 0; n < NumSpec; ++n) {
        xn_in[n] = state.xn[n];
    }

    Real t = 0.0;

    // Start the guess for the timestepping by evaluating the RHS and
    // choose a dt that is some fraction of (X / dX/dt). The exception
    // will be cases where F_plus >> F_minus (see Mott 1999, Equation 3.40).

    Array1D<Real, 1, NumSpec> f_minus_init, f_plus_init;
    Real dedt_init;

    evaluate_rhs(state, f_minus_init, f_plus_init, dedt_init);

    Real dt_sub = dt;

    for (int n = 1; n <= NumSpec; ++n) {
        if (f_plus_init(n) >= 1.e2 * f_minus_init(n)) {
            dt_sub = amrex::min(dt_sub, xn_in[n - 1] / f_minus_init(n));
        }
        else {
            dt_sub = amrex::min(dt_sub, xn_in[n - 1] / std::abs(f_plus_init(n) - f_minus_init(n)));
        }
    }

    dt_sub *= dt_init_fraction;

    dt_sub = amrex::min(dt_sub, ode_max_dt);

    // When checking the integration time to see if we're done,
    // be careful with roundoff issues.

    const Real timestep_safety_factor = 1.0e-12_rt;

    int num_timesteps = 0;

    while (t < (1.0_rt - timestep_safety_factor) * dt && num_timesteps < ode_max_steps)
    {
        // Start the step with a guess that is a small factor above the previous timestep.

        dt_sub *= dt_max_change_factor;

        dt_sub = amrex::min(dt_sub, ode_max_dt);

        // Prevent the timestep from overshooting the final time.

        if (t + dt_sub > dt) {
            dt_sub = dt - t;
        }

        // Make a copy of the old-time state, which does not change in the iterations.

        burn_t state_0 = state;

        // Evaluate the initial RHS.

        Array1D<Real, 1, NumSpec> f_minus_0, f_plus_0;
        Real dedt_0;

        evaluate_rhs(state_0, f_minus_0, f_plus_0, dedt_0);

        // Calculate the dual timestepping update. This follows Morii and Shima (2020).
        // Given our choice for dt == h, we solve Equation 22 iteratively for the converged
        // solution to the species update (terms will be defined below).
        //
        // X^{k+1} = X^{k} + h' * [(1 - theta) * F^{k} + theta * F^{0} - (X^{k} - X^{0}) / h]
        //
        // F is the RHS and is assumed to be written in the form
        //
        // F_plus - F_minus == F_plus - X / tau
        //
        // theta is a parameter that determines "how implicit" we are, with theta == 0 corresponding
        // to backward Euler and theta == 1 corresponding to forward Euler. The optimal choice is
        //
        // theta == (1 - e^(-h/tau) - (h / tau) * e^(-h / tau)) / [(h / tau) * (1 - e^(-h / tau))]
        //
        // h' is the "dual" timestep and is set by the optimal choice for convergence,
        //
        // h' = tau * h / [(1 - theta) * h + tau]
        //
        // The superscript is the iteration index, and a zero subscript corresponds to the original
        // state at time-level n.
        //
        // The paper is not clear on whether the various parameters like h' and theta (which depend
        // on the RHS) should update in each inner iteration. We will allow them to vary.
        //
        // If we fail to converge within a certain number of iterations, we cut dt and try again.
        //
        // We do not integrate temperature, and instead get it from the EOS.

        // Iterate over dt.

        int timestep_iter = 0;

        for (timestep_iter = 0; timestep_iter < num_timestep_iters; ++timestep_iter)
        {
            // Do at most a fixed number of iterations. In each iteration we update the state
            // according to Equation 22 and then measure the difference between the old state
            // and the new state. We consider ourselves converged if the maximum relative diff
            // between the old state and the new state is sufficiently small.

            state = state_0;

            int inner_iter = 0;

            Array1D<Real, 1, NumSpec> f_minus, f_plus;
            Real dedt;

            bool success = true;

            for (inner_iter = 0; inner_iter < num_inner_iters; ++inner_iter)
            {
                evaluate_rhs(state, f_minus, f_plus, dedt);

                Real max_diff = 0.0_rt;

                for (int n = 1; n <= NumSpec; ++n) {
                    Real tau = state.xn[n-1] / amrex::max(f_minus(n), 1.0e-50_rt);

                    Real h = dt_sub;
                    Real ht = h / amrex::max(tau, 1.0e-50_rt);
                    Real eht = std::exp(-ht);

                    Real theta;

                    // Avoid roundoff issues in the calculation of theta.

                    if (ht < 1.e-6_rt) {
                        theta = 0.5_rt;
                    }
                    else if (ht > 1.e6_rt) {
                        theta = 0.0_rt;
                    }
                    else {
                        theta = (1.0_rt - eht - ht * eht) / amrex::max(ht * (1.0_rt - eht), 1.0e-50_rt);
                    }

                    Real alpha = f_plus(n);
                    Real beta = tau;
                    Real alpha_prime = alpha - theta * state_0.xn[n-1] / beta + state_0.xn[n-1] / h;
                    Real beta_prime = beta * h / ((1.0_rt - theta) * h + beta);

                    Real h_prime = beta * h / ((1.0_rt - theta) * h + beta);

                    Real dX = h_prime * ((1.0_rt - theta) * (f_plus(n) - f_minus(n)) +
                                         theta * (f_plus_0(n) - f_minus_0(n)) -
                                         (state.xn[n-1] - state_0.xn[n-1]) / h);

                    max_diff = amrex::max(max_diff, dX / state.xn[n-1]);

                    state.xn[n-1] = state.xn[n-1] + dX;

                    // Check that our changes don't create unreasonable values for X.

                    if (state.xn[n-1] < -species_tolerance || state.xn[n-1] > 1.0_rt + species_tolerance) {
                        success = false;
                        break;
                    }
                }

                // Shortcut out of the loop if we failed.

                if (!success) {
                    break;
                }

                // Check that the sum of X is sufficiently close to 1.

                Real sum_X = 0.0_rt;

                for (int n = 1; n <= NumSpec; ++n) {
                    sum_X += state.xn[n-1];
                }

                if (std::abs(sum_X - 1.0_rt) > species_tolerance) {
                    success = false;
                    break;
                }

                // Compute the final energy update with a simple second-order approximation.

                state.e = state_0.e + 0.5_rt * dt_sub * (dedt_0 + dedt);

                clean_state(state);

                // If we've converged, we can stop.

                if (max_diff < rtol_spec) {
                    success = true;
                    break;
                }
            }

            // If we didn't converge, cut the timestep and try again.

            if (inner_iter >= num_inner_iters || !success) {
                dt_sub *= dt_cut_factor;
                continue;
            }

            // If the species changed by more than the allowed amount, cut the timestep and try again.
            // Note that the check above was on whether the inner iteration was converged, while this
            // is enforcement on accuracy -- the tighter we make this constraint, the more accurate
            // the evolution will be, but we will likely take more timesteps.

            Real max_diff = 0.0_rt;

            for (int n = 1; n <= NumSpec; ++n) {
                max_diff = amrex::max(max_diff, std::abs(state.xn[n-1] - state_0.xn[n-1]) / state_0.xn[n-1]);
            }

            if (max_diff > maximum_timestep_change_factor) {
                dt_sub *= dt_cut_factor;
                continue;
            }

            // If we made it to this point, we're done.
            break;
        }

        // If we didn't get a satisfactory solution in the fixed number of iterations, the integration failed.

        if (timestep_iter >= num_timestep_iters) {
            state.success = false;
            break;
        }

        t += dt_sub;
        ++num_timesteps;
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
