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
void evaluate_rhs (burn_t& state, const rate_fr_t& rr, Array1D<Real, 1, NumSpec>& dXdt, Real& dedt,
                   Array1D<Real, 1, NumSpec>& X_eq, const Real& dt)
{
    Array1D<Real, 1, NumSpec> y;

    for (int spec = 1; spec <= NumSpec; ++spec) {
        y(spec) = state.xn[spec-1] * aion_inv[spec-1];
    }

    // Number of contributions to the equilibrium value of X.
    Array1D<int, 1, NumSpec> N_eq;

    for (int spec = 1; spec <= NumSpec; ++spec) {
        dXdt(spec) = 0.0;
        X_eq(spec) = 0.0;
        N_eq(spec) = 0;
    }

    // Loop through reaction pairs.

    for (int rate = 1; rate <= Rates::NumRatesFR; ++rate) {
        rhs_t rhs_data = RHS::rhs_data(rate);

        // First, compute the Y * rate component of both the forward and
        // reverse reactions, which is the same regardless of which species
        // we're producing or consuming.

        Real forward_term = rr.forward_rates(rate);

        if (rhs_data.species_A >= 0) {
            forward_term *= std::pow(y(rhs_data.species_A), rhs_data.exponent_A_forward);
        }

        if (rhs_data.species_B >= 0) {
            forward_term *= std::pow(y(rhs_data.species_B), rhs_data.exponent_B_forward);
        }

        if (rhs_data.species_C >= 0) {
            forward_term *= std::pow(y(rhs_data.species_C), rhs_data.exponent_C_forward);
        }

        Real reverse_term = rr.reverse_rates(rate);

        if (rhs_data.species_D >= 0) {
            reverse_term *= std::pow(y(rhs_data.species_D), rhs_data.exponent_D_reverse);
        }

        if (rhs_data.species_E >= 0) {
            reverse_term *= std::pow(y(rhs_data.species_E), rhs_data.exponent_E_reverse);
        }

        if (rhs_data.species_F >= 0) {
            reverse_term *= std::pow(y(rhs_data.species_F), rhs_data.exponent_F_reverse);
        }

        // Determine if all species involved are sufficiently close to their equilibrium value.
        // If so, set this reaction pair to zero, then add the equilibrium mass fraction to the
        // running total of equilibrium mass fractions. We will average that out at the end, and
        // then directly enforce the average equilibrium value after the update.
        //
        // Now let's look at how to construct the equilibrium value. We'll visit an example first.
        // The update for a species A in the case of the reaction A + B <=> D is
        // dY_A/dt = -Y_A * Y_B * k_f + Y_D * k_r
        // where k_f is the forward rate and k_r is the reverse rate. 
        //
        // We define the following constants, where Y is evaluated at the current time:
        //
        // c_1 = Y_B - Y_A
        // c_2 = Y_B + Y_D
        //
        // These scalars are conserved by construction, as a result of conservation of nucleon number.
        //
        // The ODE can then be written as
        //
        // dY_A/dt = a * Y_A**2 + b * Y_A + c
        //
        // where
        //
        // a = -k_f
        // b = -(c_1 * k_f + k_r)
        // c = k_r * (c_2 - c_1)
        //
        // The equilibrium solution (t -> infinity) of this ODE is
        //
        // Y_A_eq = -(1/2a) * (b + sqrt(b^2 - 4*a*c))
        //
        // By conservation of nucleon number, we can then directly write down the equilibrium
        // abundances of the other two species:
        //
        // Y_B_eq = c_1 + Y_A_eq
        // Y_D_eq = c_2 - Y_B_eq
        //
        // So a very simple way to check whether we're in equilibrium is whether all three
        // species are within a certain tolerance of these equilibrium values.
        //
        // In the general case, there are three species on each side and the Y terms may have
        // nonlinear exponents. In that case the expression is a more involved polynomial equation.
        // We will solve below a few common cases where we can analytically write down the solution.
        // It is intended to cover many of the common reaction pairs. If we have not handled a particular
        // case, then it simply goes on to the normal integration scheme unmodified.
        //
        // We want to be careful that we are not triggering on the uninteresting case Y_A = Y_B = Y_D = 0.
        // So we'll verify that at least one of our species is above the integration tolerance threshold.

        const Real equilibrium_tol = 0.005_rt;

        if (rhs_data.exponent_A_forward == 1 && rhs_data.exponent_B_forward == 1 && rhs_data.exponent_C_forward == 0 &&
            rhs_data.exponent_D_reverse == 1 && rhs_data.exponent_E_reverse == 0 && rhs_data.exponent_F_reverse == 0)
        {
            // n_A * Y_A + n_B * Y_B <=> n_D * Y_D
            //
            // Conserved scalars:
            //
            // c_1 = n_B * Y_B - n_A * Y_A
            // c_2 = n_B * Y_B + n_D * Y_D
            //
            // ODE constants for d(Y_A)/dt:
            //
            // a = -(n_A * n_A / n_B) * k_f
            // b = (n_A / n_B) * c_1 * k_f - (n_A * n_A / n_D) * k_r
            // c = (n_A / n_D) * c_2 * k_r
            //
            // Equilibrium values:
            //
            // Y_A_eq = -(1/2a) * (b + sqrt(b * b - 4 * a * c))
            // Y_B_eq = (n_A * Y_A_eq - c_1) / n_B
            // Y_D_eq = (c_2 - n_A * Y_A_eq) / n_D

            Real Y_A = y(rhs_data.species_A);
            Real Y_B = y(rhs_data.species_B);
            Real Y_D = y(rhs_data.species_D);

            Real X_A = state.xn[rhs_data.species_A - 1];
            Real X_B = state.xn[rhs_data.species_B - 1];
            Real X_D = state.xn[rhs_data.species_D - 1];

            Real k_f = amrex::max(rr.forward_rates(rate), 1.0e-30_rt);
            Real k_r = amrex::max(rr.reverse_rates(rate), 1.0e-30_rt);

            Real n_A = static_cast<Real>(rhs_data.consumed_A_forward);
            Real n_B = static_cast<Real>(rhs_data.consumed_B_forward);
            Real n_D = static_cast<Real>(rhs_data.produced_D_forward);

            Real c_1 = n_A * Y_A - n_B * Y_B;
            Real c_2 = n_A * Y_A + n_D * Y_D;

            Real a = -(n_A * n_A / n_B) * k_f;
            Real b = (n_A / n_B) * c_1 * k_f - (n_A * n_A / n_D) * k_r;
            Real c = (n_A / n_D) * c_2 * k_r;

            Real Y_A_eq = -(0.5_rt / a) * (b + std::sqrt(b * b - 4.0_rt * a * c));
            Real Y_B_eq = (n_A * Y_A_eq - c_1) / n_B;
            Real Y_D_eq = (c_2 - n_A * Y_A_eq) / n_D;

            Real X_A_eq = Y_A_eq * aion[rhs_data.species_A - 1];
            Real X_B_eq = Y_B_eq * aion[rhs_data.species_B - 1];
            Real X_D_eq = Y_D_eq * aion[rhs_data.species_D - 1];

            Real Y_A_dt = Y_A; // + dt * n_A * (-forward_term + reverse_term);
            Real Y_B_dt = Y_B; // + dt * n_B * (-forward_term + reverse_term);
            Real Y_D_dt = Y_D; // + dt * n_D * (-reverse_term + forward_term);

            Real X_A_dt = Y_A_dt * aion[rhs_data.species_A - 1];
            Real X_B_dt = Y_B_dt * aion[rhs_data.species_B - 1];
            Real X_D_dt = Y_D_dt * aion[rhs_data.species_D - 1];

//            std::cout << "Method A rate = " << rate << " " << X_A_dt << " " << X_A_eq << " " << X_B_dt << " " << X_B_eq << " "
//                          << X_D_dt << " " << X_D_eq << " " << "\n";

            if (std::abs(Y_A_dt - Y_A_eq) < equilibrium_tol * Y_A_eq &&
                std::abs(Y_B_dt - Y_B_eq) < equilibrium_tol * Y_B_eq &&
                std::abs(Y_D_dt - Y_D_eq) < equilibrium_tol * Y_D_eq &&
                (X_A >= atol_spec || X_B >= atol_spec || X_D >= atol_spec))
            {               
                std::cout << "Equilibrium A\n";
                std::cout << "rate = " << rate << " " << X_A_dt << " " << X_A_eq << " " << X_B_dt << " " << X_B_eq << " "
                          << X_D_dt << " " << X_D_eq << " " << " sumX orig " << X_A_dt + X_B_dt + X_D_dt << " sum X new " << X_A_eq + X_B_eq + X_D_eq
                          << " dY_A = " << Y_A_dt - Y_A_eq << " dY_B = " << Y_B_dt - Y_B_eq << " dY_D = " << Y_D_dt - Y_D_eq << " sum dY = "
                          << Y_A_dt - Y_A_eq + Y_B_dt - Y_B_eq + Y_D_dt - Y_D_eq << "\n";
                forward_term = 0.0_rt;
                reverse_term = 0.0_rt;

                X_eq(rhs_data.species_A) += X_A_eq;
                X_eq(rhs_data.species_B) += X_B_eq;
                X_eq(rhs_data.species_D) += X_D_eq;

                N_eq(rhs_data.species_A) += 1;
                N_eq(rhs_data.species_B) += 1;
                N_eq(rhs_data.species_D) += 1;
            }
        }
        else if (rhs_data.exponent_A_forward == 1 && rhs_data.exponent_B_forward == 1 && rhs_data.exponent_C_forward == 0 &&
                 rhs_data.exponent_D_reverse == 1 && rhs_data.exponent_E_reverse == 1 && rhs_data.exponent_F_reverse == 0)
        {
            // n_A * Y_A + n_B * Y_B <=> n_D * Y_D + n_E * Y_E
            //
            // Conserved scalars:
            //
            // c_1 = n_A * Y_A - n_B * Y_B
            // c_2 = n_A * Y_A + n_D * Y_D
            // c_3 = n_A * Y_A + n_E * Y_E
            //
            // ODE constants for d(Y_A)/dt:
            //
            // a = -(n_A * n_A / n_B) * k_f + (n_A * n_A * n_A / (n_D * n_E)) * k_r
            // b = (n_A / n_B) * c_1 * k_f - (n_A * n_A / (n_D * n_E)) * (c_2 + c_3) * k_r
            // c = (n_A / (n_D * n_E)) * c_2 * c_3 * k_r
            //
            // Equilibrium values:
            //
            // Y_A_eq = -(1/2a) * (b + sqrt(b * b - 4 * a * c))
            // Y_B_eq = (n_A * Y_A_eq - c_1) / n_B
            // Y_D_eq = (c_2 - n_A * Y_A_eq) / n_D
            // Y_E_eq = (c_3 - n_A * Y_A_eq) / n_E

            Real Y_A = y(rhs_data.species_A);
            Real Y_B = y(rhs_data.species_B);
            Real Y_D = y(rhs_data.species_D);
            Real Y_E = y(rhs_data.species_E);

            Real X_A = state.xn[rhs_data.species_A - 1];
            Real X_B = state.xn[rhs_data.species_B - 1];
            Real X_D = state.xn[rhs_data.species_D - 1];
            Real X_E = state.xn[rhs_data.species_E - 1];

            Real k_f = amrex::max(rr.forward_rates(rate), 1.0e-30_rt);
            Real k_r = amrex::max(rr.reverse_rates(rate), 1.0e-30_rt);

            Real n_A = static_cast<Real>(rhs_data.consumed_A_forward);
            Real n_B = static_cast<Real>(rhs_data.consumed_B_forward);
            Real n_D = static_cast<Real>(rhs_data.produced_D_forward);
            Real n_E = static_cast<Real>(rhs_data.produced_E_forward);

            Real c_1 = n_A * Y_A - n_B * Y_B;
            Real c_2 = n_A * Y_A + n_D * Y_D;
            Real c_3 = n_A * Y_A + n_E * Y_E;

            Real a = -(n_A * n_A / n_B) * k_f + (n_A * n_A * n_A / (n_D * n_E)) * k_r;
            Real b = (n_A / n_B) * c_1 * k_f - (n_A * n_A / (n_D * n_E)) * (c_2 + c_3) * k_r;
            Real c = (n_A / (n_D * n_E)) * c_2 * c_3 * k_r;

            Real Y_A_eq = -(0.5_rt / a) * (b + std::sqrt(b * b - 4.0_rt * a * c));
            Real Y_B_eq = (n_A * Y_A_eq - c_1) / n_B;
            Real Y_D_eq = (c_2 - n_A * Y_A_eq) / n_D;
            Real Y_E_eq = (c_3 - n_A * Y_A_eq) / n_E;

            Real X_A_eq = Y_A_eq * aion[rhs_data.species_A - 1];
            Real X_B_eq = Y_B_eq * aion[rhs_data.species_B - 1];
            Real X_D_eq = Y_D_eq * aion[rhs_data.species_D - 1];
            Real X_E_eq = Y_E_eq * aion[rhs_data.species_E - 1];

            Real Y_A_dt = Y_A; // + dt * n_A * (-forward_term + reverse_term);
            Real Y_B_dt = Y_B; // + dt * n_B * (-forward_term + reverse_term);
            Real Y_D_dt = Y_D; // + dt * n_D * (-reverse_term + forward_term);
            Real Y_E_dt = Y_E; // + dt * n_E * (-reverse_term + forward_term);

            Real X_A_dt = Y_A_dt * aion[rhs_data.species_A - 1];
            Real X_B_dt = Y_B_dt * aion[rhs_data.species_B - 1];
            Real X_D_dt = Y_D_dt * aion[rhs_data.species_D - 1];
            Real X_E_dt = Y_E_dt * aion[rhs_data.species_E - 1];            

//            std::cout << "Method B rate = " << rate << " " << X_A_dt << " " << X_A_eq << " " << X_B_dt << " " << X_B_eq << " "
//                      << X_D_dt << " " << X_D_eq << " " << X_E_dt << " " << X_E_eq << " " << a << " " << b << " " << c << "\n";

            if (std::abs(Y_A_dt - Y_A_eq) < equilibrium_tol * Y_A_eq &&
                std::abs(Y_B_dt - Y_B_eq) < equilibrium_tol * Y_B_eq &&
                std::abs(Y_D_dt - Y_D_eq) < equilibrium_tol * Y_D_eq &&
                std::abs(Y_E_dt - Y_E_eq) < equilibrium_tol * Y_E_eq &&
                (X_A >= atol_spec || X_B >= atol_spec || X_D >= atol_spec || X_E >= atol_spec))
            {               
//                std::cout << "Equilibrium B rate\n";

                forward_term = 0.0_rt;
                reverse_term = 0.0_rt;

                X_eq(rhs_data.species_A) += X_A_eq;
                X_eq(rhs_data.species_B) += X_B_eq;
                X_eq(rhs_data.species_D) += X_D_eq;
                X_eq(rhs_data.species_E) += X_E_eq;

                N_eq(rhs_data.species_A) += 1;
                N_eq(rhs_data.species_B) += 1;
                N_eq(rhs_data.species_D) += 1;
                N_eq(rhs_data.species_E) += 1;
            }
        }

        // Loop through species and apply the term.
        for (int spec = 1; spec <= NumSpec; ++spec) {

            Real forward_term_spec = 0.0_rt;
            Real reverse_term_spec = 0.0_rt;

            if (rhs_data.species_A == spec) {
                forward_term_spec = -rhs_data.consumed_A_forward * forward_term;
                reverse_term_spec =  rhs_data.produced_A_reverse * reverse_term;
            }
            else if (rhs_data.species_B == spec) {
                forward_term_spec = -rhs_data.consumed_B_forward * forward_term;
                reverse_term_spec =  rhs_data.produced_B_reverse * reverse_term;
            }
            else if (rhs_data.species_C == spec) {
                forward_term_spec = -rhs_data.consumed_C_forward * forward_term;
                reverse_term_spec =  rhs_data.produced_C_reverse * reverse_term;
            }
            else if (rhs_data.species_D == spec) {
                forward_term_spec =  rhs_data.produced_D_forward * forward_term;
                reverse_term_spec = -rhs_data.consumed_D_reverse * reverse_term;
            }
            else if (rhs_data.species_E == spec) {
                forward_term_spec =  rhs_data.produced_E_forward * forward_term;
                reverse_term_spec = -rhs_data.consumed_E_reverse * reverse_term;
            }
            else if (rhs_data.species_F == spec) {
                forward_term_spec =  rhs_data.produced_F_forward * forward_term;
                reverse_term_spec = -rhs_data.consumed_F_reverse * reverse_term;
            }

            // Scale by aion to get dX/dt instead of dY/dt.
            forward_term_spec *= aion[spec - 1];
            reverse_term_spec *= aion[spec - 1];

//            if (spec == 6) {
//                std::cout << "Silicon update " << rate << " " << forward_term_spec + reverse_term_spec << "\n";
//            }

//            if (spec == 4) {
//                std::cout << "Neon update " << rate << " " << forward_term_spec << " " << reverse_term_spec << "\n";
//            }

            dXdt(spec) += forward_term_spec + reverse_term_spec;
        }
    }

    // Determine the corresponding energy and temperature RHS.

    Array1D<Real, 1, NumSpec> dYdt;
    for (int n = 1; n <= NumSpec; ++n) {
        dYdt(n) = dXdt(n) * aion_inv[n - 1];
    }

    dedt = ener_rhs(state, dYdt);

    // Average out the equilibrium values.

    for (int n = 1; n <= NumSpec; ++n) {
        if (N_eq(n) > 0) {
            X_eq(n) /= N_eq(n);
//            std::cout << "X_eq = " << n << " " << X_eq(n) << "\n";
        }
    }

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

    // When checking the integration time to see if we're done,
    // be careful with roundoff issues.

    const Real timestep_safety_factor = 1.0e-12_rt;

    // Start the guess for the timestepping by evaluating the RHS and
    // choose a dt that is some fraction of (X / dX/dt).

    Array1D<Real, 1, NumSpec> dXdt;
    Real dedt;

    Array1D<Real, 1, NumSpec> X_eq;

    rate_t rr_old;
    evaluate_rates(state, rr_old);

    rate_fr_t rr;

    // Convert old-style rates to new-style
    for (int j = 1; j <= Rates::NumRatesFR; ++j) {
        rr.forward_rates(j) = rr_old.rates(2 * j - 1);
        rr.reverse_rates(j) = rr_old.rates(2 * j);
    }

    evaluate_rhs(state, rr, dXdt, dedt, X_eq, dt);

    Real dt_sub = dt;

    for (int n = 1; n <= NumSpec; ++n) {
        if (xn_in[n-1] >= atol_spec) {
            dt_sub = amrex::min(dt_sub, xn_in[n - 1] / amrex::max(1.0e-30_rt, std::abs(dXdt(n))));
        }
    }

    dt_sub *= dt_init_fraction;

    dt_sub = amrex::min(dt_sub, ode_max_dt);

    int num_timesteps = 0;

    while (t < (1.0_rt - timestep_safety_factor) * dt && num_timesteps < ode_max_steps)
    {
        // Start the step with a guess that is a small factor above the previous timestep.

        dt_sub *= dt_max_change_factor;

        dt_sub = amrex::min(dt_sub, ode_max_dt);

        // Prevent the timestep from overshooting the final time

        if (t + dt_sub > dt) {
            dt_sub = dt - t;
        }

        // Save the initial state in case we need to revert

        burn_t state_0 = state;

        // Evaluate the rates

        evaluate_rates(state, rr_old);

        // Convert old-style rates to new-style

        for (int j = 1; j <= Rates::NumRatesFR; ++j) {
            rr.forward_rates(j) = rr_old.rates(2 * j - 1);
            rr.reverse_rates(j) = rr_old.rates(2 * j);
        }

        // Loop until we satisfy the timestep constraint.

        int timestep_iter = 0;

        for (timestep_iter = 0; timestep_iter < num_timestep_iters; ++timestep_iter)
        {
            // Evaluate the RHS

            evaluate_rhs(state, rr, dXdt, dedt, X_eq, dt_sub);

            // Instantaneously apply equilibrium constraints

            Array1D<Real, 1, NumSpec> dY_eq;
            for (int n = 1; n <= NumSpec; ++n) {
                if (X_eq(n) > 0.0_rt) {
                    dY_eq(n) = (X_eq(n) - state.xn[n-1]) * aion_inv[n-1];
                    std::cout << "Fixing up " << n << " from " << state.xn[n-1] << "  to " << X_eq(n) << " dX = " << (X_eq(n) - state.xn[n-1]) << " "
                              << " dY = " << dY_eq(n) << " de = " << dY_eq(n) * iso7::mion(n) * C::Legacy::enuc_conv2 << "\n";
                    state.xn[n-1] = X_eq(n);
                } else {
                    dY_eq(n) = 0.0_rt;
                }
            }

            // Add the energy update due to this reconfiguration

            if (integrate_energy) {
                Real de = 0.0_rt;

                for (int n = 1; n <= NumSpec; ++n) {
                    de += dY_eq(n) * iso7::mion(n);
                }

                de *= C::Legacy::enuc_conv2;

                state.e += de;
            }

            // Apply the RHS

            for (int n = 1; n <= NumSpec; ++n) {
                state.xn[n-1] += dXdt(n) * dt_sub;
            }

            if (integrate_energy) {
                state.e += dedt * dt_sub;
            }

            // Check if we violated the timestep limiter

            bool valid_dt = true;

            for (int n = 1; n <= NumSpec; ++n) {
                if (state_0.xn[n-1] >= atol_spec) {
                    if (state.xn[n-1] > maximum_timestep_change_factor * state_0.xn[n-1] ||
                        state.xn[n-1] < 1.0_rt / maximum_timestep_change_factor * state_0.xn[n-1]) {
                        std::cout << "dt limiter " << n << " " << state.xn[n-1] << " " << state_0.xn[n-1] << " " << state.xn[n-1] / state_0.xn[n-1] << "\n";
                        valid_dt = false;
                        break;
                    }
                }
            }

            if (valid_dt) {
                break;
            }
            else {
                state = state_0;
                dt_sub *= dt_cut_factor;
            }
        }

        clean_state(state);

        t += dt_sub;
        ++num_timesteps;

        std::cout << "Time = " << t << " nsteps = " << num_timesteps << " X_He = " << state.xn[0] << " X_Si = " << state.xn[6] << " T = " << state.T << "\n";

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
