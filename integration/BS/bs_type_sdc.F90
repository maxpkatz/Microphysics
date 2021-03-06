module bs_type_module

  use amrex_constants_module, only: HALF, ONE, ZERO
  use amrex_fort_module, only : rt => amrex_real
  use sdc_type_module, only: SVAR, SVAR_EVOLVE
  use bs_rpar_indices, only: n_rpar_comps

  implicit none

  ! BS parameters -- see the discussion in 16.4
  integer, parameter :: KMAXX = 7
  integer :: nseq(KMAXX+1)

  integer, parameter :: bs_neqs = SVAR_EVOLVE

  type bs_t

     logical :: first
     real(rt) :: eps_old
     real(rt) :: dt_did
     real(rt) :: dt_next
     real(rt) :: a(KMAXX+1)
     real(rt) :: alpha(KMAXX, KMAXX)
     real(rt) :: t_new
     integer :: kmax
     integer :: kopt

     real(rt) :: y(SVAR_EVOLVE), ydot(SVAR_EVOLVE), jac(SVAR_EVOLVE, SVAR_EVOLVE)
     real(rt) :: ydot_a(SVAR_EVOLVE)
     real(rt) :: atol(SVAR_EVOLVE), rtol(SVAR_EVOLVE)
     real(rt) :: y_init(SVAR_EVOLVE)
     real(rt) :: u(n_rpar_comps), u_init(n_rpar_comps), udot_a(n_rpar_comps)
     real(rt) :: t, dt, tmax
     integer         :: n

     integer         :: n_rhs, n_jac
     
     integer :: i, j, k
     logical :: T_from_eden
     logical :: self_heat  ! needed only for compatibilty with no-SDC

  end type bs_t

  !$acc declare create(nseq)

contains

  subroutine clean_state(state)

    !$acc routine seq

    use extern_probin_module, only: SMALL_X_SAFE, renormalize_abundances
    use actual_network, only: nspec
#if defined(SDC_EVOLVE_ENERGY)
    use sdc_type_module, only: SFS, SEDEN, SEINT
    use bs_rpar_indices, only: irp_SRHO, irp_SMX, irp_SMZ
#elif defined(SDC_EVOLVE_ENTHALPY)
    use sdc_type_module, only: SFS, SENTH
    use bs_rpar_indices, only: irp_SRHO, irp_p0
#endif
    use eos_module, only: eos
    use eos_type_module, only: eos_input_rt, eos_t, eos_get_small_dens, eos_get_max_dens

    implicit none

    ! this should be larger than any reasonable temperature we will encounter
    real(rt) , parameter :: MAX_TEMP = 1.0e11_rt

    real(rt)  :: max_e, ke

    type (bs_t) :: state

    type (eos_t) :: eos_state

    ! Update rho, rho*u, etc.
    call fill_unevolved_variables(state)

    ! Ensure that partial densities (rho X_k) stay positive and less than rho.
    state % y(SFS:SFS+nspec-1) = max(min(state % y(SFS:SFS+nspec-1), &
                                         state % u(irp_SRHO)), &
                                     state % u(irp_SRHO) * SMALL_X_SAFE)

    ! Renormalize abundances as necessary.
    if (renormalize_abundances) then
       call renormalize_species(state)
    endif

#ifdef SDC_EVOLVE_ENERGY

    ! Ensure that internal energy never goes above the maximum limit
    ! provided by the EOS. Same for the internal energy implied by the
    ! total energy (which we get by subtracting kinetic energy).

    eos_state % rho = state % u(irp_SRHO)
    eos_state % T = MAX_TEMP
    eos_state % xn = state % y(SFS:SFS+nspec-1) / state % u(irp_SRHO)

    call eos(eos_input_rt, eos_state)

    max_e = eos_state % e

    state % y(SEINT) = min(state % u(irp_SRHO) * max_e, state % y(SEINT))

    ke = state % y(SEDEN) - HALF * sum(state % u(irp_SMX:irp_SMZ)**2) / state % u(irp_SRHO)

    state % y(SEDEN) = min(state % u(irp_SRHO) * max_e + ke, state % y(SEDEN))

#endif

  end subroutine clean_state



  subroutine fill_unevolved_variables(state)

    !$acc routine seq

    use actual_network, only: nspec
#if defined(SDC_EVOLVE_ENERGY)
    use sdc_type_module, only: SRHO, SMX, SMZ
    use bs_rpar_indices, only: irp_SRHO, irp_SMX, irp_SMZ
#elif defined(SDC_EVOLVE_ENTHALPY)
    use sdc_type_module, only: SFS
    use bs_rpar_indices, only: irp_SRHO
#endif

    implicit none

    type (bs_t) :: state

#if defined(SDC_EVOLVE_ENERGY)

    ! we are always integrating from t = 0, so there is no offset time
    ! needed here

    state % u(irp_SRHO) = state % u_init(irp_SRHO) + &
         state % udot_a(irp_SRHO) * state % t
    state % u(irp_SMX:irp_SMZ) = state % u_init(irp_SMX:irp_SMZ) + &
         state % udot_a(irp_SMX:irp_SMZ) * state % t

#elif defined(SDC_EVOLVE_ENTHALPY)

    ! Keep density consistent with the partial densities.
    state % u(irp_SRHO) = sum(state % y(SFS:SFS - 1 + nspec))

#endif

  end subroutine fill_unevolved_variables



  subroutine renormalize_species(state)

    !$acc routine seq

    use sdc_type_module, only: SFS
    use actual_network, only: nspec
    use bs_rpar_indices, only: irp_SRHO

    implicit none

    type (bs_t) :: state

    real(rt) :: nspec_sum


#ifdef SDC_EVOLVE_ENERGY

    ! Update rho, rho*u, etc.

    call fill_unevolved_variables(state)

    nspec_sum = sum(state % y(SFS:SFS+nspec-1)) / state % u(irp_SRHO)

    state % y(SFS:SFS+nspec-1) = state % y(SFS:SFS+nspec-1) / nspec_sum

#endif

  end subroutine renormalize_species



  subroutine sdc_to_bs(sdc, bs)

    !$acc routine seq

#if defined(SDC_EVOLVE_ENERGY)
    use sdc_type_module, only: sdc_t, SVAR_EVOLVE, SRHO, SMX, SMZ
    use bs_rpar_indices, only: irp_SRHO, irp_SMX, irp_SMZ
#elif defined(SDC_EVOLVE_ENTHALPY)
    use sdc_type_module, only: sdc_t, SVAR_EVOLVE
    use bs_rpar_indices, only: irp_SRHO, irp_p0
#endif

    implicit none

    type (sdc_t) :: sdc
    type (bs_t) :: bs

    bs % y = sdc % y(1:SVAR_EVOLVE)
    bs % y_init = bs % y
    bs % ydot_a = sdc % ydot_a(1:SVAR_EVOLVE)

#if defined(SDC_EVOLVE_ENERGY)

    ! these variables are not evolved
    bs % u(irp_SRHO) = sdc % y(SRHO)
    bs % u(irp_SMX:irp_SMZ) = sdc % y(SMX:SMZ)

    bs % udot_a(irp_SRHO) = sdc % ydot_a(SRHO)
    bs % udot_a(irp_SMX:irp_SMZ) = sdc % ydot_a(SMX:SMZ)


    bs % T_from_eden = sdc % T_from_eden

#elif defined(SDC_EVOLVE_ENTHALPY)

    bs % u(irp_SRHO) = sdc % rho
    bs % u(irp_p0) = sdc % p0
    bs % udot_a(:) = ZERO

#endif

    bs % u_init = bs % u

    bs % i = sdc % i
    bs % j = sdc % j
    bs % k = sdc % k

    bs % n_rhs = sdc % n_rhs
    bs % n_jac = sdc % n_jac

  end subroutine sdc_to_bs


  subroutine bs_to_sdc(sdc, bs)

    !$acc routine seq

#if defined(SDC_EVOLVE_ENERGY)
    use sdc_type_module, only: sdc_t, SRHO, SMX, SMZ
    use bs_rpar_indices, only: irp_SRHO, irp_SMX, irp_SMZ
#elif defined(SDC_EVOLVE_ENTHALPY)
    use sdc_type_module, only: sdc_t
    use bs_rpar_indices, only: irp_SRHO, irp_p0
#endif

    implicit none

    type (sdc_t) :: sdc
    type (bs_t) :: bs

    sdc % y(1:SVAR_EVOLVE) = bs % y

    call fill_unevolved_variables(bs)

#if defined(SDC_EVOLVE_ENERGY)

    sdc % y(SRHO) = bs % u(irp_SRHO)
    sdc % y(SMX:SMZ) = bs % u(irp_SMX:irp_SMZ)

#elif defined(SDC_EVOLVE_ENTHALPY)

    sdc % p0  = bs % u(irp_p0)
    sdc % rho = bs % u(irp_SRHO)

#endif

    sdc % i = bs % i
    sdc % j = bs % j
    sdc % k = bs % k

    sdc % n_rhs = bs % n_rhs
    sdc % n_jac = bs % n_jac

  end subroutine bs_to_sdc



  subroutine rhs_to_bs(bs, burn, ydot_react)

    !$acc routine seq

    use actual_network, only: nspec, aion
    use burn_type_module, only: burn_t, net_ienuc, neqs

#if defined(SDC_EVOLVE_ENERGY)
    use sdc_type_module, only: SVAR_EVOLVE, SEDEN, SEINT, SFS
#elif defined(SDC_EVOLVE_ENTHALPY)
    use sdc_type_module, only: SVAR_EVOLVE, SENTH, SFS
#endif

    use bs_rpar_indices, only: irp_SRHO

    implicit none

    type (bs_t) :: bs
    type (burn_t) :: burn
    real(rt)         :: ydot_react(neqs)

    integer :: n

    call fill_unevolved_variables(bs)

    ! Start with the contribution from the non-reacting sources

    bs % ydot = bs % ydot_a(1:SVAR_EVOLVE)

    ! Add in the reacting terms from the burn_t

    bs % ydot(SFS:SFS+nspec-1) = bs % ydot(SFS:SFS+nspec-1) + &
                                 bs % u(irp_SRHO) * ydot_react(1:nspec) * aion(1:nspec)

#if defined(SDC_EVOLVE_ENERGY)

    bs % ydot(SEINT) = bs % ydot(SEINT) + bs % u(irp_SRHO) * ydot_react(net_ienuc)
    bs % ydot(SEDEN) = bs % ydot(SEDEN) + bs % u(irp_SRHO) * ydot_react(net_ienuc)

#elif defined(SDC_EVOLVE_ENTHALPY)

    bs % ydot(SENTH) = bs % ydot(SENTH) + bs % u(irp_SRHO) * ydot_react(net_ienuc)

#endif

  end subroutine rhs_to_bs



  subroutine jac_to_bs(bs, jac)

    !$acc routine seq

    use network, only: nspec, aion, aion_inv
    use burn_type_module, only: net_ienuc, neqs

#if defined(SDC_EVOLVE_ENERGY)
    use sdc_type_module, only: SEDEN, SEINT, SFS
#elif defined(SDC_EVOLVE_ENTHALPY)
    use sdc_type_module, only: SENTH, SFS
#endif

    implicit none

    type (bs_t), intent(inout) :: bs
    real(rt)        , intent(in) :: jac(neqs, neqs)

    integer :: n

    ! Copy the Jacobian from the burn

#if defined(SDC_EVOLVE_ENERGY)

    bs % jac(SFS:SFS+nspec-1,SFS:SFS+nspec-1) = jac(1:nspec,1:nspec)
    bs % jac(SFS:SFS+nspec-1,SEDEN) = jac(1:nspec,net_ienuc)
    bs % jac(SFS:SFS+nspec-1,SEINT) = jac(1:nspec,net_ienuc)

    bs % jac(SEDEN,SFS:SFS+nspec-1) = jac(net_ienuc,1:nspec)
    bs % jac(SEDEN,SEDEN) = jac(net_ienuc,net_ienuc)
    bs % jac(SEDEN,SEINT) = jac(net_ienuc,net_ienuc)

    bs % jac(SEINT,SFS:SFS+nspec-1) = jac(net_ienuc,1:nspec)
    bs % jac(SEINT,SEDEN) = jac(net_ienuc,net_ienuc)
    bs % jac(SEINT,SEINT) = jac(net_ienuc,net_ienuc)

#elif defined(SDC_EVOLVE_ENTHALPY)

    bs % jac(SFS:SFS+nspec-1,SFS:SFS+nspec-1) = jac(1:nspec,1:nspec)
    bs % jac(SFS:SFS+nspec-1,SENTH) = jac(1:nspec,net_ienuc)

    bs % jac(SENTH,SFS:SFS+nspec-1) = jac(net_ienuc,1:nspec)
    bs % jac(SENTH,SENTH) = jac(net_ienuc,net_ienuc)

#endif

    ! Scale it to match our variables. We don't need to worry about the rho
    ! dependence, since every one of the SDC variables is linear in rho, so
    ! we just need to focus on the Y --> X conversion.

    do n = 1, nspec
       bs % jac(SFS+n-1,:) = bs % jac(SFS+n-1,:) * aion(n)
       bs % jac(:,SFS+n-1) = bs % jac(:,SFS+n-1) * aion_inv(n)
    enddo

  end subroutine jac_to_bs


  subroutine bs_to_burn(bs, burn)

    !$acc routine seq

    use actual_network, only: nspec
    use burn_type_module, only: burn_t, burn_to_eos, eos_to_burn
    use eos_type_module, only: eos_input_re, eos_input_rp, eos_input_rh, eos_t, eos_get_small_temp, eos_get_max_temp
    use eos_module, only: eos

#if defined(SDC_EVOLVE_ENERGY)

    use sdc_type_module, only: SEDEN, SEINT, SFS
    use bs_rpar_indices, only: irp_SRHO, irp_SMX, irp_SMZ

#elif defined(SDC_EVOLVE_ENTHALPY)

    use sdc_type_module, only: SENTH, SFS
    use bs_rpar_indices, only: irp_SRHO, irp_p0
    use meth_params_module, only: use_tfromp

#endif

    implicit none

    type (bs_t) :: bs
    type (burn_t) :: burn
    type (eos_t) :: eos_state

    real(rt) :: rhoInv, min_temp, max_temp

    ! Update rho, rho*u, etc.

    call fill_unevolved_variables(bs)

    rhoInv = ONE / bs % u(irp_SRHO)

    eos_state % rho = bs % u(irp_SRHO)
    eos_state % xn  = bs % y(SFS:SFS+nspec-1) * rhoInv

#if defined(SDC_EVOLVE_ENERGY)

    if (bs % T_from_eden) then
       eos_state % e = (bs % y(SEDEN) - HALF * rhoInv * sum(bs % u(irp_SMX:irp_SMZ)**2)) * rhoInv
    else
       eos_state % e = bs % y(SEINT) * rhoInv
    endif

#elif defined(SDC_EVOLVE_ENTHALPY)

    if (use_tfromp) then
       ! NOT SURE IF THIS MODE IS VALID
       eos_state % p = bs % u(irp_p0)
    else
       eos_state % h = bs % y(SENTH) * rhoInv
    endif

#endif

    ! Give the temperature an initial guess -- use the geometric mean
    ! of the minimum and maximum temperatures.

    call eos_get_small_temp(min_temp)
    call eos_get_max_temp(max_temp)

    eos_state % T = sqrt(min_temp * max_temp)

#if defined(SDC_EVOLVE_ENERGY)

    call eos(eos_input_re, eos_state)

#elif defined(SDC_EVOLVE_ENTHALPY)

    if (use_tfromp) then
       ! NOT SURE IF THIS MODE IS VALID
       ! used to be an Abort statement
       call eos(eos_input_rp, eos_state)
    else
       call eos(eos_input_rh, eos_state)
    endif

#endif

    call eos_to_burn(eos_state, burn)

    burn % time = bs % t

    burn % n_rhs = bs % n_rhs
    burn % n_jac = bs % n_jac

    burn % self_heat = bs % self_heat

#ifdef SDC_EVOLVE_ENTHALPY

    burn % p0 = bs % u(irp_p0)

#endif

#ifdef NONAKA_PLOT

    burn % i = bs % i
    burn % j = bs % j
    burn % k = bs % k

#endif

  end subroutine bs_to_burn

  subroutine dump_bs_state(bs)

    use burn_type_module, only: burn_t

    type (bs_t) :: bs
    type (burn_t) :: burn

    call bs_to_burn(bs, burn)

    print *, "time: ", bs % t
    print *, "T:    ", burn % T
    print *, "rho:  ", burn % rho
    print *, "X:    ", burn % xn(:)

  end subroutine dump_bs_state

end module bs_type_module
