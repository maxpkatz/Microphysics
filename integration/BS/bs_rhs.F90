module rhs_module

contains

  ! The rhs routine provides the right-hand-side for the BS solver.
  ! This is a generic interface that calls the specific RHS routine in the
  ! network you're actually using.

  subroutine f_rhs(bs)

    !$acc routine seq

    use actual_network, only: aion, nspec
    use amrex_fort_module, only : rt => amrex_real
    use burn_type_module, only: burn_t, net_ienuc, net_itemp
    use amrex_constants_module, only: ZERO, ONE
    use network_rhs_module, only: network_rhs
    use extern_probin_module, only: integrate_temperature, integrate_energy, react_boost
    use bs_type_module, only: bs_t, clean_state, renormalize_species, update_thermodynamics, &
                              burn_to_bs, bs_to_burn
    use bs_rpar_indices, only: irp_t0

    implicit none

    type (bs_t) :: bs

    ! We are integrating a system of
    !
    ! y(1:nspec)    = dX/dt
    ! y(net_itemp) = dT/dt
    ! y(net_ienuc) = denuc/dt

    ! Initialize the RHS to zero.

    bs % ydot(:) = ZERO

    ! Fix the state as necessary.
    call clean_state(bs)

    ! Update the thermodynamic quantities as necessary.
    call update_thermodynamics(bs)

    ! Call the specific network routine to get the RHS.
    call bs_to_burn(bs)
    call network_rhs(bs % burn_s, bs % ydot, bs % upar(irp_t0))

    ! We integrate X, not Y
    bs % ydot(1:nspec) = bs % ydot(1:nspec) * aion(1:nspec)

    ! Allow temperature and energy integration to be disabled.
    if (.not. integrate_temperature) then
       bs % ydot(net_itemp) = ZERO
    endif

    if (.not. integrate_energy) then
       bs % ydot(net_ienuc) = ZERO
    endif

    ! apply fudge factor:
    if (react_boost > ZERO) then
       bs % ydot(:) = react_boost * bs % ydot(:)
    endif

    call burn_to_bs(bs)

    ! Increment the evaluation counter.

    bs % burn_s % n_rhs = bs % burn_s % n_rhs + 1

  end subroutine f_rhs

end module rhs_module
