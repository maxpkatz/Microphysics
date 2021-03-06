module cuvode_dvstep_module

  use cuvode_parameters_module, only: VODE_NEQS
  use cuvode_types_module, only: dvode_t, VODE_LMAX, HMIN, HMXI
  use amrex_fort_module, only: rt => amrex_real
  use cuvode_dvset_module
  use cuvode_dvjust_module
  use cuvode_dvnlsd_module

  implicit none

contains

#if defined(AMREX_USE_CUDA) && !defined(AMREX_USE_GPU_PRAGMA)
  attributes(device) &
#endif
  subroutine advance_nordsieck(vstate)

    ! Effectively multiplies the Nordsieck history
    ! array by the Pascal triangle matrix.

    implicit none

    ! Declare arguments
    type(dvode_t), intent(inout) :: vstate

    ! Declare local variables
    integer :: k, j, i

    !$gpu

    do k = vstate % NQ, 1, -1
       do j = k, vstate % NQ
          do i = 1, VODE_NEQS
             vstate % YH(i, j) = vstate % YH(i, j) + vstate % YH(i, j+1)
          end do
       end do
    end do

  end subroutine advance_nordsieck


#if defined(AMREX_USE_CUDA) && !defined(AMREX_USE_GPU_PRAGMA)
  attributes(device) &
#endif
  subroutine retract_nordsieck(vstate)

    ! Undoes the Pascal triangle matrix multiplication
    ! implemented in subroutine advance_nordsieck.

    implicit none

    ! Declare arguments
    type(dvode_t), intent(inout) :: vstate

    ! Declare local variables
    integer :: k, j, i

    !$gpu

    do k = vstate % NQ, 1, -1
       do j = k, vstate % NQ
          do i = 1, VODE_NEQS
             vstate % YH(i, j) = vstate % YH(i, j) - vstate % YH(i, j+1)
          end do
       end do
    end do

  end subroutine retract_nordsieck


#if defined(AMREX_USE_CUDA) && !defined(AMREX_USE_GPU_PRAGMA)
  attributes(device) &
#endif
  subroutine dvstep(pivot, vstate)

    !$acc routine seq

    ! -----------------------------------------------------------------------
    !  DVSTEP performs one step of the integration of an initial value
    !  problem for a system of ordinary differential equations.
    !  DVSTEP calls subroutine VNLS for the solution of the nonlinear system
    !  arising in the time step.  Thus it is independent of the problem
    !  Jacobian structure and the type of nonlinear system solution method.
    !  DVSTEP returns a completion flag KFLAG (in COMMON).
    !  A return with KFLAG = -1 or -2 means either abs(H) = HMIN or 10
    !  consecutive failures occurred.  On a return with KFLAG negative,
    !  the values of TN and the YH array are as of the beginning of the last
    !  step, and H is the last step size attempted.
    !
    !  Communication with DVSTEP is done with the following variables:
    !
    !  Y      = An array of length N used for the dependent variable array.
    !  YH     = An LDYH by LMAX array containing the dependent variables
    !           and their approximate scaled derivatives, where
    !           LMAX = MAXORD + 1.  YH(i,j+1) contains the approximate
    !           j-th derivative of y(i), scaled by H**j/factorial(j)
    !           (j = 0,1,...,NQ).  On entry for the first step, the first
    !           two columns of YH must be set from the initial values.
    !  LDYH   = A constant integer .ge. N, the first dimension of YH.
    !           N is the number of ODEs in the system.
    !  EWT    = An array of length N containing multiplicative weights
    !           for local error measurements.  Local errors in y(i) are
    !           compared to 1.0/EWT(i) in various error tests.
    !  SAVF   = An array of working storage, of length N.
    !  VSAV   = A work array of length N passed to subroutine VNLS.
    !  ACOR   = A work array of length N, used for the accumulated
    !           corrections.  On a successful return, ACOR(i) contains
    !           the estimated one-step local error in y(i).
    !  F      = Dummy name for the user supplied subroutine for f.
    !  JAC    = Dummy name for the user supplied Jacobian subroutine.
    !  PSOL   = Dummy name for the subroutine passed to VNLS, for
    !           possible use there.
    !  VNLS   = Dummy name for the nonlinear system solving subroutine,
    !           whose real name is dependent on the method used.
    !  RPAR, IPAR = Dummy names for user's real and integer work arrays.
    ! -----------------------------------------------------------------------

    ! -----------------------------------------------------------------------
    !  On a successful return, ETAMAX is reset and ACOR is scaled.
    ! -----------------------------------------------------------------------

#ifdef TRUE_SDC
    use sdc_vode_rhs_module, only: f_rhs, jac
#else
    use vode_rhs_module, only: f_rhs, jac
#endif

    implicit none

    ! Declare arguments
    type(dvode_t), intent(inout) :: vstate
    integer,       intent(inout) :: pivot(VODE_NEQS)

    ! Declare local variables
    real(rt) :: CNQUOT, DDN, DSM, DUP, TOLD
    real(rt) :: ETAQ, ETAQM1, ETAQP1, FLOTL, R
    integer  :: I, IBACK, J, NCF, NFLAG

    ! Parameter declarations
    integer, parameter :: KFC = -3
    integer, parameter :: KFH = -7
    integer, parameter :: MXNCF = 10
    real(rt), parameter :: ADDON = 1.0e-6_rt
    real(rt), parameter :: BIAS1 = 6.0e0_rt
    real(rt), parameter :: BIAS2 = 6.0e0_rt
    real(rt), parameter :: BIAS3 = 10.0e0_rt
    real(rt), parameter :: ETACF = 0.25e0_rt
    real(rt), parameter :: ETAMIN = 0.1e0_rt
    real(rt), parameter :: ETAMXF = 0.2e0_rt
    real(rt), parameter :: ETAMX1 = 1.0e4_rt
    real(rt), parameter :: ETAMX2 = 10.0e0_rt
    real(rt), parameter :: ETAMX3 = 10.0e0_rt
    real(rt), parameter :: ONEPSM = 1.00001e0_rt
    real(rt), parameter :: THRESH = 1.5e0_rt

    logical :: do_initialization
    logical :: already_set_eta
    !$gpu

    ETAQ   = 1.0_rt
    ETAQM1 = 1.0_rt

    vstate % KFLAG = 0
    TOLD = vstate % TN
    NCF = 0
    vstate % JCUR = 0
    NFLAG = 0

    do_initialization = .false.

    if (vstate % JSTART == 0) then

       ! -----------------------------------------------------------------------
       !  On the first call, the order is set to 1, and other variables are
       !  initialized.  ETAMAX is the maximum ratio by which H can be increased
       !  in a single step.  It is normally 10, but is larger during the
       !  first step to compensate for the small initial H.  If a failure
       !  occurs (in corrector convergence or error test), ETAMAX is set to 1
       !  for the next increase.
       ! -----------------------------------------------------------------------
       vstate % NQ = 1
       vstate % L = 2
       vstate % NQNYH = vstate % NQ * VODE_NEQS
       vstate % TAU(1) = vstate % H
       vstate % PRL1 = 1.0_rt
       vstate % RC = 0.0_rt
       vstate % ETAMAX = ETAMX1
       vstate % NQWAIT = 2
       vstate % HSCAL = vstate % H

    else
       ! -----------------------------------------------------------------------
       !  Take preliminary actions on a normal continuation step (JSTART>0).
       !  If the driver changed H, then ETA must be reset and NEWH set to 1.
       !  If a change of order was dictated on the previous step, then
       !  it is done here and appropriate adjustments in the history are made.
       !  On an order decrease, the history array is adjusted by DVJUST.
       !  On an order increase, the history array is augmented by a column.
       !  On a change of step size H, the history array YH is rescaled.
       ! -----------------------------------------------------------------------

       if (vstate % NEWH == 0) then
          do_initialization = .false.
       else

          if (vstate % NEWQ < vstate % NQ) then
             call dvjust(-1, vstate)
             vstate % NQ = vstate % NEWQ
             vstate % L = vstate % NQ + 1
             vstate % NQWAIT = vstate % L
          else if (vstate % NEWQ > vstate % NQ) then
             call dvjust(1, vstate)
             vstate % NQ = vstate % NEWQ
             vstate % L = vstate % NQ + 1
             vstate % NQWAIT = vstate % L
          end if

          do_initialization = .true.
       end if

    end if

    if (do_initialization) then
       ! Rescale the history array for a change in H by a factor of ETA. ------

       R = 1.0_rt

       do J = 2, vstate % L
          R = R * vstate % ETA
          vstate % YH(:,J) = vstate % YH(:,J) * R
       end do
       vstate % H = vstate % HSCAL * vstate % ETA
       vstate % HSCAL = vstate % H
       vstate % RC = vstate % RC * vstate % ETA
       vstate % NQNYH = vstate % NQ*VODE_NEQS
    end if

    ! -----------------------------------------------------------------------
    !  This section computes the predicted values by effectively
    !  multiplying the YH array by the Pascal triangle matrix.
    !  DVSET is called to calculate all integration coefficients.
    !  RC is the ratio of new to old values of the coefficient H/EL(2)=h/l1.
    ! -----------------------------------------------------------------------

    do while (.true.)

       vstate % TN = vstate % TN + vstate % H

       call advance_nordsieck(vstate)

       call dvset(vstate)
       vstate % RL1 = 1.0_rt/vstate % EL(2)
       vstate % RC = vstate % RC * (vstate % RL1/vstate % PRL1)
       vstate % PRL1 = vstate % RL1

       !  Call the nonlinear system solver. ------------------------------------
       call dvnlsd(pivot, NFLAG, vstate)

       if (NFLAG /= 0) then
          ! -----------------------------------------------------------------------
          !  The VNLS routine failed to achieve convergence (NFLAG .NE. 0).
          !  The YH array is retracted to its values before prediction.
          !  The step size H is reduced and the step is retried, if possible.
          !  Otherwise, an error exit is taken.
          ! -----------------------------------------------------------------------
          NCF = NCF + 1
          vstate % ETAMAX = 1.0_rt
          vstate % TN = TOLD

          call retract_nordsieck(vstate)

          if (NFLAG < -1) then
             if (NFLAG == -2) vstate % KFLAG = -3
             if (NFLAG == -3) vstate % KFLAG = -4
             vstate % JSTART = 1
             return
          end if
          if (abs(vstate % H) <= HMIN * ONEPSM) then
             vstate % KFLAG = -2
             vstate % JSTART = 1
             return
          end if
          if (NCF == MXNCF) then
             vstate % KFLAG = -2
             vstate % JSTART = 1
             return
          end if
          vstate % ETA = ETACF
          vstate % ETA = max(vstate % ETA, HMIN / abs(vstate % H))
          NFLAG = -1

          ! Rescale the history array for a change in H by a factor of ETA. ------
          R = 1.0_rt

          do J = 2, vstate % L
             R = R * vstate % ETA
             vstate % YH(:,J) = vstate % YH(:,J) * R
          end do
          vstate % H = vstate % HSCAL * vstate % ETA
          vstate % HSCAL = vstate % H
          vstate % RC = vstate % RC * vstate % ETA
          vstate % NQNYH = vstate % NQ*VODE_NEQS

          cycle
       end if

       ! -----------------------------------------------------------------------
       !  The corrector has converged (NFLAG = 0).  The local error test is
       !  made and control passes to statement 500 if it fails.
       ! -----------------------------------------------------------------------

       DSM = vstate % ACNRM/vstate % TQ(2)
       if (DSM <= 1.0_rt) then
          ! -----------------------------------------------------------------------
          !  After a successful step, update the YH and TAU arrays and decrement
          !  NQWAIT.  If NQWAIT is then 1 and NQ .lt. MAXORD, then ACOR is saved
          !  for use in a possible order increase on the next step.
          !  If ETAMAX = 1 (a failure occurred this step), keep NQWAIT .ge. 2.
          ! -----------------------------------------------------------------------
          vstate % KFLAG = 0
          vstate % NST = vstate % NST + 1
          vstate % HU = vstate % H
          do IBACK = 1, vstate % NQ
             I = vstate % L - IBACK
             vstate % TAU(I+1) = vstate % TAU(I)
          end do
          vstate % TAU(1) = vstate % H
          do J = 1, vstate % L
             vstate % yh(:,J) = vstate % yh(:,J) + vstate % EL(J) * vstate % acor(:)
          end do
          vstate % NQWAIT = vstate % NQWAIT - 1
          if ((vstate % L /= VODE_LMAX) .and. (vstate % NQWAIT == 1)) then
             vstate % yh(1:VODE_NEQS,VODE_LMAX) = vstate % acor(1:VODE_NEQS)

             vstate % CONP = vstate % TQ(5)
          end if
          if (vstate % ETAMAX .NE. 1.0_rt) exit
          if (vstate % NQWAIT < 2) vstate % NQWAIT = 2
          vstate % NEWQ = vstate % NQ
          vstate % NEWH = 0
          vstate % ETA = 1.0_rt
          vstate % HNEW = vstate % H
          vstate % ETAMAX = ETAMX3
          if (vstate % NST <= 10) vstate % ETAMAX = ETAMX2
          R = 1.0_rt/vstate % TQ(2)
          vstate % acor(:) = vstate % acor(:) * R
          vstate % JSTART = 1
          return

       endif

       ! -----------------------------------------------------------------------
       !  The error test failed.  KFLAG keeps track of multiple failures.
       !  Restore TN and the YH array to their previous values, and prepare
       !  to try the step again.  Compute the optimum step size for the
       !  same order.  After repeated failures, H is forced to decrease
       !  more rapidly.
       ! -----------------------------------------------------------------------

       vstate % KFLAG = vstate % KFLAG - 1
       NFLAG = -2
       vstate % TN = TOLD

       call retract_nordsieck(vstate)

       if (abs(vstate % H) <= HMIN * ONEPSM) then
          vstate % KFLAG = -1
          vstate % JSTART = 1
          return
       end if
       vstate % ETAMAX = 1.0_rt
       if (vstate % KFLAG > KFC) then
          ! Compute ratio of new H to current H at the current order. ------------
          FLOTL = REAL(vstate % L)
          vstate % ETA = 1.0_rt/((BIAS2*DSM)**(1.0_rt/FLOTL) + ADDON)
          vstate % ETA = max(vstate % ETA, HMIN / abs(vstate % H), ETAMIN)
          if ((vstate % KFLAG <= -2) .and. (vstate % ETA > ETAMXF)) vstate % ETA = ETAMXF

          ! Rescale the history array for a change in H by a factor of ETA. ------
          R = 1.0_rt

          do J = 2, vstate % L
             R = R * vstate % ETA
             vstate % YH(:,J) = vstate % YH(:,J) * R
          end do
          vstate % H = vstate % HSCAL * vstate % ETA
          vstate % HSCAL = vstate % H
          vstate % RC = vstate % RC * vstate % ETA
          vstate % NQNYH = vstate % NQ*VODE_NEQS

          cycle
       end if

       ! -----------------------------------------------------------------------
       !  Control reaches this section if 3 or more consecutive failures
       !  have occurred.  It is assumed that the elements of the YH array
       !  have accumulated errors of the wrong order.  The order is reduced
       !  by one, if possible.  Then H is reduced by a factor of 0.1 and
       !  the step is retried.  After a total of 7 consecutive failures,
       !  an exit is taken with KFLAG = -1.
       ! -----------------------------------------------------------------------
       if (vstate % KFLAG == KFH) then
          vstate % KFLAG = -1
          vstate % JSTART = 1
          return
       end if
       if (vstate % NQ /= 1) then
          vstate % ETA = max(ETAMIN, HMIN / abs(vstate % H))
          call DVJUST (-1, vstate)
          vstate % L = vstate % NQ
          vstate % NQ = vstate % NQ - 1
          vstate % NQWAIT = vstate % L

          ! Rescale the history array for a change in H by a factor of ETA. ------
          R = 1.0_rt

          do J = 2, vstate % L
             R = R * vstate % ETA
             vstate % YH(:,J) = vstate % YH(:,J) * R
          end do
          vstate % H = vstate % HSCAL * vstate % ETA
          vstate % HSCAL = vstate % H
          vstate % RC = vstate % RC * vstate % ETA
          vstate % NQNYH = vstate % NQ*VODE_NEQS

          cycle

       end if

       vstate % ETA = max(ETAMIN, HMIN / abs(vstate % H))
       vstate % H = vstate % H * vstate % ETA
       vstate % HSCAL = vstate % H
       vstate % TAU(1) = vstate % H
       call f_rhs (vstate % TN, vstate, vstate % savf)
       vstate % NFE = vstate % NFE + 1
       do I = 1, VODE_NEQS
          vstate % yh(I,2) = vstate % H * vstate % savf(I)
       end do
       vstate % NQWAIT = 10

    end do


    ! -----------------------------------------------------------------------
    !  If NQWAIT = 0, an increase or decrease in order by one is considered.
    !  Factors ETAQ, ETAQM1, ETAQP1 are computed by which H could
    !  be multiplied at order q, q-1, or q+1, respectively.
    !  The largest of these is determined, and the new order and
    !  step size set accordingly.
    !  A change of H or NQ is made only if H increases by at least a
    !  factor of THRESH.  If an order change is considered and rejected,
    !  then NQWAIT is set to 2 (reconsider it after 2 steps).
    ! -----------------------------------------------------------------------

    already_set_eta = .false.

    !  Compute ratio of new H to current H at the current order. ------------
    FLOTL = REAL(vstate % L)
    ETAQ = 1.0_rt/((BIAS2*DSM)**(1.0_rt/FLOTL) + ADDON)
    if (vstate % NQWAIT == 0) then
       vstate % NQWAIT = 2
       ETAQM1 = 0.0_rt
       if (vstate % NQ /= 1) then
          ! Compute ratio of new H to current H at the current order less one. ---
          DDN = sqrt(sum((vstate % yh(:,vstate % L) * vstate % ewt)**2) / VODE_NEQS) / vstate % TQ(1)
          ETAQM1 = 1.0_rt/((BIAS1*DDN)**(1.0_rt/(FLOTL - 1.0_rt)) + ADDON)
       end if
       ETAQP1 = 0.0_rt
       if (vstate % L /= VODE_LMAX) then
          ! Compute ratio of new H to current H at current order plus one. -------
          CNQUOT = (vstate % TQ(5)/vstate % CONP)*(vstate % H/vstate % TAU(2))**vstate % L
          do I = 1, VODE_NEQS
             vstate % savf(I) = vstate % acor(I) - CNQUOT * vstate % yh(I,VODE_LMAX)
          end do
          DUP = sqrt(sum((vstate % savf * vstate % ewt)**2) / VODE_NEQS) / vstate % TQ(3)
          ETAQP1 = 1.0_rt/((BIAS3*DUP)**(1.0_rt/(FLOTL + 1.0_rt)) + ADDON)
       end if
       if (ETAQ < ETAQP1) then
          if (ETAQP1 > ETAQM1) then
             vstate % ETA = ETAQP1
             vstate % NEWQ = vstate % NQ + 1
             vstate % yh(1:VODE_NEQS,VODE_LMAX) = vstate % acor(1:VODE_NEQS)
          else
             vstate % ETA = ETAQM1
             vstate % NEWQ = vstate % NQ - 1
          end if
          already_set_eta = .true.
       end if

       if (ETAQ < ETAQM1 .and. .not. already_set_eta) then
          vstate % ETA = ETAQM1
          vstate % NEWQ = vstate % NQ - 1
          already_set_eta = .true.
       end if
    end if

    if (.not. already_set_eta) then
       vstate % ETA = ETAQ
       vstate % NEWQ = vstate % NQ
    end if

    ! Test tentative new H against THRESH and ETAMAX, and HMXI, then exit. ----
    if (vstate % ETA >= THRESH .and. vstate % ETAMAX /= 1.0_rt) then
       vstate % ETA = min(vstate % ETA,vstate % ETAMAX)
       vstate % ETA = vstate % ETA / max(1.0_rt, abs(vstate % H) * HMXI * vstate % ETA)
       vstate % NEWH = 1
       vstate % HNEW = vstate % H * vstate % ETA
       vstate % ETAMAX = ETAMX3
       if (vstate % NST <= 10) vstate % ETAMAX = ETAMX2
       R = 1.0_rt/vstate % TQ(2)
       vstate % acor(:) = vstate % acor(:) * R
       vstate % JSTART = 1
       return
    end if

    vstate % NEWQ = vstate % NQ
    vstate % NEWH = 0
    vstate % ETA = 1.0_rt
    vstate % HNEW = vstate % H
    vstate % ETAMAX = ETAMX3
    if (vstate % NST <= 10) vstate % ETAMAX = ETAMX2
    R = 1.0_rt/vstate % TQ(2)
    vstate % acor(:) = vstate % acor(:) * R
    vstate % JSTART = 1

  end subroutine dvstep

end module cuvode_dvstep_module
