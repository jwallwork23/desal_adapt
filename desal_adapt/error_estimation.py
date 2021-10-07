from thetis import *
from pyroteus import *


__all__ = ["ErrorEstimator"]


class ErrorEstimator(object):
    """
    Error estimation for advection-diffusion
    desalination outfall modelling applications.
    """
    def __init__(self, options,
                 mesh=None,
                 norm_type='L2',
                 error_estimator='difference_quotient',
                 metric='isotropic_dwr',
                 boundary=True):
        """
        :args options: :class:`PlantOptions` parameter object.
        :kwarg mesh: the mesh
        :kwarg norm_type: norm type for error estimator
        :kwarg error_estimator: error estimator type
        :kwarg metric: metric type
        :kwarg boundary: should boundary contributions be considered?
        """
        self.options = options
        self.mesh = mesh or options.mesh2d
        self.P0 = FunctionSpace(self.mesh, "DG", 0)
        self.p0test = TestFunction(self.P0)
        self.p0trial = TrialFunction(self.P0)
        self.P1_ten = TensorFunctionSpace(self.mesh, "CG", 1)
        self.h = CellSize(self.mesh)
        self.n = FacetNormal(self.mesh)

        # Discretisation parameters
        assert self.options.tracer_timestepper_type in ('CrankNicolson', 'SteadyState')
        self.steady = self.options.tracer_timestepper_type == 'SteadyState'
        self.theta = None if self.steady else self.options.timestepper_options.implicitness_theta
        self.is_dg = self.options.tracer_element_family == 'dg'
        if self.is_dg:
            raise NotImplementedError  # TODO

        # Error estimation parameters
        assert norm_type in ('L1', 'L2')
        self.norm_type = norm_type
        if error_estimator != 'difference_quotient':
            raise NotImplementedError  # TODO
        self.error_estimator = error_estimator
        if metric not in ['hessian', 'isotropic_dwr', 'weighted_hessian']:
            raise NotImplementedError  # TODO
        self.metric = metric
        self.boundary = boundary

    def _Psi_steady(self, uv, c):
        D = self.options.horizontal_diffusivity
        adv = inner(uv, grad(c))
        diff = div(D*grad(uv))
        S = self.options.tracer.tracer_2d.source
        return adv - diff - S

    def _restrict(self, v):
        return jump(abs(v), self.p0test) if self.norm_type == 'L1' else jump(v*v, self.p0test)

    def _get_bnd_functions(self, uv_in, c_in, bnd_in):
        funcs = self.options.bnd_conditions.get(bnd_in)
        uv_ext = funcs['uv'] if 'uv' in funcs else uv_in
        c_ext = funcs['value'] if 'value' in funcs else c_in
        return uv_ext, c_ext

    def _psi_steady(self, *args):
        if len(args) == 2:
            uv, c = args
            uv_old, c_old = uv, c
        elif len(args) == 4:
            uv, c, uv_old, c_old = args
        else:
            raise Exception(f'Expected two or four arguments, got {len(args)}.')
        # TODO: tracer_advective_velocity_factor?
        D = self.horizontal_diffusivity
        psi = Function(self.P0)
        ibp_terms = inner(D*grad(c), self.n)
        flux_terms = 0
        bnd_terms = {}
        if self.boundary:
            bnd_conditions = self.options.bnd_conditions
            for bnd_marker in bnd_conditions:
                funcs = bnd_conditions.get(bnd_marker)
                ds_bnd = ds(int(bnd_marker))
                c_in = c

                # Terms from integration by parts
                bnd_terms[ds_bnd] = inner(D*grad(c_in), self.n)

                # Terms from boundary conditions
                if 'value' in funcs:
                    uv_ext, c_ext = self._get_bnd_functions(uv, c, bnd_marker)
                    uv_ext_old, c_ext_old = self._get_bnd_functions(uv_old, bnd_marker)
                    uv_av = 0.5*(uv + uv_ext)
                    s = 0.5*(sign(dot(uv_av, self.n)) + 1.0)
                    c_up = c_in*s + c_ext*(1-s)
                    diff_flux_up = D*grad(c_up)
                    bnd_terms[ds_bnd] += -dot(diff_flux_up, self.n)
                elif 'diff_flux' in funcs:
                    bnd_terms[ds_bnd] += -funcs['diff_flux']

        # Compute flux norm
        mass_term = self.p0test*self.p0trial*dx
        if self.norm_type == 'L1':
            flux_terms = 2*avg(self.p0test)*abs(flux_terms)*dS
            ibp_terms = self._restrict(abs(ibp_terms))*dS
            bnd_terms = sum(self.p0test*abs(term)*ds_bnd for ds_bnd, term in bnd_terms)
        else:
            flux_terms = 2*avg(self.p0test)*flux_terms*flux_terms*dS
            ibp_terms = self._restrict(ibp_terms*ibp_terms)*dS
            bnd_terms = sum(self.p0test*term*term*ds_bnd for ds_bnd, term in bnd_terms)
        sp = {
            'mat_type': 'matfree',
            'snes_type': 'ksponly',
            'ksp_type': 'preonly',
            'pc_type': 'python',
            'pc_python_type': 'firedrake.MassInvPC',
        }
        solve(mass_term == flux_terms + ibp_terms + bnd_terms, psi, solver_parameters=sp)
        psi.interpolate(abs(psi))
        return sqrt(psi) if self.norm_type == 'L2' else psi

    def _Psi_unsteady(self, c, c_old):
        f_time = (uv[0] - uv_old[0])/self.options.timestep
        f = self._Psi_steady(c)
        f_old = self._Psi_steady(c_old)
        return f_time + self.theta*f + (1-self.theta)*f_old

    def _psi_unsteady(self, c, c_old):
        f = self._psi_steady(c, c)    # NOTE: Not semi-implicit
        f_old = self._psi_steady(c_old, c_old)
        return self.theta*f + (1-self.theta)*f_old

    def strong_residual(self, *args):
        """
        Compute the strong residual over a single
        timestep, given current solution and
        lagged solution.

        If :attr:`timestepper` is set to
        `'SteadyState'` then only the tracer
        solution and velocity are used.
        """
        Psi = self._Psi_steady(*args) if self.steady else self._Psi_unsteady(*args)
        return assemble(self.p0test*abs(Psi)*dx) if self.norm_type == 'L1' else sqrt(abs(assemble(self.p0test*Psi*Psi*dx)))

    def flux_terms(self, *args):
        """
        Compute flux jump terms over a single
        timestep, given current solution and
        lagged solution.

        If :attr:`timestepper` is set to
        `'SteadyState'` then only the tracer
        solution and velocity are used.
        """
        # TODO: Account for boundary conditions
        return self._psi_steady(*args) if self.steady else self._psi_unsteady(*args)

    def recover_laplacian(self, c):
        """
        Recover the Laplacian of solution
        tuple `(uv, elev)`.
        """
        P1_vec = VectorFunctionSpace(self.mesh, 'CG', 1)
        g, phi = TrialFunction(P1_vec), TestFunction(P1_vec)
        a = inner(phi, g)*dx
        L = c*dot(phi, self.n)*ds - div(phi)*c*dx
        proj = Function(P1_vec)
        sp = {
            'ksp_type': 'gmres',
            'ksp_gmres_restart': 20,
            'ksp_rtol': 1.0e-05,
            'pc_type': 'sor',
        }
        solve(a == L, proj, solver_parameters=sp)
        if self.norm_type == 'L1':
            return interpolate(abs(div(proj)), self.P0)
        else:
            return sqrt(interpolate(inner(div(proj), div(proj)), self.P0))

    def recover_hessian(self, c):
        """
        Recover the Hessian of the solution.
        """
        return hessian_metric(recover_hessian(c, mesh=self.mesh))

    def difference_quotient(self, *args, flux_form=False):
        """
        Evaluate the dual weighted residual
        error estimator in difference quotient
        formulation.
        """
        nargs = len(args)
        assert nargs == 4 if self.steady else 8

        # Terms for standard a posteriori error estimate
        Psi = self.strong_residual(*args[:nargs//2])
        psi = self.flux_terms(*args[:nargs//2])

        # Weighting term for the adjoint
        if flux_form:
            R = self.flux_terms(*args[nargs//2:])
        else:
            R = self.recover_laplacian(*args[nargs//2:2+nargs//2])
            if not self.steady:  # Average recovered Laplacians
                R += self.recover_laplacian(*args[-2:])
                R *= 0.5

        # Combine the two
        dq = Function(self.P0, name='Difference quotient')
        dq.project((Psi + psi/sqrt(self.h))*R)
        dq.interpolate(abs(dq))  # Ensure positivity
        return dq

    def error_indicator(self, *args, **kwargs):
        if self.error_estimator == 'difference_quotient':
            flux_form = kwargs.get('flux_form', False)
            return self.difference_quotient(*args, flux_form=flux_form)
        else:
            raise NotImplementedError  # TODO

    def metric(self, *args, **kwargs):
        if self.metric == 'hessian':
            if not self.steady:
                raise NotImplementedError  # TODO
            return self.recover_hessian(args[1])
        elif self.metric == 'isotropic_dwr':
            return isotropic_metric(self.error_indicator(*args, **kwargs))
        elif self.metric == 'weighted_hessian':
            flux_form = kwargs.get('flux_form', False)
            nargs = len(args)
            assert nargs == 4 if self.steady else 8

            # Terms for standard a posteriori error estimate
            Psi = self.strong_residual(*args[:nargs//2])
            psi = self.flux_terms(*args[:nargs//2])

            # Weighting term for the adjoint
            if flux_form:
                raise ValueError('Flux form is incompatible with WH.')
            else:
                H = self.recover_hessian(*args[nargs//2:2+nargs//2])
                if not self.steady:  # Average recovered Hessians
                    H += self.recover_hessians(*args[-2:])
                    H *= 0.5

            # Combine the two
            M = Function(self.P1_ten, name='Weighted Hessian metric')
            M.project((Psi + psi/sqrt(self.h))*H)
            M.assign(hessian_metric(M))
            return M
        else:
            raise NotImplementedError  # TODO
