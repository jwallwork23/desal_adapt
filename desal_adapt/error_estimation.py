from thetis import *
from pyroteus import *


__all__ = ["ErrorEstimator"]


class ErrorEstimator(object):
    """
    Error estimation for advection-diffusion
    desalination outfall modelling applications.
    """
    @PETSc.Log.EventDecorator('ErrorEstimator.__init__')
    def __init__(self, options,
                 mesh=None,
                 norm_type='L2',
                 error_estimator='difference_quotient',
                 metric='isotropic_dwr',
                 boundary=True,
                 recovery_method='L2',
                 mixed_L2=False,
                 weighted_gradient_source=True):
        """
        :args options: :class:`PlantOptions` parameter object.
        :kwarg mesh: the mesh
        :kwarg norm_type: norm type for error estimator
        :kwarg error_estimator: error estimator type
        :kwarg metric: metric type
        :kwarg boundary: should boundary contributions be considered?
        :kwarg recovery_method: choose from 'L2' and 'Clement'
        :kwarg mixed_L2: should L2 projection solve a mixed system?
        :kwarg weighted_gradient_source: include source terms in
            the weighted gradient metric formulation?
        """
        self.options = options
        if mesh is not None:
            self.mesh = mesh
        elif hasattr(options, 'mesh2d'):
            self.mesh = options.mesh2d
        elif hasattr(options, 'mesh3d'):
            self.mesh = options.mesh3d
        else:
            raise ValueError('Mesh not provided!')
        self.dim = self.mesh.topological_dimension()
        self.name = f'tracer_{self.dim}d'
        self.P0 = FunctionSpace(self.mesh, "DG", 0)
        self.P0_vec = VectorFunctionSpace(self.mesh, "DG", 0)
        self.p0test = TestFunction(self.P0)
        self.p0trial = TrialFunction(self.P0)
        self.P1_vec = VectorFunctionSpace(self.mesh, "CG", 1)
        self.P1_ten = TensorFunctionSpace(self.mesh, "CG", 1)
        self.h = options.cell_size
        self.n = FacetNormal(self.mesh)

        # Discretisation parameters
        assert self.options.tracer_timestepper_type in ('CrankNicolson', 'SteadyState')
        self.steady = self.options.tracer_timestepper_type == 'SteadyState'
        self.theta = None if self.steady else self.options.timestepper_options.implicitness_theta
        self.is_dg = self.options.tracer_element_family == 'dg'
        if self.is_dg:
            raise NotImplementedError  # TODO
        else:
            if self.options.use_supg_tracer:
                assert options.test_function is not None

        # Error estimation parameters
        assert norm_type in ('L1', 'L2')
        self.norm_type = norm_type
        if error_estimator != 'difference_quotient':
            raise NotImplementedError  # TODO
        self.error_estimator = error_estimator
        if metric not in ['hessian', 'isotropic_dwr', 'anisotropic_dwr',
                          'weighted_hessian', 'weighted_gradient']:
            raise ValueError(f'Metric type {metric} not recognised')
        self.metric_type = metric
        self.boundary = boundary
        self.recovery_method = recovery_method
        self.mixed_L2 = mixed_L2
        self.weighted_gradient_source = weighted_gradient_source

    def _replace_supg(self, f):
        if self.options.use_supg_tracer:
            return replace(self.options.test_function, {TestFunction(self.options.Q): f})
        else:
            return f

    def _Psi_steady(self, uv, c):
        """
        Strong residual for steady state mode.

        :arg uv: velocity field
        :arg c: tracer concentration
        """
        D = self.options.tracer[self.name].diffusivity
        adv = inner(uv, grad(c))
        diff = div(D*grad(c))
        S = self.options.tracer[self.name].source
        return adv - diff - S

    def _potential_steady(self, uv, c):
        """
        Potential providing the flux form of the advection-diffusion equation.

        :arg uv: velocity field
        :arg c: tracer concentration
        """
        D = self.options.tracer[self.name].diffusivity
        return uv*c - D*grad(c)

    def _bnd_potential_steady(self, uv, c):
        """
        Potential providing the flux form for the boundary conditions
        of the advection-diffusion equation.

        :arg uv: velocity field
        :arg c: tracer concentration
        """
        D = self.options.tracer[self.name].diffusivity
        bnd_terms = {'interior': -D*c}
        if self.boundary:
            bnd_conditions = self.options.bnd_conditions
            for bnd_marker in bnd_conditions:
                funcs = bnd_conditions.get(bnd_marker)
                tag = int(bnd_marker)
                bnd_terms[tag] = dot(D*grad(c), self.n)
                c_in = c
                if 'value' in funcs:
                    uv_ext, c_ext = self._get_bnd_functions(uv, c, bnd_marker)
                    uv_av = 0.5*(uv + uv_ext)
                    s = 0.5*(sign(dot(uv_av, self.n)) + 1.0)
                    c_up = c_in*s + c_ext*(1-s)
                    diff_flux_up = D*grad(c_up)
                    bnd_terms[tag] += -dot(diff_flux_up, self.n)
                elif 'diff_flux' in funcs:
                    bnd_terms[tag] += -funcs['diff_flux']
        return bnd_terms

    def _source_steady(self):
        """
        Source term for the steady advection-diffusion equation.
        """
        # TODO: this won't work for time-dependent problems
        return self.options.tracer[self.name].source

    def _restrict(self, v):
        """
        Restrict an expression `v` over facets, according to the :attr:`norm_type`.

        :arg v: the UFL expression
        """
        return jump(abs(v), self.p0test) if self.norm_type == 'L1' else jump(v*v, self.p0test)

    def _get_bnd_functions(self, uv_in, c_in, bnd_in):
        """
        Get boundary contributions on segment `bnd_in`.

        :arg uv_in: the current velocity
        :arg c_in: the current tracer concentration
        :arg bnd_in: the boundary marker
        """
        funcs = self.options.bnd_conditions.get(bnd_in)
        uv_ext = funcs['uv'] if 'uv' in funcs else uv_in
        c_ext = funcs['value'] if 'value' in funcs else c_in
        return uv_ext, c_ext

    def _psi_steady(self, *args):
        """
        Inter-element flux contributions for steady state mode.
        """
        if len(args) == 2:
            uv, c = args
            uv_old, c_old = uv, c
        elif len(args) == 4:
            uv, c, uv_old, c_old = args
        else:
            raise Exception(f'Expected two or four arguments, got {len(args)}.')
        # TODO: tracer_advective_velocity_factor?
        D = self.options.tracer[self.name].diffusivity
        psi = Function(self.P0)
        ibp_terms = self._restrict(inner(D*grad(c), self.n))*dS
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
                    uv_ext_old, c_ext_old = self._get_bnd_functions(uv_old, c_old, bnd_marker)
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
            flux_terms = 2*avg(self.p0test)*abs(flux_terms)*dS(domain=self.mesh)
            bnd_terms = sum(self.p0test*abs(term)*ds_bnd for ds_bnd, term in bnd_terms.items())
        else:
            flux_terms = 2*avg(self.p0test)*flux_terms*flux_terms*dS(domain=self.mesh)
            bnd_terms = sum(self.p0test*term*term*ds_bnd for ds_bnd, term in bnd_terms.items())
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

    def _Psi_unsteady(self, uv, c, uv_old, c_old):
        """
        Strong residual for unsteady mode.

        :arg uv: velocity field at current timestep
        :arg c: tracer concentration at current timestep
        :arg uv_old: velocity field at previous timestep
        :arg c_old: tracer concentration at previous timestep
        """
        f_time = (c - c_old)/self.options.timestep
        f = self._Psi_steady(uv, c)
        f_old = self._Psi_steady(uv_old, c_old)
        return f_time + self.theta*f + (1-self.theta)*f_old

    def _psi_unsteady(self, uv, c, uv_old, c_old):
        """
        Inter-element flux terms for unsteady mode.

        :arg uv: velocity field at current timestep
        :arg c: tracer concentration at current timestep
        :arg uv_old: velocity field at previous timestep
        :arg c_old: tracer concentration at previous timestep
        """
        f = self._psi_steady(uv, c, uv, c)    # NOTE: Not semi-implicit
        f_old = self._psi_steady(uv_old, c_old, uv_old, c_old)
        return self.theta*f + (1-self.theta)*f_old

    def _potential_unsteady(self, uv, c, uv_old, c_old):
        """
        Potential providing the flux form of the transient advection-diffusion equation.

        :arg uv: velocity field at current timestep
        :arg c: tracer concentration at current timestep
        :arg uv_old: velocity field at previous timestep
        :arg c_old: tracer concentration at previous timestep
        """
        f_time = (c - c_old)/self.options.timestep
        f = self._potential_steady(uv, c)
        f_old = self._potential_steady(uv_old, c_old)
        return f_time + self.theta*f + (1-self.theta)*f_old

    def _bnd_potential_unsteady(self, uv, c, uv_old, c_old):
        """
        Potential providing the boundary part of the
        flux form of the transient advection-diffusion equation.

        :arg uv: velocity field at current timestep
        :arg c: tracer concentration at current timestep
        :arg uv_old: velocity field at previous timestep
        :arg c_old: tracer concentration at previous timestep
        """
        f = self._bnd_potential_steady(uv, c)
        f_old = self._bnd_potential_steady(uv_old, c_old)
        return {tag: self.theta*f[tag] + (1-self.theta)*f_old[tag] for tag in f}

    def _source_unsteady(self):
        """
        Source term for the unsteady advection-diffusion equation.
        """
        raise NotImplementedError  # TODO

    @PETSc.Log.EventDecorator('ErrorEstimator.strong_residual')
    def strong_residual(self, *args):
        """
        Compute the strong residual over a single
        timestep, given current solution and
        lagged solution.
        """
        Psi = self._Psi_steady(*args) if self.steady else self._Psi_unsteady(*args)
        return assemble(self.p0test*abs(Psi)*dx) if self.norm_type == 'L1' else sqrt(abs(assemble(self.p0test*Psi*Psi*dx)))

    @PETSc.Log.EventDecorator('ErrorEstimator.flux_terms')
    def flux_terms(self, *args):
        """
        Compute flux jump terms over a single
        timestep, given current solution and
        lagged solution.
        """
        return self._psi_steady(*args) if self.steady else self._psi_unsteady(*args)

    @PETSc.Log.EventDecorator('ErrorEstimator.potential')
    def potential(self, *args):
        """
        Compute flux form potential over a single
        timestep, given current solution and
        lagged solution.
        """
        return self._potential_steady(*args) if self.steady else self._potential_unsteady(*args)

    @PETSc.Log.EventDecorator('ErrorEstimator.bnd_potential')
    def bnd_potential(self, *args):
        """
        Compute boundary part of the flux form
        potential over a single timestep, given
        current solution and lagged solution.
        """
        return self._bnd_potential_steady(*args) if self.steady else self._bnd_potential_unsteady(*args)

    @PETSc.Log.EventDecorator('ErrorEstimator.source')
    def source(self):
        """
        Source term for the advection-diffusion equation.
        """
        return self._source_steady() if self.steady else self._source_unsteady()

    @PETSc.Log.EventDecorator('ErrorEstimator.recover_gradient')
    def recover_gradient(self, f):
        """
        Recover the Laplacian of some scalar field `f`.

        :arg f: the scalar field to be projected
        """
        if self.recovery_method == 'Clement':
            from pyroteus.interpolation import clement_interpolant
            return clement_interpolant(interpolate(grad(f), self.P0_vec))
        elif self.recovery_method == 'L2':
            return firedrake.project(grad(f), self.P1_vec)
        else:
            raise ValueError(f'Gradient recovery method "{method}" not recognised.')

    @PETSc.Log.EventDecorator('ErrorEstimator.recover_laplacian')
    def recover_laplacian(self, c):
        """
        Recover the Laplacian of `c`.
        """
        proj = self.recover_gradient(self._replace_supg(c))
        if self.norm_type == 'L1':
            return interpolate(abs(div(proj)), self.P0)
        else:
            return sqrt(interpolate(inner(div(proj), div(proj)), self.P0))

    @PETSc.Log.EventDecorator('ErrorEstimator.recover_hessian')
    def recover_hessian(self, c):
        """
        Recover the Hessian of `c`.
        """
        if self.metric_type == 'weighted_hessian':
            c = self._replace_supg(c)
        return hessian_metric(recover_hessian(c, mesh=self.mesh, method=self.recovery_method, mixed=self.mixed_L2))

    @PETSc.Log.EventDecorator('ErrorEstimator.difference_quotient')
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
            R = self.recover_laplacian(args[nargs//2+1])
            if not self.steady:  # Average recovered Laplacians
                R += self.recover_laplacian(args[-1])
                R *= 0.5

        # Combine the two
        dq = Function(self.P0, name='Difference quotient')
        dq.project((Psi + psi/sqrt(self.h))*R)
        dq.interpolate(abs(dq))  # Ensure positivity
        return dq

    @PETSc.Log.EventDecorator('ErrorEstimator.error_indicator')
    def error_indicator(self, *args, **kwargs):
        """
        Evaluate the error indicator of choice.
        """
        if self.error_estimator == 'difference_quotient':
            flux_form = kwargs.get('flux_form', False)
            return self.difference_quotient(*args, flux_form=flux_form)
        else:
            raise NotImplementedError  # TODO

    @PETSc.Log.EventDecorator('ErrorEstimator.metric')
    def metric(self, *args, **kwargs):
        """
        Construct the metric of choice.
        """
        nargs = len(args)
        assert nargs == 4 if self.steady else 8
        if self.metric_type == 'hessian':
            if not self.steady:
                raise NotImplementedError  # TODO
            return self.recover_hessian(args[1])
        elif self.metric_type == 'isotropic_dwr':
            return isotropic_metric(self.error_indicator(*args, **kwargs),
                                    target_space=self.P1_ten,
                                    interpolant=self.recovery_method)
        elif self.metric_type in ('weighted_hessian', 'anisotropic_dwr'):
            flux_form = kwargs.get('flux_form', False)
            kwargs['approach'] = self.metric_type

            # Terms for standard a posteriori error estimate
            Psi = self.strong_residual(*args[:nargs//2])
            psi = self.flux_terms(*args[:nargs//2])
            ee = Psi + psi/sqrt(self.h)

            # Adjoint weighting for anisotropic DWR
            if self.metric_type == 'anisotropic_dwr':
                if flux_form:
                    R = self.flux_terms(*args[nargs//2:])
                else:
                    R = self.recover_laplacian(args[nargs//2+1])
                    if not self.steady:  # Average recovered Laplacians
                        R += self.recover_laplacian(args[-1])
                        R *= 0.5
                ee *= R
            elif flux_form:
                raise ValueError('Flux form is incompatible with weighted Hessian.')
            ee = firedrake.project(ee, self.P0)

            # Hessian weighting term
            istart = nargs//2 if self.metric_type == 'weighted_hessian' else 0
            H = hessian_metric(self.recover_hessian(args[istart+1]))
            if not self.steady:  # Average recovered Hessians
                iend = istart + nargs//2
                H += hessian_metric(self.recover_hessian(args[iend-1]))
                H *= 0.5

            # Combine the two
            return anisotropic_metric([ee], [H], target_space=self.P1_ten,
                                      interpolant=self.recovery_method, **kwargs)
        elif self.metric_type == 'weighted_gradient':
            F = self.potential(*args[:nargs//2])
            adj = args[nargs//2+1]  # NOTE: Only picks current adjoint solution
            g = self.recover_gradient(adj)
            if not self.steady:
                raise NotImplementedError  # TODO: how do we handle both adjoint solutions? average?

            # Interior metric
            H0 = hessian_metric(self.recover_hessian(F[0]))
            H0.interpolate(H0*abs(g[0]))
            H1 = hessian_metric(self.recover_hessian(F[1]))
            H1.interpolate(H1*abs(g[1]))
            interior_metrics = [H0, H1]
            if self.weighted_gradient_source:
                Hs = hessian_metric(self.recover_hessian(self.source()))
                Hs.interpolate(Hs*abs(adj))
                interior_metrics.append(Hs)
            Hint = combine_metrics(*interior_metrics, average=kwargs.get('average', False))

            # Get tags related to Neumann and natural conditions
            tags = self.mesh.exterior_facets.unique_markers
            bcs = {tag: self.options.bnd_conditions.get(tag) or {} for tag in tags}
            tags = [tag for tag in tags if 'value' not in bcs[tag]]

            # Boundary metric
            Fbar = self.bnd_potential(*args[:nargs//2])
            Hbar = recover_boundary_hessian(Fbar, mesh=self.mesh, target_space=self.P1_ten)
            Hbar = interpolate(hessian_metric(Hbar)*abs(adj), self.P1_ten)

            # Adjust target complexity
            target = kwargs.get('target_complexity')
            p = kwargs.get('norm_order')
            C = determine_metric_complexity(Hint, Hbar, target, p)

            # Enforce max/min sizes and anisotropy
            enforce_element_constraints(Hint, 1.0e-30, 1.0e+30, 1.0e+12, optimise=True)
            enforce_element_constraints(Hbar, 1.0e-30, 1.0e+30, 1.0e+12, optimise=True)

            # Normalise the two metrics
            space_normalise(Hint, target, p, global_factor=C, boundary=False)
            space_normalise(Hbar, target, p, global_factor=C, boundary=True)

            # Combine the two
            return metric_intersection(Hint, Hbar, boundary_tag=tags)
        else:
            raise ValueError(f'Metric type {metric} not recognised')
