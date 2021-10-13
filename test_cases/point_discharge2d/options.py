from desal_adapt import *
from pyroteus.math import bessk0
from thetis.configuration import PositiveFloat
import numpy as np


__all__ = ["PointDischarge2dOptions"]


class PointDischarge2dOptions(PlantOptions):
    """
    Problem specification for a simple
    advection-diffusion test case with a
    point source, from [Riadh et al. 2014].

    [Riadh et al. 2014] A. Riadh, G.
        Cedric, M. Jean, "TELEMAC modeling
        system: 2D hydrodynamics TELEMAC-2D
        software release 7.0 user manual."
        Paris: R&D, Electricite de France,
        p. 134 (2014).
    """
    domain_length = PositiveFloat(50.0).tag(config=False)
    domain_width = PositiveFloat(10.0).tag(config=False)

    def __init__(self, configuration='aligned', level=0, family='cg', mesh=None, shift=1.0):
        """
        :kwarg configuration: choose from 'aligned and 'offset'
        :kwarg level: mesh resolution level
        :kwarg family: choose from 'cg' and 'dg'
        :kwarg mesh: user-provided mesh
        :kwarg shift: number of units to shift the point source to the right
        """
        super(PointDischarge2dOptions, self).__init__()
        assert configuration in ('aligned', 'offset')
        assert level >= 0
        assert family in ('cg', 'dg')
        assert shift >= 0.0

        # Setup mesh
        self.mesh2d = mesh
        if self.mesh2d is None:
            self.mesh2d = RectangleMesh(100*2**level, 20*2**level, self.domain_length, self.domain_width)
        self.setup_mesh(self.mesh2d)

        # Physics
        self.tracer_only = True
        self.horizontal_velocity_scale = Constant(1.0)
        D = Constant(0.1)
        self.horizontal_diffusivity_scale = D
        self.bathymetry2d = Constant(1.0)  # arbitrary
        self.bnd_conditions = {
            1: {'value': Constant(0.0)},  # inflow
        }

        # Point source parametrisation
        self.source_value = 100.0
        self.source_x = 1.0 + shift
        self.source_y = 5.0
        self.source_r = 0.05606388
        self.add_tracer_2d('tracer_2d',
                           'Depth averaged tracer',
                           'Tracer2d',
                           shortname='Tracer',
                           diffusivity=D,
                           source=self.source)

        # Receiver parametrisation
        self.receiver_x = 20.0
        self.receiver_y = 5.0 if configuration == 'aligned' else 7.5
        self.receiver_r = 0.5

        # Spatial discretisation
        self.tracer_element_family = family
        self.use_lax_friedrichs_tracer = family == 'dg'
        self.use_supg_tracer = family == 'cg'
        self.use_limiter_for_tracers = family == 'dg'

        # Temporal discretisation
        self.tracer_timestepper_type = 'SteadyState'
        self.timestep = 20.0
        self.simulation_export_time = 18.0
        self.simulation_end_time = 18.0

        # Solver parameters
        self.tracer_timestepper_options.solver_parameters.update({
            'ksp_converged_reason': None,
            'ksp_max_it': 10000,
            'ksp_type': 'gmres',
            'pc_type': 'bjacobi',
        })

        # I/O
        self.fields_to_export = ['tracer_2d']
        self.fields_to_export_hdf5 = []

        self._isfrozen = True

    def apply_boundary_conditions(self, solver_obj):
        if len(solver_obj.function_spaces.keys()) == 0:
            solver_obj.create_function_spaces()
        solver_obj.bnd_functions['tracer'] = self.bnd_conditions

    def apply_initial_conditions(self, solver_obj):
        if len(solver_obj.function_spaces.keys()) == 0:
            solver_obj.create_function_spaces()
        uv = Function(solver_obj.function_spaces.U_2d)
        uv.interpolate(as_vector([self.horizontal_velocity_scale.values()[0], 0.0]))
        solver_obj.assign_initial_conditions(uv=uv)

    @property
    def source(self):
        x, y = SpatialCoordinate(self.mesh2d)
        x0 = self.source_x
        y0 = self.source_y
        r = self.source_r
        return self.source_value*exp(-((x - x0)**2 + (y - y0)**2)/r**2)

    @property
    def qoi_kernel(self):
        x, y = SpatialCoordinate(self.mesh2d)
        x0 = self.receiver_x
        y0 = self.receiver_y
        r = self.receiver_r
        ball = conditional((x - x0)**2 + (y - y0)**2 < r**2, 1, 0)
        area = assemble(ball*dx)
        scaling = 1.0 if np.isclose(area, 0.0) else pi*r**2/area
        return scaling*ball

    def qoi(self, solution, quadrature_degree=12):
        dx_qoi = dx(degree=quadrature_degree)
        return assemble(self.qoi_kernel*solution*dx_qoi)

    def analytical_solution(self):
        fs = get_functionspace(self.mesh2d, self.tracer_element_family.upper(), 1)
        solution = Function(fs, name='Analytical solution')
        x, y = SpatialCoordinate(self.mesh2d)
        x0 = self.source_x
        y0 = self.source_y
        r = self.source_r
        u = self.horizontal_velocity_scale
        D = self.horizontal_diffusivity_scale
        Pe = 0.5*u/D
        q = 1.0
        rr = max_value(sqrt((x - x0)**2 + (y - y0)**2), r)
        solution.interpolate(0.5*q/(pi*D)*exp(Pe*(x - x0))*bessk0(Pe*rr))
        return solution

    def analytical_qoi(self, quadrature_degree=12):
        x, y = SpatialCoordinate(self.mesh2d)
        x0 = self.source_x
        y0 = self.source_y
        r = self.source_r
        u = self.horizontal_velocity_scale
        D = self.horizontal_diffusivity_scale
        Pe = 0.5*u/D
        q = 1.0
        rr = max_value(sqrt((x - x0)**2 + (y - y0)**2), r)
        solution = 0.5*q/(pi*D)*exp(Pe*(x - x0))*bessk0(Pe*rr)
        dx_qoi = dx(degree=quadrature_degree)
        return assemble(self.qoi_kernel*solution*dx_qoi)
