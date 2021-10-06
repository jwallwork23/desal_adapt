from desal_adapt import *
from thetis.configuration import PositiveFloat


__all__ = ["PointDischarge2dOptions"]


class PointDischarge2dOptions(PlantOptions):
    # TODO: docstring
    resource_dir = create_directory(os.path.join(os.path.dirname(__file__), 'resources'))

    domain_length = PositiveFloat(50.0).tag(config=False)
    domain_width = PositiveFloat(10.0).tag(config=False)

    def __init__(self, configuration='aligned', level=0, family='cg', mesh=None, shift=1.0):
        super(PointDischarge2dOptions, self).__init__()
        self.mesh2d = mesh
        if self.mesh2d is None:
            self.mesh2d = RectangleMesh(100*2**level, 20*2**level, self.domain_length, self.domain_width)
        P0_2d = get_functionspace(self.mesh2d, "DG", 0)
        P1_2d = get_functionspace(self.mesh2d, "CG", 1)
        self.mesh2d.delta_x = interpolate(CellSize(self.mesh2d), P0_2d)
        boundary_markers = sorted(self.mesh2d.exterior_facets.unique_markers)
        one = Function(P1_2d).assign(1.0)
        self.mesh2d.boundary_len = OrderedDict({i: assemble(one*ds(int(i))) for i in boundary_markers})

        # Physics
        self.tracer_only = True
        self.horizontal_velocity_scale = Constant(1.0)
        D = Constant(0.1)
        self.horizontal_diffusivity_scale = D
        self.bathymetry2d = Constant(1.0)  # arbitrary

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

        # I/O
        self.fields_to_export = ['tracer_2d']
        self.fields_to_export_hdf5 = []

    def get_bnd_conditions(self, Q_2d):
        self.bnd_conditions = {
            1: {'value': Constant(0.0)},  # inflow
        }

    def apply_boundary_conditions(self, solver_obj):
        if len(solver_obj.function_spaces.keys()) == 0:
            solver_obj.create_function_spaces()
        self.get_bnd_conditions(solver_obj.function_spaces.Q_2d)
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

    def analytical_solution(self, fs):
        raise NotImplementedError  # TODO

    def analytical_qoi(self, fs):
        raise NotImplementedError  # TODO
