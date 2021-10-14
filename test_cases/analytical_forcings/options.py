from desal_adapt import *
from thetis.configuration import PositiveFloat


__all__ = ["AnalyticalForcingOptions"]


class AnalyticalForcingsOptions(PlantOptions):
    """
    Parameters for an idealised desalination plant
    outfall scenario with analytically prescribed
    tidal forcings.
    """
    resource_dir = create_directory(os.path.join(os.path.dirname(__file__), 'resources'))
    domain_length = PositiveFloat(3000.0).tag(config=False)
    domain_width = PositiveFloat(1000.0).tag(config=False)
    background_salinity = PositiveFloat(39.0).tag(config=True)

    def __init__(self, configuration='aligned', level=0, family='cg', mesh=None, **kwargs):
        """
        :kwarg configuration: choose from 'aligned' and 'offset'
        :kwarg level: mesh resolution level
        :kwarg family: choose from 'cg' and 'dg'
        :kwarg mesh: user-provided mesh
        """
        super(AnalyticalForcingsOptions, self).__init__(**kwargs)
        assert configuration in ('aligned', 'offset')
        assert level >= 0
        assert family in ('cg', 'dg')

        # Setup mesh
        if kwargs.get('meshgen', False):
            return
        elif mesh is None:
            mesh_file = os.path.join(self.resource_dir, f'channel_{level}.msh')
            if os.path.exists(mesh_file):
                self.mesh2d = Mesh(mesh_file)
            else:
                raise IOError(f'Mesh file "{mesh_file}" needs to be generated.')
        else:
            self.mesh2d = mesh
        self.setup_mesh(self.mesh2d)

        # Physics
        self.tracer_only = True
        u = Constant(1.15)  # Maximum fluid speed
        self.horizontal_velocity_scale = u
        D = Constant(10.0)
        self.horizontal_diffusivity_scale = D
        self.bathymetry2d = Constant(50.0)
        self.bnd_conditions = {
            2: {'value': self.background_salinity},
            4: {'value': self.background_salinity},
        }

        # Outlet parametrisation
        self.outlet_value = 2.0  # Discharge rate
        outlet_x = 0.0
        outlet_y = 100.0
        outlet_r = 25.0
        x, y = SpatialCoordinate(self.mesh2d)
        self.source = self.outlet_value*exp(-((x - outlet_x)**2 + (y - outlet_y)**2)/outlet_r**2)
        self.add_tracer_2d('salinity_2d',
                           'Depth averaged salinity',
                           'Salinity2d',
                           shortname='Salinity',
                           diffusivity=D,
                           source=self.source)

        # Inlet parametrisation
        inlet_x = 0.0 if configuration == 'aligned' else 400.0
        inlet_y = -100.0
        inlet_r = 25.0
        ball = conditional((x - inlet_x)**2 + (y - inlet_y)**2 < inlet_r**2, 1, 0)
        area = assemble(ball*dx)
        scaling = 1.0 if np.isclose(area, 0.0) else pi*inlet_r**2/area
        self.qoi_kernel = scaling*ball

        # Spatial discretisation
        self.tracer_element_family = family
        self.use_lax_friedrichs_tracer = family == 'dg'
        self.use_supg_tracer = family == 'cg'
        self.use_limiter_for_tracers = family == 'dg'

        # Temporal discretisation
        self.tracer_timestepper_type = 'CrankNicolson'
        self.timestep = 2.232
        self.simulation_export_time = 5*self.timestep
        T_tide = 0.05*self.M2_tide_period
        self.simulation_end_time = 2*T_tide

        # Tidal forcing
        self.tc = Constant(0.0)
        self.forced_velocity = as_vector([u*sin(2*pi/T_tide*self.tc), 0.0])

        # Solver parameters
        self.tracer_timestepper_options.solver_parameters.update({
            'ksp_converged_reason': None,
            'ksp_max_it': 10000,
            'ksp_type': 'gmres',
            'pc_type': 'bjacobi',
        })

        # I/O
        self.fields_to_export = ['salinity_2d']
        self.fields_to_export_hdf5 = []

        self._isfrozen = True

    def apply_boundary_conditions(self, solver_obj):
        if len(solver_obj.function_spaces.keys()) == 0:
            solver_obj.create_function_spaces()
        solver_obj.bnd_functions['salinity'] = self.bnd_conditions

    def apply_initial_conditions(self, solver_obj):
        if len(solver_obj.function_spaces.keys()) == 0:
            solver_obj.create_function_spaces()
        self.tc.assign(0.0)
        uv = Function(solver_obj.function_spaces.U_2d)
        uv.interpolate(self.forced_velocity)
        solver_obj.assign_initial_conditions(uv=uv)

    def get_update_forcings(self, solver_obj):
        """
        Get a function for updating the tidal forcing.
        """
        def update_forcings(t):
            self.tc.assign(t)
            solver_obj.fields.uv_2d.interpolate(self.forced_velocity)

        return update_forcings

    def qoi(self, solution, quadrature_degree=12):
        dx_qoi = dx(degree=quadrature_degree)
        return assemble(self.qoi_kernel*solution*dx_qoi)
