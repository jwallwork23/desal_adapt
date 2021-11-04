from thetis import *
from thetis.configuration import PositiveFloat, FiredrakeScalarExpression
from desal_adapt.options import PlantOptions


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
    # background_salinity = FiredrakeScalarExpression(Constant(39.0)).tag(config=True)
    background_salinity = FiredrakeScalarExpression(Constant(0.0)).tag(config=True)
    tracer_old = FiredrakeScalarExpression(None, allow_none=True).tag(config=True)

    def __init__(self, configuration='aligned', level=0, family='cg', mesh=None, **kwargs):
        """
        :kwarg configuration: choose from 'aligned' and 'offset'
        :kwarg level: mesh resolution level
        :kwarg family: choose from 'cg' and 'dg'
        :kwarg mesh: user-provided mesh
        :kwarg nuum_niform_refinements: number of uniform refinements
            to take over the whole domain
        """
        super(AnalyticalForcingsOptions, self).__init__(**kwargs)
        assert configuration in ('aligned', 'offset')
        level = int(np.round(level))
        assert level >= 0
        assert family in ('cg', 'dg')
        self.configuration = configuration
        self.qoi_value = 0

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
        if 'num_uniform_refinements' in kwargs:
            refs = kwargs.get('num_uniform_refinements')
            if refs > 0:
                self.mesh2d = MeshHierarchy(self.mesh2d, refs)[-1]
        self.setup_mesh(self.mesh2d)

        # Physics
        self.tracer_only = True
        u = Constant(1.15)  # Maximum fluid speed
        self.horizontal_velocity_scale = u
        D = Constant(10.0)
        self.horizontal_diffusivity_scale = D
        self.bathymetry2d = Constant(50.0)
        self.bnd_conditions = {
            1: {'diff_flux': Constant(0.0)},
            2: {'value': self.background_salinity},
            3: {'diff_flux': Constant(0.0)},
            4: {'value': self.background_salinity},
        }

        # Sponge condition for CG
        x, y = SpatialCoordinate(self.mesh2d)
        # if family == 'cg':
        #     D = D + conditional(abs(x) > 1400, exp((abs(x) - 1400)/20), 0)

        # Outlet parametrisation
        self.outlet_value = 2.0  # Discharge rate
        outlet_x = 0.0
        outlet_y = 100.0
        outlet_r = 25.0
        self.source = self.outlet_value*exp(-((x - outlet_x)**2 + (y - outlet_y)**2)/outlet_r**2)
        self.add_tracer_2d('tracer_2d',
                           'Depth averaged salinity',
                           'Salinity2d',
                           shortname='Salinity',
                           diffusivity=D,
                           source=self.source)

        # Inlet parametrisation
        inlet_x = 0.0 if self.configuration == 'aligned' else 400.0
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
        self.tide_time = 0.05*self.M2_tide_period
        self.simulation_end_time = 2*self.tide_time

        # Tidal forcing
        self.tc = Constant(0.0)
        self.forced_velocity = as_vector([u*sin(2*pi/self.tide_time*self.tc), 0.0])

        # Solver parameters
        self.tracer_timestepper_options.solver_parameters.update({
            'mat_type': 'aij',
            'ksp_type': 'preonly',
            'pc_type': 'lu',
            'pc_factor_mat_solver_type': 'mumps',
        })

        # I/O
        self.fields_to_export = ['tracer_2d']
        self.fields_to_export_hdf5 = []

        self._isfrozen = True

    def get_bnd_conditions(self, fs):
        return self.bnd_conditions

    def apply_boundary_conditions(self, solver_obj):
        solver_obj.bnd_functions['tracer'] = self.bnd_conditions

    def apply_initial_conditions(self, solver_obj):
        if len(solver_obj.function_spaces.keys()) == 0:
            solver_obj.create_function_spaces()
        self.tc.assign(0.0)
        uv = Function(solver_obj.function_spaces.U_2d)
        uv.interpolate(self.forced_velocity)
        if self.tracer_old is None:
            tracer = Function(solver_obj.function_spaces.Q_2d)
            tracer.assign(self.background_salinity)
            solver_obj.assign_initial_conditions(uv=uv, tracer=tracer)
        else:
            solver_obj.assign_initial_conditions(uv=uv, tracer=self.tracer_old)

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
        return assemble(self.qoi_kernel*(solution - self.background_salinity)*dx_qoi)

    def rebuild_mesh_dependent_components(self, mesh):
        self.mesh2d = mesh
        if 'tracer_2d' in field_metadata:
            field_metadata.pop('tracer_2d')
        if 'tracer_2d' in self.tracer:
            self.tracer.pop('tracer_2d')

        # Outlet parametrisation
        self.outlet_value = 2.0  # Discharge rate
        outlet_x = 0.0
        outlet_y = 100.0
        outlet_r = 25.0
        x, y = SpatialCoordinate(self.mesh2d)
        self.source = self.outlet_value*exp(-((x - outlet_x)**2 + (y - outlet_y)**2)/outlet_r**2)
        self.add_tracer_2d('tracer_2d',
                           'Depth averaged salinity',
                           'Salinity2d',
                           shortname='Salinity',
                           diffusivity=self.horizontal_diffusivity_scale,
                           source=self.source)

        # Inlet parametrisation
        inlet_x = 0.0 if self.configuration == 'aligned' else 400.0
        inlet_y = -100.0
        inlet_r = 25.0
        ball = conditional((x - inlet_x)**2 + (y - inlet_y)**2 < inlet_r**2, 1, 0)
        area = assemble(ball*dx)
        scaling = 1.0 if np.isclose(area, 0.0) else pi*inlet_r**2/area
        self.qoi_kernel = scaling*ball
