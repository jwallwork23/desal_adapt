from thetis.utility import *
from thetis.callback import CallbackManager
from thetis.field_defs import field_metadata
from pyroteus.thetis_compat import FlowSolver2d
from pyroteus.log import debug
from . import tracer_eq_3d


__all__ = ["PlantSolver2d", "PlantSolver3d"]


class PlantSolver2d(FlowSolver2d):
    """
    Modified solver which accepts :class:`ModelOptions2d` objects
    with more attributes than expected.
    """
    @PETSc.Log.EventDecorator('PlantSolver2d.__init__')
    def __init__(self, options, mesh=None, optimise=False):
        """
        :arg options: :class:`PlantOptions` parameter object
        :kwarg mesh: :class:`MeshGeometry` upon which to solve
        :kwarg optimise: is this a timed run?
        """
        debug("Initialising solver")
        self._initialized = False
        self.options = options
        self.mesh2d = mesh or options.mesh2d
        self.comm = self.mesh2d.comm
        self.optimise = optimise

        self.dt = options.timestep
        self.simulation_time = 0
        self.iteration = 0
        self.i_export = 0
        self.next_export_t = self.simulation_time + options.simulation_export_time

        self.callbacks = CallbackManager()
        self.fields = FieldDict()
        self._field_preproc_funcs = {}
        self.function_spaces = AttrDict()
        self.fields.bathymetry_2d = options.bathymetry2d

        self.export_initial_state = True
        self.sediment_model = None
        self.bnd_functions = {'shallow_water': {}, 'tracer': {}, 'sediment': {}}
        if 'tracer_2d' in field_metadata:
            field_metadata.pop('tracer_2d')
        self.solve_tracer = False
        self._isfrozen = True

        self.initialize()
        options.apply_boundary_conditions(self)
        options.apply_initial_conditions(self)

    @PETSc.Log.EventDecorator("PlantSolver2d.create_function_spaces")
    def create_function_spaces(self):
        """
        Creates function spaces

        Function spaces are accessible via :attr:`.function_spaces`
        object.
        """
        debug("Creating function spaces")
        super(PlantSolver2d, self).create_function_spaces()
        self.options._isfrozen = False
        self.options.Q = self.function_spaces.Q_2d
        self.options._isfrozen = True

    def create_equations(self):
        debug("Creating equations")
        super(PlantSolver2d, self).create_equations()
        self.options.test_function = self.equations.tracer_2d.test

    def create_timestepper(self):
        debug("Creating timestepper")
        super(PlantSolver2d, self).create_timestepper()

    @PETSc.Log.EventDecorator('PlantSolver2d.compute_mesh_stats')
    def compute_mesh_stats(self):
        """
        Computes number of elements, nodes etc and prints to sdtout
        """
        debug("Computing mesh stats")
        nnodes = self.function_spaces.P1_2d.dim()
        P1DG_2d = self.function_spaces.P1DG_2d
        nelem2d = int(P1DG_2d.dim()/P1DG_2d.ufl_cell().num_vertices())
        # dofs_u2d = self.function_spaces.U_2d.dim()
        dofs_tracer2d = self.function_spaces.Q_2d.dim()
        dofs_tracer2d_core = int(dofs_tracer2d/self.comm.size)

        if not self.options.tracer_only:
            raise NotImplementedError  # TODO
        print_output(f'Tracer element family: {self.options.tracer_element_family}, degree: 1')
        print_output(f'2D cell type: {self.mesh2d.ufl_cell()}')
        msg = f'2D mesh: {nnodes} vertices, {nelem2d} elements'
        if not self.optimise:
            from pyroteus.mesh_quality import get_aspect_ratios2d
            msg += f', aspect ratio {get_aspect_ratios2d(self.mesh2d).vector().gather().max():.1f}'
        print_output(msg)
        print_output(f'Number of 2D tracer DOFs: {dofs_tracer2d}')
        print_output(f'Number of cores: {self.comm.size}')
        print_output(f'Tracer DOFs per core: ~{dofs_tracer2d_core:.1f}')


class PlantSolver3d(PlantSolver2d):
    """
    Modified solver which accepts :class:`ModelOptions2d` objects
    with more attributes than expected.
    """
    @PETSc.Log.EventDecorator('PlantSolver3d.__init__')
    def __init__(self, options, mesh=None, optimise=False):
        """
        :arg options: :class:`PlantOptions` parameter object
        :kwarg mesh: :class:`MeshGeometry` upon which to solve
        :kwarg optimise: is this a timed run?
        """
        self._initialized = False
        self.options = options
        self.mesh3d = mesh or options.mesh3d
        self.comm = self.mesh3d.comm
        self.optimise = optimise

        self.dt = options.timestep
        self.simulation_time = 0
        self.iteration = 0
        self.i_export = 0
        self.next_export_t = self.simulation_time + options.simulation_export_time

        self.callbacks = CallbackManager()
        self.fields = FieldDict()
        self._field_preproc_funcs = {}
        self.fields.bathymetry_2d = Constant(1.0)  # TODO: avoid this hack
        self.function_spaces = AttrDict()

        self.export_initial_state = True
        self.sediment_model = None
        self.bnd_functions = {'shallow_water': {}, 'tracer': {}, 'sediment': {}}
        if 'tracer_3d' in field_metadata:
            field_metadata.pop('tracer_3d')
        if 'solution_2d' in field_metadata:
            field_metadata.pop('solution_2d')
        self.solve_tracer = False
        self._isfrozen = True

        self.initialize()
        options.apply_boundary_conditions(self)
        options.apply_initial_conditions(self)

    @PETSc.Log.EventDecorator('PlantSolver3d.compute_mesh_stats')
    def compute_mesh_stats(self):
        """
        Computes number of elements, nodes etc and prints to sdtout
        """
        nnodes = self.function_spaces.P1_3d.dim()
        P1DG_3d = self.function_spaces.P1DG_3d
        nelem3d = int(P1DG_3d.dim()/P1DG_3d.ufl_cell().num_vertices())
        # dofs_u3d = self.function_spaces.U_3d.dim()
        dofs_tracer3d = self.function_spaces.Q_3d.dim()
        dofs_tracer3d_core = int(dofs_tracer3d/self.comm.size)

        if not self.options.tracer_only:
            raise NotImplementedError  # TODO
        print_output(f'Tracer element family: {self.options.tracer_element_family}, degree: 1')
        print_output(f'3D cell type: {self.mesh3d.ufl_cell()}')
        msg = f'3D mesh: {nnodes} vertices, {nelem3d} elements'
        if not self.optimise:
            from pyroteus.mesh_quality import get_aspect_ratios3d
            msg += f', aspect ratio {get_aspect_ratios3d(self.mesh3d).vector().gather().max():.1f}'
        print_output(msg)
        print_output(f'Number of 3D tracer DOFs: {dofs_tracer3d}')
        print_output(f'Number of cores: {self.comm.size}')
        print_output(f'Tracer DOFs per core: ~{dofs_tracer3d_core:.1f}')

    @PETSc.Log.EventDecorator('PlantSolver3d.create_function_spaces')
    def create_function_spaces(self):
        """
        Creates function spaces

        Function spaces are accessible via :attr:`.function_spaces`
        object.
        """
        debug("Creating function spaces")
        self._isfrozen = False
        DG = 'DG' if self.mesh3d.ufl_cell().cellname() == 'tetrahedron' else 'DQ'
        self.function_spaces.P0_3d = FunctionSpace(self.mesh3d, DG, 0, name='P0_3d')
        self.function_spaces.P1_3d = FunctionSpace(self.mesh3d, 'CG', 1, name='P1_3d')
        self.function_spaces.P1DG_3d = FunctionSpace(self.mesh3d, DG, 1, name='P1DG_3d')

        # Velocity space
        if self.options.element_family == 'dg-cg':
            self.function_spaces.U_3d = VectorFunctionSpace(self.mesh3d, DG, self.options.polynomial_degree, name='U_3d')
        elif self.options.element_family == 'dg-dg':
            self.function_spaces.U_3d = VectorFunctionSpace(self.mesh3d, DG, self.options.polynomial_degree, name='U_3d')
        else:
            raise Exception('Unsupported finite element family {:}'.format(self.options.element_family))

        # Tracer space
        if self.options.tracer_element_family == 'dg':
            self.function_spaces.Q_3d = FunctionSpace(self.mesh3d, 'DG', 1, name='Q_3d')
        elif self.options.tracer_element_family == 'cg':
            self.function_spaces.Q_3d = FunctionSpace(self.mesh3d, 'CG', 1, name='Q_3d')
        else:
            raise Exception('Unsupported finite element family {:}'.format(self.options.tracer_element_family))
        self.options._isfrozen = False
        self.options.Q = self.function_spaces.Q_3d
        self.options._isfrozen = True

        self._isfrozen = True

    @PETSc.Log.EventDecorator('PlantSolver3d.create_equations')
    def create_equations(self):
        """
        Creates equation instances
        """
        if not hasattr(self.function_spaces, 'U_3d'):
            self.create_function_spaces()
        self._isfrozen = False
        # ----- fields
        self.fields.uv_3d = Function(self.function_spaces.U_3d, name='uv_3d')
        # self.fields.h_elem_size_3d = Function(self.function_spaces.P1_3d)
        # get_horizontal_elem_size_3d(self.fields.h_elem_size_3d)
        self.depth = Constant(1.0)

        # ----- Equations
        self.equations = AttrDict()
        if not self.options.tracer_only:
            raise NotImplementedError  # TODO: SWE
        for label, tracer in self.options.tracer.items():
            self.add_new_field(tracer.function or Function(self.function_spaces.Q_3d, name=label),
                               label,
                               tracer.metadata['name'],
                               tracer.metadata['filename'],
                               shortname=tracer.metadata['shortname'],
                               unit=tracer.metadata['unit'])
            if tracer.use_conservative_form:
                raise NotImplementedError  # TODO
            else:
                self.equations[label] = tracer_eq_3d.TracerEquation3D(
                    self.function_spaces.Q_3d, self.depth, self.options, self.fields.uv_3d)
        self.solve_tracer = self.options.tracer != {}
        if self.solve_tracer:
            if self.options.use_limiter_for_tracers and self.options.polynomial_degree > 0:
                self.tracer_limiter = limiter.VertexBasedP1DGLimiter(self.function_spaces.Q_3d)
            else:
                self.tracer_limiter = None
        self.options.test_function = self.equations.tracer_3d.test

        self._isfrozen = True

    @PETSc.Log.EventDecorator('PlantSolver3d.get_tracer_timestepper')
    def get_tracer_timestepper(self, integrator, label):
        """
        Gets tracer timestepper object with appropriate parameters
        """
        uv = self.fields.uv_3d
        fields = {
            'elev_3d': Constant(0.0),
            'uv_3d': uv,
            'diffusivity_h': self.options.tracer[label].diffusivity,
            'source': self.options.tracer[label].source,
            'lax_friedrichs_tracer_scaling_factor': self.options.lax_friedrichs_tracer_scaling_factor,
            'tracer_advective_velocity_factor': self.options.tracer_advective_velocity_factor,
        }
        bcs = {}
        if label in self.bnd_functions:
            bcs = self.bnd_functions[label]
        elif label[:-3] in self.bnd_functions:
            bcs = self.bnd_functions[label[:-3]]
        return integrator(self.equations[label], self.fields[label], fields, self.dt,
                          self.options.tracer_timestepper_options, bcs)

    @PETSc.Log.EventDecorator('PlantSolver3d.assign_initial_conditions')
    def assign_initial_conditions(self, uv=None, **tracers):
        r"""
        Assigns initial conditions

        :kwarg uv: Initial condition for velocity
        :type uv: vector valued :class:`Function`, :class:`Constant`, or an expression
        :kwarg tracers: Initial conditions for tracer fields
        :type tracers: scalar valued :class:`Function`\s, :class:`Constant`\s, or an expressions
        """
        if not self._initialized:
            self.initialize()
        uv_3d = self.fields.uv_3d
        if uv is not None:
            uv_3d.project(uv)
        for l, func in tracers.items():
            label = l if len(l) > 3 and l[-3:] == '_3d' else l + '_3d'
            assert label in self.options.tracer, f"Unknown tracer label {label}"
            self.fields[label].project(func)

        self.add_new_field(self.fields.uv_3d, 'solution_2d', 'dummy', 'dummy')  # TODO: avoid this hack
        self.timestepper.initialize(self.fields.uv_3d)
