from pyroteus.thetis_compat import FlowSolver2d
from thetis.utility import AttrDict, FieldDict
from thetis.callback import CallbackManager
from thetis.field_defs import field_metadata


__all__ = ["PlantSolver"]


class PlantSolver(FlowSolver2d):
    """
    Modified solver which accepts :class:`ModelOptions2d` objects
    with more attributes than expected.
    """
    def __init__(self, options, mesh=None):
        """
        :arg options: :class:`PlantOptions` parameter object
        :kwarg mesh: :class:`MeshGeometry` upon which to solve
        """
        self._initialized = False
        self.options = options
        self.mesh2d = mesh or options.mesh2d
        self.comm = self.mesh2d.comm

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

    def create_function_spaces(self):
        super(PlantSolver, self).create_function_spaces()
        self.options.Q_2d = self.function_spaces.Q_2d

    def create_equations(self):
        super(PlantSolver, self).create_equations()
        self.options.test_function = self.equations.tracer_2d.test
