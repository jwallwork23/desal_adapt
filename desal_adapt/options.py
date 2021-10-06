from thetis.options import ModelOptions2d
from thetis.configuration import PositiveFloat, Bool


__all__ = ["PlantOptions"]


class PlantOptions(ModelOptions2d):
    """
    Base class for parameters associated with
    tidal farm problems.
    """
    sea_water_density = PositiveFloat(1030.0, help="""
        Density of sea water in kg m^{-3}.

        This is used when computing power and
        energy output.
        """).tag(config=True)
    M2_tide_period = PositiveFloat(12.4*3600, help="""
        Period of the M2 tidal constituent in
        seconds.
        """).tag(config=False)
    gravitational_acceleration = PositiveFloat(9.81, help="""
        Gravitational acceleration in m s^{-2}.
        """).tag(config=False)
    discrete_pipes = Bool(False, help="""
        Toggle whether to consider inlet and outlet pipes
        as indicator functions over their footprints (discrete)
        or as a density field (continuous).
        """).tag(config=True)

    def __init__(self, **kwargs):
        super(PlantOptions, self).__init__()
        self._isfrozen = False
        self.update(kwargs)

    def get_update_forcings(self):
        return lambda t: None

    @property
    def update_forcings(self):
        return self.get_update_forcings()

    def get_export_func(self):
        return lambda: None

    @property
    def export_func(self):
        return self.get_export_func()

    def get_bnd_conditions(self, V_2d):
        self.bnd_conditions = {}

    def apply_boundary_conditions(self, solver_obj):
        """
        Should be implemented in derived class.
        """
        pass

    def apply_initial_conditions(self, solver_obj):
        """
        Should be implemented in derived class.
        """
        pass

    def get_depth(self):
        if hasattr(self, 'depth'):
            return self.depth
        elif hasattr(self, 'max_depth'):
            return self.max_depth
        elif hasattr(self, 'bathymetry2d'):
            return self.bathymetry2d.vector().gather().max()
        else:
            raise ValueError("Cannot deduce maximum depth")

    @property
    def courant_number(self):
        """
        Compute the Courant number based on velocity
        scale and minimum element spacing.
        """
        u = self.horizontal_velocity_scale
        delta_x = self.mesh2d.delta_x.vector().gather().min()
        return u*self.timestep/delta_x
