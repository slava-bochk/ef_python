import io
from configparser import ConfigParser

from ef import simulation
from ef.config.components import *
from ef.config.section import ConfigSection
from ef.util.data_class import DataClass


class Config(DataClass):
    def __init__(self, time_grid=TimeGridConf(), spatial_mesh=SpatialMeshConf(), sources=(), inner_regions=(),
                 output_file=OutputFileConf(), boundary_conditions=BoundaryConditionsConf(),
                 particle_interaction_model=ParticleInteractionModelConf(), external_fields=()):
        self.time_grid = time_grid
        self.spatial_mesh = spatial_mesh
        self.sources = list(sources)
        self.inner_regions = list(inner_regions)
        self.output_file = output_file
        self.boundary_conditions = boundary_conditions
        self.particle_interaction_model = particle_interaction_model
        self.external_fields = list(external_fields)

    @classmethod
    def from_components(cls, components):
        parents = {'time_grid': TimeGridConf, 'spatial_mesh': SpatialMeshConf,
                   'sources': ParticleSourceConf, 'inner_regions': InnerRegionConf,
                   'output_file': OutputFileConf, 'boundary_conditions': BoundaryConditionsConf,
                   'particle_interaction_model': ParticleInteractionModelConf,
                   'external_fields': FieldConf}
        singletons = TimeGridConf, SpatialMeshConf, OutputFileConf, BoundaryConditionsConf, ParticleInteractionModelConf
        kwargs = {}
        for arg, parent in parents.items():
            children = [c for c in components if isinstance(c, parent)]
            if parent in singletons:
                if len(children) > 1:
                    raise Exception("Several {} configured, cannot init Config".format(parent))
                if len(children) < 1:
                    raise Exception("No {} configuration found, cannot init Config".format(parent))
                kwargs[arg] = children[0]
            else:
                kwargs[arg] = children
        return cls(**kwargs)

    @classmethod
    def from_configparser(cls, parser):
        return cls.from_components([section.make() for section in ConfigSection.parser_to_confs(parser)])

    @classmethod
    def from_fname(cls, fname):
        parser = ConfigParser()
        parser.read(fname)
        return cls.from_configparser(parser)

    @classmethod
    def from_file(cls, file):
        parser = ConfigParser()
        parser.read_file(file)
        return cls.from_configparser(parser)

    @classmethod
    def from_string(cls, s):
        parser = ConfigParser()
        parser.read_string(s)
        return cls.from_configparser(parser)

    @property
    def components(self):
        return [self.time_grid, self.spatial_mesh] + self.sources + self.inner_regions + \
               [self.output_file, self.boundary_conditions, self.particle_interaction_model] + self.external_fields

    def get_potentials(self):
        bc = self.boundary_conditions
        return [bc.left, bc.right, bc.top, bc.bottom, bc.near, bc.far] + [region.potential for region in
                                                                          self.inner_regions]

    def to_sections(self):
        return [c.to_conf() for c in self.components]

    def visualize_all(self, visualizer):
        p = self.get_potentials()
        visualizer.set_potential_lim(min(p), max(p))
        self.boundary_conditions.visualize(visualizer, self.spatial_mesh.size)
        visualizer.visualize(self.sources)
        visualizer.visualize(self.inner_regions)
        visualizer.visualize(self.external_fields)
        visualizer.show()

    def export_to_fname(self, fname):
        with open(fname, 'w') as f:
            self.export_to_file(f)

    def export_to_file(self, file):
        parser = ConfigParser()
        for section in self.to_sections():
            section.add_section_to_parser(parser)
        parser.write(file)

    def export_to_string(self):
        iostr = io.StringIO()
        self.export_to_file(iostr)
        return iostr.getvalue()

    def make(self):
        grid = self.time_grid.make()
        mesh, charge, potential, electric_field = self.spatial_mesh.make()
        potential.apply_boundary_values(self.boundary_conditions)
        regions = [ir.make() for ir in self.inner_regions]
        sources = [s.make() for s in self.sources]
        fields = [f.make() for f in self.external_fields]
        electric_fields = [f for f in fields if f.electric_or_magnetic == 'electric']
        magnetic_fields = [f for f in fields if f.electric_or_magnetic == 'magnetic']
        model = self.particle_interaction_model.make()
        return simulation.Simulation(grid, mesh, charge, potential, electric_field,
                                     regions, sources, electric_fields, magnetic_fields, model)

    def is_trivial(self):
        if not self.boundary_conditions.is_the_same_on_all_boundaries:
            return False
        return len({self.boundary_conditions.right} | {ir.potential for ir in self.inner_regions}) == 1


def main():
    ef = Config()
    print(ef)


if __name__ == "__main__":
    main()
