import numpy as np
from h5py import File

from ef.field.expression import FieldExpression
from ef.field.from_csv import FieldFromCSVFile
from ef.field.on_grid import FieldOnGrid
from ef.field.uniform import FieldUniform
from ef.output import OutputWriterNumberedH5
from ef.particle_array import ParticleArray
from ef.util.physical_constants import speed_of_light


class OutputWriterCpp(OutputWriterNumberedH5):
    def do_write(self, sim: 'Simulation', h5file: File) -> None:
        gg = h5file.create_group('SpatialMesh')
        sim.mesh.export_h5(gg)
        for i, c in enumerate('xyz'):
            gg[f'electric_field_{c}'] = sim.electric_field.array.data[..., i].flatten()
        gg['charge_density'] = sim.charge_density.data.flatten()
        gg['potential'] = sim.potential.data.flatten()

        sim.time_grid.export_h5(h5file.create_group('TimeGrid'))
        g = h5file.create_group('ParticleSources')
        g.attrs['number_of_sources'] = [len(sim.particle_sources)]
        for s in sim.particle_sources:
            s.export_h5(g.create_group(s.name))
        for p in sim.particle_arrays:
            s = next(s for s in sim.particle_sources if s.charge == p.charge and s.mass == p.mass)
            p.export_h5(g[s.name])
        for s in sim.particle_sources:
            if 'particle_id' not in g[s.name]:
                ParticleArray([], s.charge, s.mass, np.empty((0, 3)), np.empty((0, 3)), True).export_h5(g[s.name])

        g = h5file.create_group('InnerRegions')
        g.attrs['number_of_regions'] = [len(sim.inner_regions)]
        for s in sim.inner_regions:
            s.export_h5(g.create_group(s.name))

        g = h5file.create_group('ExternalFields')
        if sim.electric_fields.__class__.__name__ == "FieldZero":
            ff = []
        elif sim.electric_fields.__class__.__name__ == "FieldSum":
            ff = sim.electric_fields.fields
        else:
            ff = [sim.electric_fields]
        g.attrs['number_of_electric_fields'] = len(ff)
        for s in ff:
            sg = g.create_group(s.name)
            if s.__class__ is FieldUniform:
                ft = 'electric_uniform'
                for i, c in enumerate('xyz'):
                    sg.attrs[f'electric_uniform_field_{c}'] = s.uniform_field_vector[i]
            elif s.__class__ is FieldExpression:
                ft = 'electric_tinyexpr'
                for i, c in enumerate('xyz'):
                    expr = getattr(s, f'expression_{c}')
                    expr = np.string_(expr.encode('utf8')) + b'\x00'
                    sg.attrs[f'electric_tinyexpr_field_{c}'] = expr
            elif s.__class__ is FieldFromCSVFile:
                ft = 'electric_on_regular_grid'
                sg.attrs['h5filename'] = np.string_(s.field_filename.encode('utf8') + b'\x00')
            sg.attrs['field_type'] = np.string_(ft.encode('utf8') + b'\x00')

        if sim.magnetic_fields.__class__.__name__ == "FieldZero":
            ff = []
        elif sim.magnetic_fields.__class__.__name__ == "FieldSum":
            ff = sim.magnetic_fields.fields
        else:
            ff = [sim.magnetic_fields]
        g.attrs['number_of_magnetic_fields'] = len(ff)
        for s in ff:
            sg = g.create_group(s.name)
            if s.__class__ is FieldUniform:
                ft = 'magnetic_uniform'
                sg.attrs['speed_of_light'] = speed_of_light
                for i, c in enumerate('xyz'):
                    sg.attrs[f'magnetic_uniform_field_{c}'] = s.uniform_field_vector[i]
            elif s.__class__ is FieldExpression:
                ft = 'magnetic_tinyexpr'
                sg.attrs['speed_of_light'] = speed_of_light
                for i, c in enumerate('xyz'):
                    expr = getattr(s, f'expression_{c}')
                    expr = np.string_(expr.encode('utf8')) + b'\x00'
                    sg.attrs[f'magnetic_tinyexpr_field_{c}'] = expr
            elif s.__class__ is FieldFromCSVFile:
                ft = 'magnetic_on_regular_grid'
                sg.attrs['h5filename'] = np.string_(s.field_filename.encode('utf8') + b'\x00')
            sg.attrs['field_type'] = np.string_(ft.encode('utf8') + b'\x00')

        g = h5file.create_group('ParticleInteractionModel')
        g.attrs['particle_interaction_model'] = \
            np.string_(sim.particle_interaction_model.name.encode('utf8') + b'\x00')
