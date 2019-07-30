from ef.output import OutputWriterNone
from ef.simulation import Simulation


class Reader:
    @staticmethod
    def guess_h5_format(h5file):
        if 'SpatialMesh' in h5file:
            return 'cpp'
        elif 'history' in h5file:
            return 'history'
        elif 'spat_mesh' in h5file:
            return 'python'
        else:
            raise ValueError('Cannot guess hdf5 file format')

    @staticmethod
    def read_simulation(h5file):
        format_ = Reader.guess_h5_format(h5file)
        if format_ == 'cpp':
            return Simulation.import_from_h5(h5file, OutputWriterNone())
        elif format_ == 'python':
            return Simulation.load_h5(h5file)
        else:
            return Simulation.load_h5(h5file['simulation'])
