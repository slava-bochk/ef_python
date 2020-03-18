from ef.output import OutputWriterNumberedH5
from ef.util.serializable_h5 import tree_to_hdf5


class OutputWriterPython(OutputWriterNumberedH5):
    @classmethod
    def do_write(cls, sim, h5file):
        tree_to_hdf5(sim.tree, h5file)
