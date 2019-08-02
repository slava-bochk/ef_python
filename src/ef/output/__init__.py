import h5py


class OutputWriter:
    def write(self, sim, name=None):
        raise NotImplementedError()


class OutputWriterNone(OutputWriter):
    def write(self, sim, name=None):
        pass


class OutputWriterNumberedH5(OutputWriter):
    def __init__(self, prefix, suffix):
        self.prefix = prefix
        self.suffix = suffix

    def format_filename(self, specific_name):
        return "{prefix}{name}{suffix}".format(prefix=self.prefix, suffix=self.suffix, name=specific_name)

    @classmethod
    def get_filename(cls, sim):
        return "{:07}".format(sim.time_grid.current_node)

    def write(self, sim, name=None):
        if name is None:
            name = self.get_filename(sim)
        name = self.format_filename(name)
        print("Writing to file {}".format(name))
        with h5py.File(name, 'w') as h5file:
            self.do_write(sim, h5file)

    @classmethod
    def do_write(cls, sim, h5file):
        raise NotImplementedError()
