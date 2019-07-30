from ef.output import OutputWriterNumberedH5


class OutputWriterPython(OutputWriterNumberedH5):
    @classmethod
    def do_write(cls, sim, h5file):
        sim.save_h5(h5file)
