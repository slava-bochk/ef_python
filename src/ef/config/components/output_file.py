__all__ = ["OutputFileConf", "OutputFilenameSection"]

from collections import namedtuple

from ef.config.component import ConfigComponent
from ef.config.section import ConfigSection
from ef.output import OutputWriterNone
from ef.output.cpp import OutputWriterCpp
from ef.output.history import OutputWriterHistory
from ef.output.python import OutputWriterPython


class OutputFileConf(ConfigComponent):
    def __init__(self, prefix="out_", suffix=".h5", format_="cpp"):
        self.prefix = prefix
        self.suffix = suffix
        self.format_ = format_

    def to_conf(self):
        return OutputFilenameSection(self.prefix, self.suffix)

    def make(self):
        if self.format_ == "python":
            return OutputWriterPython(self.prefix, self.suffix)
        elif self.format_ == "cpp":
            return OutputWriterCpp(self.prefix, self.suffix)
        elif self.format_ == "history":
            return OutputWriterHistory(self.prefix, self.suffix)
        elif self.format_ == "none":
            return OutputWriterNone()
        else:
            raise ValueError("Unknown simulation output writer format")


class OutputFilenameSection(ConfigSection):
    section = "OutputFilename"
    ContentTuple = namedtuple("OutputFileNameTuple", ('output_filename_prefix', 'output_filename_suffix'))
    convert = ContentTuple(str, str)

    def make(self):
        return OutputFileConf(*self.content)
