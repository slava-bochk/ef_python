from ef.field import uniform

__all__ = ["ExternalMagneticFieldUniformConf", "ExternalElectricFieldUniformConf",
           "ExternalMagneticFieldUniformSection", "ExternalElectricFieldUniformSection"]

from collections import namedtuple

import numpy as np

from ef.config.components.fields.field import FieldConf
from ef.config.section import NamedConfigSection


class ExternalMagneticFieldUniformConf(FieldConf):
    def __init__(self, name="ExternalFieldUniform_1",
                 field=(0, 0, 0)):
        self.name = name
        self.field = np.array(field, np.float)

    def to_conf(self):
        return ExternalMagneticFieldUniformSection(self.name, *self.field)

    def make(self):
        return uniform.FieldUniform(self.name, 'magnetic', self.field)

class ExternalElectricFieldUniformConf(FieldConf):
    def __init__(self, name="ExternalFieldUniform_1",
                 field=(0, 0, 0)):
        self.name = name
        self.field = np.array(field, np.float)

    def to_conf(self):
        return ExternalElectricFieldUniformSection(self.name, *self.field)

    def make(self):
        return uniform.FieldUniform(self.name, 'electric', self.field)


class ExternalMagneticFieldUniformSection(NamedConfigSection):
    section = "ExternalMagneticFieldUniform"
    ContentTuple = namedtuple("ExternalMagneticFieldUniform",
                              ('magnetic_field_x', 'magnetic_field_y', 'magnetic_field_z'))
    convert = ContentTuple(float, float, float)

    def make(self):
        return ExternalMagneticFieldUniformConf(self.name, self.content)

    @classmethod
    def _from_section(cls, section):
        section.pop('speed_of_light', None)
        return super()._from_section(section)


class ExternalElectricFieldUniformSection(NamedConfigSection):
    section = "ExternalElectricFieldUniform"
    ContentTuple = namedtuple("ExternalElectricFieldUniform",
                              ('electric_field_x', 'electric_field_y', 'electric_field_z'))
    convert = ContentTuple(float, float, float)

    def make(self):
        return ExternalElectricFieldUniformConf(self.name, self.content)
