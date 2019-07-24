from ef.field import expression

__all__ = ["ExternalMagneticFieldExpressionConf", "ExternalElectricFieldExpressionConf",
           "ExternalMagneticFieldExpressionSection", "ExternalElectricFieldExpressionSection"]

from collections import namedtuple

from ef.config.components.fields.field import FieldConf
from ef.config.section import NamedConfigSection


class ExternalMagneticFieldExpressionConf(FieldConf):
    def __init__(self, name="ExternalFieldExpression_1",
                 field=('0', '0', '0')):
        self.name = name
        self.field = field

    def to_conf(self):
        return ExternalMagneticFieldExpressionSection(self.name, *self.field)

    def make(self):
        return expression.FieldExpression(self.name, 'magnetic', *self.field)


class ExternalElectricFieldExpressionConf(FieldConf):
    def __init__(self, name="ExternalFieldExpression_1",
                 field=('0', '0', '0')):
        self.name = name
        self.field = field

    def to_conf(self):
        return ExternalElectricFieldExpressionSection(self.name, *self.field)

    def make(self):
        return expression.FieldExpression(self.name, 'electric', *self.field)


class ExternalMagneticFieldExpressionSection(NamedConfigSection):
    section = "ExternalMagneticFieldTinyexpr"
    ContentTuple = namedtuple("ExternalMagneticFieldTinyexpr",
                              ('magnetic_field_x', 'magnetic_field_y', 'magnetic_field_z'))
    convert = ContentTuple(str, str, str)

    def make(self):
        return ExternalMagneticFieldExpressionConf(self.name, self.content)

    @classmethod
    def _from_section(cls, section):
        section.pop('speed_of_light', None)
        return super()._from_section(section)


class ExternalElectricFieldExpressionSection(NamedConfigSection):
    section = "ExternalElectricFieldTinyexpr"
    ContentTuple = namedtuple("ExternalElectricFieldTinyexpr",
                              ('electric_field_x', 'electric_field_y', 'electric_field_z'))
    convert = ContentTuple(str, str, str)

    def make(self):
        return ExternalElectricFieldExpressionConf(self.name, self.content)
