from configparser import ConfigParser

import pytest

from ef.config.components import *
from ef.config.config import Config
from ef.config.section import ConfigSection

comp_list = [BoundaryConditionsConf, InnerRegionConf, OutputFileConf, ParticleInteractionModelConf,
             ParticleSourceConf, SpatialMeshConf, TimeGridConf,
             ExternalMagneticFieldUniformConf, ExternalElectricFieldUniformConf]


def test_components_to_conf_and_back(backend):
    for Component in comp_list:
        x = Component()
        y = x.to_conf().make()
        assert x == y


def test_conf_to_configparser_and_back(backend):
    confs = [C().to_conf() for C in comp_list]
    parser = ConfigParser()
    for c in confs:
        c.add_section_to_parser(parser)
    conf2 = ConfigSection.parser_to_confs(parser)
    assert conf2 == confs


def test_minimal_example():
    parser = ConfigParser()
    parser.read("examples/minimal_working_example/minimal_conf.conf")
    components = [conf.make() for conf in ConfigSection.parser_to_confs(parser)]
    assert components == [TimeGridConf(1e-7, 1e-9, 1e-9), SpatialMeshConf((5, 5, 15), (0.5, 0.5, 1.5)),
                          ParticleInteractionModelConf('noninteracting'), BoundaryConditionsConf(0),
                          ExternalMagneticFieldUniformConf('mgn_uni'),
                          ExternalElectricFieldUniformConf('el_uni'),
                          OutputFileConf('example_', '.h5')]


@pytest.mark.usefixtures("backend")
class TestEfConf:
    def test_conf_export(self):
        conf = Config(sources=[ParticleSourceConf()], inner_regions=(InnerRegionConf(),))
        s = conf.export_to_string()
        c1 = Config.from_string(s)
        assert c1 == conf

    def test_conf_repr(self):
        # noinspection PyUnresolvedReferences
        from numpy import array  # for use in eval
        conf = Config(sources=[ParticleSourceConf()], inner_regions=(InnerRegionConf(),))
        s = repr(conf)
        c1 = eval(s)
        assert c1 == conf


class TestPrint:
    def test_time_grid(self):
        assert repr(TimeGridConf()) == "TimeGridConf(total=100.0, save_step=10.0, step=1.0)"
        assert str(TimeGridConf()) == ("### TimeGridConf:\n"
                                       "total = 100.0\n"
                                       "save_step = 10.0\n"
                                       "step = 1.0")


def test_potentials():
    assert Config().get_potentials() == [0., 0., 0., 0., 0., 0.]


def test_is_trivial(backend):
    assert BoundaryConditionsConf().is_the_same_on_all_boundaries
    assert BoundaryConditionsConf(3.14).is_the_same_on_all_boundaries
    assert not BoundaryConditionsConf(*range(6)).is_the_same_on_all_boundaries
    assert not BoundaryConditionsConf(-1, 1, 1, 1, 1, 1).is_the_same_on_all_boundaries
    assert Config().is_trivial()
    assert Config(boundary_conditions=BoundaryConditionsConf(1)).is_trivial()
    assert not Config(boundary_conditions=BoundaryConditionsConf(1, 1, 1, 2, 1, 1)).is_trivial()
    assert Config(inner_regions=[InnerRegionConf(), InnerRegionConf()]).is_trivial()
    assert Config(inner_regions=[InnerRegionConf(potential=-2), InnerRegionConf(potential=-2)],
                  boundary_conditions=BoundaryConditionsConf(-2)).is_trivial()
    assert not Config(inner_regions=[InnerRegionConf(potential=1)]).is_trivial()
    assert not Config(inner_regions=[InnerRegionConf(potential=1), InnerRegionConf(potential=0)]).is_trivial()
