import h5py
import pytest

from ef.config.components import *
from ef.config.config import Config
from ef.output import OutputWriterNumberedH5, OutputWriterNone
from ef.output.cpp import OutputWriterCpp
from ef.output.history import OutputWriterHistory
from ef.output.python import OutputWriterPython
from ef.output.reader import Reader


@pytest.fixture()
def sim_full_config():
    return Config(TimeGridConf(200, 20, 2), SpatialMeshConf((5, 5, 5), (.1, .1, .1)),
                  sources=[ParticleSourceConf('a', Box()),
                           ParticleSourceConf('c', Cylinder()),
                           ParticleSourceConf('d', Tube(start=(0, 0, 0), end=(0, 0, 1)))],
                  inner_regions=[InnerRegionConf('1', Box(), 1),
                                 InnerRegionConf('2', Sphere(), -2),
                                 InnerRegionConf('3', Cylinder(), 0),
                                 InnerRegionConf('4', Tube(), 4)],
                  output_file=OutputFileConf(), boundary_conditions=BoundaryConditionsConf(-2.7),
                  particle_interaction_model=ParticleInteractionModelConf('binary'),
                  external_fields=[ExternalElectricFieldUniformConf('x', (-2, -2, 1)),
                                   ExternalMagneticFieldExpressionConf('y',
                                                                       ('0', '0',
                                                                        '3*x + sqrt(y) - z**2'))])


@pytest.mark.usefixtures('backend')
class TestOutput:
    def test_write_none(self, monkeypatch, tmpdir, sim_full_config):
        monkeypatch.chdir(tmpdir)
        sim = sim_full_config.make()
        writer = OutputWriterNone()
        writer.write(sim, 'asdf')
        writer.write(sim)
        assert tmpdir.listdir() == []

    def test_write_cpp(self, monkeypatch, tmpdir, sim_full_config):
        monkeypatch.chdir(tmpdir)
        sim = sim_full_config.make()
        writer = OutputWriterCpp('test_', '.ext')
        writer.write(sim, 'asdf')
        assert tmpdir.join('test_asdf.ext').exists()
        writer.write(sim)
        assert tmpdir.join('test_0000000.ext').exists()
        with h5py.File('test_0000000.ext') as h5file:
            assert Reader().guess_h5_format(h5file) == 'cpp'
            Reader().read_simulation(h5file).assert_eq(sim)

    def test_write_python(self, monkeypatch, tmpdir, sim_full_config):
        monkeypatch.chdir(tmpdir)
        sim = sim_full_config.make()
        writer = OutputWriterPython('test_', '.ext')
        writer.write(sim, 'asdf')
        assert tmpdir.join('test_asdf.ext').exists()
        writer.write(sim)
        assert tmpdir.join('test_0000000.ext').exists()
        with h5py.File('test_0000000.ext') as h5file:
            assert Reader().guess_h5_format(h5file) == 'python'
            Reader().read_simulation(h5file).assert_eq(sim)

    def test_write_history(self, monkeypatch, tmpdir, sim_full_config):
        monkeypatch.chdir(tmpdir)
        sim = sim_full_config.make()
        assert tmpdir.listdir() == []
        writer = OutputWriterHistory('test_', '.ext')
        assert tmpdir.join('test_history.ext').exists()
        assert writer.h5file.keys() == set()
        writer.write(sim, 'asdf')
        assert writer.h5file.keys() == set()
        writer.write(sim)
        assert writer.h5file.keys() == {'history', 'simulation'}
        with h5py.File('test_history.ext') as h5file:
            assert Reader().guess_h5_format(h5file) == 'history'
            Reader().read_simulation(h5file).assert_eq(sim)

    def test_numbered(self, capsys, mocker):
        mocker.patch('h5py.File')
        sim = Config().make()
        writer = OutputWriterNumberedH5('a_', '.b')
        mocker.patch.object(writer, 'do_write')
        assert writer.format_filename('test') == 'a_test.b'
        assert writer.get_filename(sim) == '0000000'
        writer.write(sim)
        h5py.File.assert_called_once_with('a_0000000.b', 'w')
        writer.do_write.assert_called_once()
        out, err = capsys.readouterr()
        assert err == ""
        assert out == "Writing to file a_0000000.b\n"
