import h5py
import pytest

from ef.config.components import *
from ef.config.config import Config
from ef.field.expression import FieldExpression
from ef.field.uniform import FieldUniform
from ef.inner_region import InnerRegion
from ef.meshgrid import MeshGrid
from ef.output import OutputWriterNumberedH5, OutputWriterNone
from ef.output.cpp import OutputWriterCpp
from ef.output.history import OutputWriterHistory
from ef.output.python import OutputWriterPython
from ef.output.reader import Reader
from ef.particle_interaction_model import Model
from ef.particle_source import ParticleSource
from ef.simulation import Simulation
from ef.time_grid import TimeGrid


@pytest.fixture()
def sim_full():
    return Simulation(TimeGrid(200, 2, 20), MeshGrid(5, 51),
               particle_sources=[ParticleSource('a', Box()),
                                 ParticleSource('c', Cylinder()),
                                 ParticleSource('d', Tube(start=(0, 0, 0), end=(0, 0, 1)))],
               inner_regions=[InnerRegion('1', Box(), 1),
                              InnerRegion('2', Sphere(), -2),
                              InnerRegion('3', Cylinder(), 0),
                              InnerRegion('4', Tube(), 4)],
               particle_interaction_model=Model.binary,
               electric_fields=[FieldUniform('x', 'electric', (-2, -2, 1))],
               magnetic_fields=[FieldExpression('y', 'magnetic', '0', '0', '3*x + sqrt(y) - z**2')])


@pytest.mark.usefixtures('backend')
class TestOutput:
    def test_write_none(self, monkeypatch, tmpdir, sim_full):
        monkeypatch.chdir(tmpdir)
        sim = sim_full
        writer = OutputWriterNone()
        writer.write(sim, 'asdf')
        writer.write(sim)
        assert tmpdir.listdir() == []

    def test_write_cpp(self, monkeypatch, tmpdir, sim_full):
        monkeypatch.chdir(tmpdir)
        sim = sim_full
        writer = OutputWriterCpp('test_', '.ext')
        writer.write(sim, 'asdf')
        assert tmpdir.join('test_asdf.ext').exists()
        writer.write(sim)
        assert tmpdir.join('test_0000000.ext').exists()
        with h5py.File('test_0000000.ext') as h5file:
            assert Reader().guess_h5_format(h5file) == 'cpp'
            Reader().read_simulation(h5file).assert_eq(sim)

    def test_write_python(self, monkeypatch, tmpdir, sim_full):
        monkeypatch.chdir(tmpdir)
        sim = sim_full
        writer = OutputWriterPython('test_', '.ext')
        writer.write(sim, 'asdf')
        assert tmpdir.join('test_asdf.ext').exists()
        writer.write(sim)
        assert tmpdir.join('test_0000000.ext').exists()
        with h5py.File('test_0000000.ext') as h5file:
            assert Reader().guess_h5_format(h5file) == 'python'
            Reader().read_simulation(h5file).assert_eq(sim)

    def test_write_history(self, monkeypatch, tmpdir, sim_full):
        monkeypatch.chdir(tmpdir)
        sim = sim_full
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
