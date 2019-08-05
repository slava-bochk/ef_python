import h5py

from ef.config.config import Config
from ef.output import OutputWriterNumberedH5, OutputWriterNone
from ef.output.cpp import OutputWriterCpp
from ef.output.history import OutputWriterHistory
from ef.output.python import OutputWriterPython
from ef.output.reader import Reader
from test_simulation import TestSimulation


class TestOutput:
    def test_write_none(self, monkeypatch, tmpdir):
        monkeypatch.chdir(tmpdir)
        sim = TestSimulation.sim_full_config.make()
        writer = OutputWriterNone()
        writer.write(sim, 'asdf')
        writer.write(sim)
        assert tmpdir.listdir() == []

    def test_write_cpp(self, monkeypatch, tmpdir):
        monkeypatch.chdir(tmpdir)
        sim = TestSimulation.sim_full_config.make()
        writer = OutputWriterCpp('test_', '.ext')
        writer.write(sim, 'asdf')
        assert tmpdir.join('test_asdf.ext').exists()
        writer.write(sim)
        assert tmpdir.join('test_0000000.ext').exists()
        with h5py.File('test_0000000.ext') as h5file:
            assert Reader().guess_h5_format(h5file) == 'cpp'
            assert Reader().read_simulation(h5file) == sim

    def test_write_python(self, monkeypatch, tmpdir):
        monkeypatch.chdir(tmpdir)
        sim = TestSimulation.sim_full_config.make()
        writer = OutputWriterPython('test_', '.ext')
        writer.write(sim, 'asdf')
        assert tmpdir.join('test_asdf.ext').exists()
        writer.write(sim)
        assert tmpdir.join('test_0000000.ext').exists()
        with h5py.File('test_0000000.ext') as h5file:
            assert Reader().guess_h5_format(h5file) == 'python'
            assert Reader().read_simulation(h5file) == sim

    def test_write_history(self, monkeypatch, tmpdir):
        monkeypatch.chdir(tmpdir)
        sim = TestSimulation.sim_full_config.make()
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
            assert Reader().read_simulation(h5file) == sim

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
