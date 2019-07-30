import h5py
import pytest

from ef.config.config import Config
from ef.output import OutputWriterNumberedH5
from ef.output.cpp import OutputWriterCpp
from ef.output.python import OutputWriterPython
from test_simulation import TestSimulation


class TestOutput:
    @pytest.mark.parametrize('Writer', [OutputWriterCpp, OutputWriterPython])
    def test_write(self, monkeypatch, tmpdir, Writer):
        monkeypatch.chdir(tmpdir)
        sim = TestSimulation.sim_full_config.make()
        writer = Writer('test_', '.ext')
        writer.write(sim, 'asdf')
        assert tmpdir.join('test_asdf.ext').exists()
        writer.write(sim)
        assert tmpdir.join('test_0000000.ext').exists()

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
