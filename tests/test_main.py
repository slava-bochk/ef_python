from argparse import ArgumentTypeError
from configparser import ConfigParser
from io import StringIO

import h5py
import inject
import pytest
from pytest import raises

from ef.config.components import TimeGridConf, OutputFileConf
from ef.config.config import Config
from ef.main import main, extract_prefix_and_suffix_from_h5_filename, guess_input_type, read_conf, \
    merge_h5_prefix_suffix


def test_prefix_extraction():
    assert extract_prefix_and_suffix_from_h5_filename('asdf1234567qwer') == ('asdf', 'qwer')
    assert extract_prefix_and_suffix_from_h5_filename('asdfhistoryqwer') == ('asdf', 'qwer')
    with raises(ValueError):
        assert extract_prefix_and_suffix_from_h5_filename('asdf123456qwer')


def test_guess(tmpdir):
    h5 = tmpdir.join("test.h5")
    h5py.File(h5, 'w')
    assert guess_input_type(h5) == (False, h5)

    conf = tmpdir.join("test.conf")
    Config().export_to_fname(conf)
    p = ConfigParser()
    p.read_string(Config().export_to_string())
    assert guess_input_type(conf) == (True, p)

    with raises(ArgumentTypeError):
        guess_input_type(tmpdir.join('missing.txt'))

    text = tmpdir.join('some.txt')
    text.write('eowjfm ievmiefraivuenwaeofiunapewovjfiajief asdwouhd \n adfaef afef')
    with raises(ArgumentTypeError):
        guess_input_type(text)

    data = tmpdir.join('some.bin')
    data.write(bytes(range(255)), 'wb')
    with raises(ArgumentTypeError):
        guess_input_type(data)


def test_merge_h5_prefix_suffix(tmpdir):
    assert merge_h5_prefix_suffix('pre_1234567.suff', None, None) == ('pre_', '.suff')
    assert merge_h5_prefix_suffix('random.file', None, None) == ('random.file', '.h5')
    assert merge_h5_prefix_suffix('pre_1234567.suff', 'override', None) == ('override', '.h5')
    assert merge_h5_prefix_suffix('random.file', 'override', None) == ('override', '.h5')
    assert merge_h5_prefix_suffix('pre_1234567.suff', None, 'override') == ('', 'override')
    assert merge_h5_prefix_suffix('random.file', None, 'override') == ('', 'override')
    assert merge_h5_prefix_suffix('pre_1234567.suff', 'mypre', 'mysuf') == ('mypre', 'mysuf')
    assert merge_h5_prefix_suffix('random.file', 'mypre', 'mysuf') == ('mypre', 'mysuf')


def test_read_conf():
    p = ConfigParser()
    c = Config(output_file=OutputFileConf("conf-prefix", "conf-suffix"))
    p.read_string(c.export_to_string())
    assert read_conf(p, None, None, 'cpp') == c
    assert read_conf(p, 'prefix2', None, 'cpp') == Config(output_file=OutputFileConf("prefix2", "conf-suffix"))
    assert read_conf(p, 'prefix3', 'suffix3', 'cpp') == Config(output_file=OutputFileConf("prefix3", "suffix3"))
    assert read_conf(p, None, 'suffix4', 'cpp') == Config(output_file=OutputFileConf("conf-prefix", "suffix4"))


def test_guess_stdin(tmpdir, monkeypatch):
    monkeypatch.setattr('sys.stdin', StringIO(Config().export_to_string()))
    p = ConfigParser()
    p.read_string(Config().export_to_string())
    assert guess_input_type('-') == (True, p)


@pytest.mark.parametrize('solver_', [' ', 'amg', pytest.param('amgx', marks=pytest.mark.amgx)])
@pytest.mark.parametrize('backend_', [' ', 'numpy', pytest.param('cupy', marks=pytest.mark.cupy)])
def test_main(mocker, capsys, tmpdir, monkeypatch, solver_, backend_):
    inject.clear()
    monkeypatch.chdir(tmpdir)
    config = tmpdir.join("test_main.conf")
    Config(time_grid=TimeGridConf(10, 5, 1)).export_to_fname("test_main.conf")
    argv = ["main.py", str(config)]
    if solver_ != ' ':
        argv += ["--solver", solver_]
    if backend_ != ' ':
        argv += ["--backend", backend_]
    mocker.patch("sys.argv", argv)
    main()
    inject.clear()
    out, err = capsys.readouterr()
    assert err == ""
    assert out == f"""Trying to guess input file type: {config}
### Config:
time_grid = TimeGridConf(total=10.0, save_step=5.0, step=1.0)
spatial_mesh = SpatialMeshConf(size=array([10., 10., 10.]), step=array([1., 1., 1.]))
sources = []
inner_regions = []
output_file = OutputFileConf(prefix='out_', suffix='.h5', format_='cpp')
boundary_conditions = BoundaryConditionsConf(right=0.0, left=0.0, bottom=0.0, top=0.0, near=0.0, far=0.0)
particle_interaction_model = ParticleInteractionModelConf(model='PIC')
external_fields = []
Writing initial fields to file
Writing to file out_fieldsWithoutParticles.h5
Writing step 0 to file
Writing to file out_0000000.h5
\rTime step from 0 to 1 of 10\rTime step from 1 to 2 of 10\rTime step from 2 to 3 of 10\rTime step from 3 to 4 of 10\rTime step from 4 to 5 of 10
Writing step 5 to file
Writing to file out_0000005.h5
\rTime step from 5 to 6 of 10\rTime step from 6 to 7 of 10\rTime step from 7 to 8 of 10\rTime step from 8 to 9 of 10\rTime step from 9 to 10 of 10
Writing step 10 to file
Writing to file out_0000010.h5
"""

    argv = ["main.py", "out_0000005.h5"]
    if solver_ != ' ':
        argv += ["--solver", solver_]
    if backend_ != ' ':
        argv += ["--backend", backend_]
    mocker.patch("sys.argv", argv)
    main()
    inject.clear()
    out, err = capsys.readouterr()
    assert err == ""
    assert out == f"""Trying to guess input file type: out_0000005.h5
Continuing from h5 file: out_0000005.h5
Using output prefix and suffix: out_ .h5
\rTime step from 5 to 6 of 10\rTime step from 6 to 7 of 10\rTime step from 7 to 8 of 10\rTime step from 8 to 9 of 10\rTime step from 9 to 10 of 10
Writing step 10 to file
Writing to file out_0000010.h5
"""
