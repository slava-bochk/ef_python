import os
import subprocess
from os.path import basename
from shutil import copyfile

import pytest

from ef.main import main

_examples_conf = [("examples/minimal_working_example/minimal_conf.conf", ()),
                  ("examples/single_particle_in_free_space/single_particle_in_free_space.conf",
              pytest.mark.slowish),
                  ("examples/single_particle_in_magnetic_field/single_particle_in_magnetic_field.conf",
              pytest.mark.slowish),
                  ("examples/single_particle_in_magnetic_field/large_time_step.conf",
              pytest.mark.slowish),
                  ("examples/tube_source_test/contour.conf",
              pytest.mark.slow),
                  ("examples/single_particle_in_radial_electric_field/single_particle_in_radial_electric_field.conf",
              pytest.mark.slowish),
                  ("examples/ribbon_beam_contour/contour_bin.conf",
              pytest.mark.slow),
                  ("examples/ribbon_beam_contour/contour.conf",
              pytest.mark.slow),
                  ("examples/drift_tube_potential/pot.conf",
              pytest.mark.slow)]

_examples_jupyter = [("examples/axially_symmetric_beam_contour/axially_symmetric_beam_contour.ipynb",
                      pytest.mark.slow)]

_pytest_params_example_conf = [pytest.param(f.replace('/', os.path.sep), marks=m) for f, m in _examples_conf]
_pytest_params_example_jupyter = [pytest.param(f.replace('/', os.path.sep), marks=m) for f, m in _examples_jupyter]


@pytest.mark.parametrize("fname", _pytest_params_example_conf)
def test_example_conf(fname, mocker, capsys, tmpdir, monkeypatch):
    copyfile(fname, tmpdir.join(basename(fname)))
    monkeypatch.chdir(tmpdir)
    mocker.patch("sys.argv", ["main.py", str(basename(fname))])
    main()
    out, err = capsys.readouterr()
    assert err == ""


@pytest.mark.parametrize("fname", _pytest_params_example_jupyter)
def test_example_jupyter(fname, tmpdir, monkeypatch):
    import nbformat
    from nbconvert.preprocessors import ExecutePreprocessor
    with open(fname) as f:
        nb = nbformat.read(f, as_version=4)
    ep = ExecutePreprocessor(timeout=600)
    ep.preprocess(nb, {'metadata': {'path': tmpdir}})


@pytest.mark.requires_install
@pytest.mark.parametrize("fname", _pytest_params_example_conf)
def test_main_shell(fname, tmpdir, monkeypatch):
    basedir = os.path.join(os.path.dirname(__file__), '..')
    monkeypatch.chdir(tmpdir)
    result = subprocess.run(['ef', os.path.join(basedir, fname)], check=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    for line in result.stderr.split("\n"):
        assert line == '' or line.startswith("WARNING:")
    assert result.stdout != ""
