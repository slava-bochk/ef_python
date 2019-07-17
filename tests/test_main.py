from ef.config.components import TimeGridConf
from ef.config.config import Config
from ef.main import main


def test_main(mocker, capsys, tmpdir, monkeypatch):
    monkeypatch.chdir(tmpdir)
    config = tmpdir.join("test_main.conf")
    Config(time_grid=TimeGridConf(10, 5, 1)).export_to_fname("test_main.conf")
    mocker.patch("sys.argv", ["main.py", str(config)])
    main()
    out, err = capsys.readouterr()
    assert err == ""
    assert out == f"""Config file is:  {config}
[ TimeGrid ]
total_time = 10
time_save_step = 5
time_step_size = 1
[ SpatialMesh ]
grid_x_size = 10.0
grid_x_step = 1.0
grid_y_size = 10.0
grid_y_step = 1.0
grid_z_size = 10.0
grid_z_step = 1.0
[ OutputFilename ]
output_filename_prefix = out_
output_filename_suffix = .h5
[ BoundaryConditions ]
boundary_phi_right = 0.0
boundary_phi_left = 0.0
boundary_phi_bottom = 0.0
boundary_phi_top = 0.0
boundary_phi_near = 0.0
boundary_phi_far = 0.0
[ ParticleInteractionModel ]
particle_interaction_model = PIC
Writing initial fields to file
Writing to file out_fieldsWithoutParticles_new.h5
Writing to file out_fieldsWithoutParticles.h5
Writing step 0 to file
Writing to file out_0000000_new.h5
Writing to file out_0000000.h5
Time step from 0 to 1 of 10
Time step from 1 to 2 of 10
Time step from 2 to 3 of 10
Time step from 3 to 4 of 10
Time step from 4 to 5 of 10
Writing step 5 to file
Writing to file out_0000005_new.h5
Writing to file out_0000005.h5
Time step from 5 to 6 of 10
Time step from 6 to 7 of 10
Time step from 7 to 8 of 10
Time step from 8 to 9 of 10
Time step from 9 to 10 of 10
Writing step 10 to file
Writing to file out_0000010_new.h5
Writing to file out_0000010.h5
"""
