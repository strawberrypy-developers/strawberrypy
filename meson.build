project(
  'strawberrypy', 
  ['c', 'fortran'],
  version: run_command(
    'python3', '-c', 
    'import re; print(re.search(r".+version[^0-9]+([0-9.]+)", open("strawberrypy/config.py").read()).group(1))',
    check: true
  ).stdout().strip()
)


py_mod = import('python').find_installation('python3', pure: false)
ext_suffix = run_command(py_mod, '-c', 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))', check: true).stdout().strip()


if get_option('use_elpa')
cmd_diagonalize = [
  py_mod,
  '-m', 'numpy.f2py',
  '-c', meson.current_source_dir() / 'strawberrypy' / 'backends' / 'fortran' / 'diagonalize_elpa.f90',
  '-m', 'diagonalize',
  '--backend', 'meson',
]
else
cmd_diagonalize = [
  py_mod,
  '-m', 'numpy.f2py',
  '-c', meson.current_source_dir() / 'strawberrypy' / 'backends' / 'fortran' / 'diagonalize.f90',
  '-m', 'diagonalize',
  '--backend', 'meson',
]
endif

cmd_matmul = [
  py_mod,
  '-m', 'numpy.f2py',
  '-c', meson.current_source_dir() / 'strawberrypy' / 'backends' / 'fortran' / 'matmul.f90',
  '-m', 'matmul',
  '--backend', 'meson',
]

cmd_linsolve = [
  py_mod,
  '-m', 'numpy.f2py',
  '-c', meson.current_source_dir() / 'strawberrypy' / 'backends' / 'fortran' / 'linsolve.f90',
  '-m', 'linsolve',
  '--backend', 'meson',
]

if get_option('use_mpi')
  custom_target(
    'diagonalize_module',
    command: cmd_diagonalize,
    output: 'diagonalize' + ext_suffix,
    install: true,
    install_dir:  join_paths(py_mod.get_install_dir(),  'strawberrypy/backends/fortran')
  )

  custom_target(
    'matmul_module',
    command: cmd_matmul,
    output: 'matmul' + ext_suffix,
    install: true,
    install_dir:  join_paths(py_mod.get_install_dir(),  'strawberrypy/backends/fortran')
  )

  custom_target(
    'linsolve_module',
    command: cmd_linsolve,
    output: 'linsolve' + ext_suffix,
    install: true,
    install_dir:  join_paths(py_mod.get_install_dir(),  'strawberrypy/backends/fortran')
  )
endif

py_sources = [
    'strawberrypy/__init__.py',
    'strawberrypy/config.py',
    'strawberrypy/backends/__init__.py',
    'strawberrypy/backends/serial_routine.py',
    'strawberrypy/backends/serial_linalg.py',
    'strawberrypy/backends/serial_physics.py',
    'strawberrypy/backends/mpi_routine.py',
    'strawberrypy/backends/mpi_linalg.py',
    'strawberrypy/backends/mpi_physics.py',
    'strawberrypy/classes.py',
    'strawberrypy/finite_model.py',
    'strawberrypy/supercell.py',
    'strawberrypy/utils.py',
    'strawberrypy/_pythtb/__init__.py',
    'strawberrypy/_pythtb/attribute.py',
    'strawberrypy/_tbmodels/__init__.py',
    'strawberrypy/_tbmodels/attribute.py',
    'strawberrypy/_wberri/__init__.py',
    'strawberrypy/_wberri/attribute.py',
    'strawberrypy/example_models/__init__.py',
    'strawberrypy/example_models/haldane.py',
    'strawberrypy/example_models/kane_mele.py',
    'strawberrypy/postprocessing/__init__.py',
    'strawberrypy/postprocessing/averages.py',
    'strawberrypy/postprocessing/contractions.py',
    'strawberrypy/postprocessing/pair_distance.py',
    'strawberrypy/postprocessing/plotting.py',
    'strawberrypy/postprocessing/tracemarker.py',
    'strawberrypy/postprocessing/marker_io.py',
    'strawberrypy/postprocessing/correlations.py',
]

py_mod.install_sources(
    py_sources,
    preserve_path: true,
    pure: false
)
