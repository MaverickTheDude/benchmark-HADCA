# compile main project
# https://mesonbuild.com/Build-options.html#visual-studio
project('example', 'cpp', default_options: ['backend_startup_project=main'])
sourcelist = ['src/assembly.cpp',
    'src/body.cpp',
    'src/derivatives.cpp',
    'src/input.cpp',
    'src/ksiCoefs.cpp',
    'src/utils.cpp',
    'src/task/h.cpp',
    'src/task/F.cpp',
    'src/task/Phi.cpp',
    'src/RHS.cpp',
    'src/solution.cpp',
    'src/task/M.cpp',
    'src/adjoint.cpp',
    'src/solvers.cpp']
executable('main', 'main.cpp', sources : sourcelist)

# tests:
e = executable('tests', 'tests.cpp', sources : sourcelist)
test('unitTest', e)

# Jacobi matrices building performance test
#executable('jacobiBuildPerf', 'misc/jacobiBuildPerf.cpp', sources: sourcelist)