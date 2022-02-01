project('example', 'cpp')

sourcelist = ['src/assembly.cpp',
    'src/body.cpp',
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
    'src/solvers.cpp',
    'src/RHSA.cpp',
    'src/RHS-odeint.cpp']

cc = meson.get_compiler('cpp')
dep1 = cc.find_library('m')
dep2 = cc.find_library('nlopt', dirs : '/usr/local/lib/')
libgomp = cc.find_library('gomp')

openmp = dependency('openmp')
boost_dep = dependency('boost')

executable('main', 'main.cpp', sources : sourcelist, dependencies : [dep1, dep2, libgomp, boost_dep],
            cpp_args : '-fopenmp')

# Tests:
# acutests = executable('tests', 'tests.cpp', sources : sourcelist, dependencies : [libgomp, boost_dep])
# test('unitTest', acutests)

# Timings:
# executable('timing', 'timing.cpp', dependencies : libgomp)

# Utils test (time of execution)
# executable('utilsTest', 'misc/utilsTests.cpp', sources : sourcelist, dependencies : libgomp)

# Jacobi matrices building performance test
# executable('jacobiBuildPerf', 'misc/jacobiBuildPerf.cpp', sources : sourcelist,  dependencies : libgomp)

# Legacy

# compile main project
# https://mesonbuild.com/Build-options.html#visual-studio

# Add openmp support. Note: tdhis can be done with a single dependency, but we keep verbose version to show more syntax
# https://mesonbuild.com/Adding-arguments.html
# https://stackoverflow.com/questions/45465892/how-to-use-openmp-in-a-c-project-built-with-meson

# https://mesonbuild.com/howtox.html#add-math-library-lm-portably

# === NLOPT setup: ===
# In case the executable file cannot open shared object file libnlopt.so.* - try manual linking
# global setup (picked here):
# ln -s /usr/local/lib/libnlopt.so.0.10.0 /lib/x86_64-linux-gnu/libnlopt.so.0

# https://opensource.com/article/20/6/linux-libraries
# session based setup:
# LD_LIBRARY_PATH=$(pwd):$LD_LIBRARY_PATH (or LD_LIBRARY_PATH=/usr/local/lib:LD_LIBRARY_PATH)
# export LD_LIBRARY_PATH (unset LD_LIBRARY_PATH to revert)