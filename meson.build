# compile main project
# https://mesonbuild.com/Build-options.html#visual-studio
project('example', 'cpp', default_options: ['backend_startup_project=main'])
sourcelist = ['src/assembly.cpp',
    'src/body.cpp',
    'src/derivatives.cpp',
    'src/input.cpp',
    'src/ksiCoefs.cpp',
    'src/utils.cpp',
    'src/RHS.cpp' ]
executable('main', 'main.cpp', sources : sourcelist)

# tests:
e = executable('tests', 'tests.cpp', sources : sourcelist)
test('unitTest', e)