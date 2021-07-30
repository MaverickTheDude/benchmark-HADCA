# compile main project
project('example', 'cpp')
sourcelist = ['src/assembly.cpp',
    'src/body.cpp',
    'src/derivatives.cpp',
    'src/input.cpp',
    'src/ksiCoefs.cpp',
    'src/utils.cpp']
executable('main', 'main.cpp', sources : sourcelist)

# tests:
e = executable('tests', 'tests.cpp', sources : sourcelist)
test('unitTest', e)