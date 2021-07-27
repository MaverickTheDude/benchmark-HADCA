# compile main project
project('example', 'cpp')
sourcelist = ['utils.cpp', 'derivatives.cpp', 'HDCA.cpp']
executable('main', 'main.cpp', sources : sourcelist)

# tests:
e = executable('testing', 'testing.cpp', sources : sourcelist)
test('Fq', e)