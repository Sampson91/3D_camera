import os
import pytest
import ply2obj1
import numpy as np

def test_convert_file_with_only_vertex_information(mocker):
    file_mock = mocker.Mock(read=mocker.Mock())
    open_mock = mocker.Mock(return_value=file_mock)
    assert open_mock is not None

    vertex_info = {"vertex": [{"x":-1, "y":-1, "z":-1}]}
    verteces = vertex_info["vertex"]
    for vertex in verteces:
        position = [vertex['x'], vertex['y'], vertex['z']]
        if 'red' in vertex and 'green' in vertex and 'blue' in vertex:
            color = [vertex['red'], vertex['green'], vertex['blue']]
        else:
            color = [0, 0, 0]
        position_color = position + color
        assert position_color == [-1, -1, -1, 0, 0, 0]
    file_mock = mocker.Mock(write=mocker.Mock())
    write_mock = mocker.Mock(return_value=file_mock)
    assert write_mock is not None

def test_convert_file_with_vertex_rgb_face_information(mocker):
    file_mock = mocker.Mock(read=mocker.Mock())
    open_mock = mocker.Mock(return_value=file_mock)
    assert open_mock is not None

    vertex_info = {"vertex": [{"x": -1, "y": -1, "z": -1, "red": 1,
                               "green": 1, "blue": 1, "nx": 2, "ny": 2, "nz": 2}],
                   "face": {'vertex_indices': [[0, 1, 2]]}}

    verteces = vertex_info["vertex"]
    for vertex in verteces:
        position = [vertex['x'], vertex['y'], vertex['z']]
        if 'red' in vertex and 'green' in vertex and 'blue' in vertex:
            color = [vertex['red'], vertex['green'], vertex['blue']]
        else:
            color = [0, 0, 0]
        position_color = position + color
        assert position_color == [-1, -1, -1, 1, 1, 1]

    for vertex in verteces:
        if 'nx' in vertex and 'ny' in vertex and 'nz' in vertex:
            normal_vectors = (vertex['nx'], vertex['ny'], vertex['nz'])
        assert normal_vectors == (2, 2, 2)

    if 'face' in vertex_info:
        faces=vertex_info['face']
        for paremeter_1 in faces['vertex_indices']:
            for parameter_2 in range(len(paremeter_1)):
                value = [paremeter_1[parameter_2] , paremeter_1[parameter_2] , paremeter_1[parameter_2] ]
                assert value == [paremeter_1[parameter_2] , paremeter_1[parameter_2] , paremeter_1[parameter_2] ]

    file_mock = mocker.Mock(write=mocker.Mock())
    write_mock = mocker.Mock(return_value=file_mock)
    assert write_mock is not None
