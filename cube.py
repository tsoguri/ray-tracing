from utils import *
from ray import *
from cli import render

tan = Material(vec([0.7, 0.7, 0.4]), 0.6)
gray = Material(vec([0.2, 0.2, 0.2]))

# Read the triangle mesh for a 2x2x2 cube, and scale it down to 1x1x1 to fit the scene.
vs_list = 0.5 * read_obj_triangles(open("cube.obj"))

scene = Scene(
    [
        # Make a big sphere for the floor
        Sphere(vec([0, -40, 0]), 39.5, gray),
    ] + [
        # Make triangle objects from the vertex coordinates
        Triangle(vs, tan) for vs in vs_list
    ])

lights = [
    PointLight(vec([12, 10, 5]), vec([300, 300, 300])),
    AmbientLight(0.1),
]

camera = Camera(vec([3, 1.7, 5]), target=vec([0, 0, 0]), vfov=25, aspect=16/9)

render(camera, scene, lights)
