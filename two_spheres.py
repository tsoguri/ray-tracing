from utils import *
from ray import *
from cli import render

tan = Material(vec([0.7, 0.7, 0.4]), 0.6)
gray = Material(vec([0.2, 0.2, 0.2]))

scene = Scene([
    Sphere(vec([0, 0, 0]), 0.5, tan),
    Sphere(vec([0, -40, 0]), 39.5, gray),
])

lights = [
    PointLight(vec([12, 10, 5]), vec([300, 300, 300])),
    AmbientLight(0.1),
]

camera = Camera(vec([3, 1.7, 5]), target=vec([0, 0, 0]), vfov=25, aspect=16/9)

render(camera, scene, lights)
