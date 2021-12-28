from utils import *
from ray import *
from cli import render

tan = Material(vec([0.4, 0.4, 0.2]), k_s=0.3, p=90, k_m=0.3)
blue = Material(vec([0.2, 0.2, 0.5]), k_m=0.5)
gray = Material(vec([0.2, 0.2, 0.2]), k_m=0.4)

scene = Scene([
    Sphere(vec([-0.7,0,0]), 0.5, tan),
    Sphere(vec([0.7,0,0]), 0.5, blue),
    Sphere(vec([0,-40,0]), 39.5, gray),
])

lights = [
    PointLight(vec([12,10,5]), vec([300,300,300])),
    AmbientLight(0.1),
]

camera = Camera(vec([3,1.2,5]), target=vec([0,-0.4,0]), vfov=24, aspect=16/9)

render(camera, scene, lights)