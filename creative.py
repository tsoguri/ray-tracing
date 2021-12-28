from utils import *
from ray import *
from cli import render

tan = Material(vec([0.5, 0.2, -0.2]), k_s=0.2)
blue = Material(vec([-0.2, 0.2, 0.5]), k_m=0.3, k_s=0.3)
green = Material(vec([-0.2, 0.2, -0.4]))

vs_list = 0.2 * read_obj_triangles(open("cube.obj"))
grass_list = read_obj_triangles(open("grass_simple.obj"))
tree_list = read_obj_triangles(open("trees.obj"))

scene_arr = [
    Sphere(vec([-0.25, 0.35, 0.1]), 0.2, blue),
    # Sphere(vec([-0.5, 0.6, 0]), 0.2, blue),
    # Sphere(vec([-0.25, 0.85, 0.1]), 0.2, blue),
    # Sphere(vec([0, 0.6, 0]), 0.2, blue),
    # Sphere(vec([0.5, 0.6, 0]), 0.2, blue),
    # Sphere(vec([0.25, 0.35, -0.1]), 0.2, blue),
    # Sphere(vec([0, 1.1, 0]), 0.2, blue),
    # Sphere(vec([0.25, 0.85, -0.25]), 0.2, blue),
    # Sphere(vec([0, -40, 0]), 39.5, green),
]
scene = Scene(scene_arr + [
    (Triangle(vs, tan) for vs in vs_list) +
    (Triangle(tr, tan) for tr in grass_list)
])

lights = [
    PointLight(vec([12, 12, 8]), vec([400, 400, 400])),
    AmbientLight(0.1),
]

camera = Camera(vec([1.5, 1.5, 5]), target=vec(
    [0.1, 0.1, 0.1]), vfov=24, aspect=4/5)

render(camera, scene, lights)
