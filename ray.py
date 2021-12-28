import numpy as np
from numpy.core.numeric import array_equal
import math
from utils import *

"""
Core implementation of the ray tracer.  This module contains the classes (Sphere, Mesh, etc.)
that define the contents of scenes, as well as classes (Ray, Hit) and functions (shade) used in
the rendering algorithm, and the main entry point `render_image`.

In the documentation of these classes, we indicate the expected types of arguments with a
colon, and use the convention that just writing a tuple means that the expected type is a
NumPy array of that shape.  Implementations can assume these types are preconditions that
are met, and if they fail for other type inputs it's an error of the caller.  (This might
not be the best way to handle such validation in industrial-strength code but we are adopting
this rule to keep things simple and efficient.)
"""


class Ray:

    def __init__(self, origin, direction, start=0., end=np.inf):
        """Create a ray with the given origin and direction.

        Parameters:
          origin : (3,) -- the start point of the ray, a 3D point
          direction : (3,) -- the direction of the ray, a 3D vector (not necessarily normalized)
          start, end : float -- the minimum and maximum t values for intersections
        """
        # Convert these vectors to double to help ensure intersection
        # computations will be done in double precision
        self.origin = np.array(origin, np.float64)
        self.direction = np.array(direction, np.float64)
        self.start = start
        self.end = end


class Material:

    def __init__(self, k_d, k_s=0., p=20., k_m=0., k_a=None):
        """Create a new material with the given parameters.

        Parameters:
          k_d : (3,) -- the diffuse coefficient
          k_s : (3,) or float -- the specular coefficient
          p : float -- the specular exponent
          k_m : (3,) or float -- the mirror reflection coefficient
          k_a : (3,) -- the ambient coefficient (defaults to match diffuse color)
        """
        self.k_d = k_d
        self.k_s = k_s
        self.p = p
        self.k_m = k_m
        self.k_a = k_a if k_a is not None else k_d


class Hit:

    def __init__(self, t, point=None, normal=None, material=None):
        """Create a Hit with the given data.

        Parameters:
          t : float -- the t value of the intersection along the ray
          point : (3,) -- the 3D point where the intersection happens
          normal : (3,) -- the 3D outward-facing unit normal to the surface at the hit point
          material : (Material) -- the material of the surface
        """
        self.t = t
        self.point = point
        self.normal = normal
        self.material = material


# Value to represent absence of an intersection
no_hit = Hit(np.inf)


class Sphere:

    def __init__(self, center, radius, material):
        """Create a sphere with the given center and radius.

        Parameters:
          center : (3,) -- a 3D point specifying the sphere's center
          radius : float -- a Python float specifying the sphere's radius
          material : Material -- the material of the surface
        """
        self.center = center
        self.radius = radius
        self.material = material

    def intersect(self, ray):
        """Computes the first (smallest t) intersection between a ray and this sphere.

        Parameters:
          ray : Ray -- the ray to intersect with the sphere
        Return:
          Hit -- the hit data
        """
        o_c = ray.origin - self.center
        a = np.dot(ray.direction, ray.direction)
        b = 2.0*np.dot(ray.direction, o_c)
        c = np.dot(o_c, o_c)-self.radius**2
        disc = b**2 - 4*a*c
        if (disc < 0):
            return no_hit
        else:
            tplus = (- b + np.sqrt(disc))/(2*a)
            tminus = (- b - np.sqrt(disc))/(2*a)
            if tminus > ray.start and (tminus < tplus) and tminus < ray.end:
                t = tminus
            elif tplus > ray.start and tplus < ray.end:
                t = tplus
            else:
                return no_hit
            point = ray.origin + t*ray.direction
            normal = normalize(point - self.center)
            return Hit(t, point, normal, self.material)


class Triangle:

    def __init__(self, vs, material):
        """Create a triangle from the given vertices.

        Parameters:
          vs (3,3) -- an arry of 3 3D points that are the vertices (CCW order)
          material : Material -- the material of the surface
        """
        self.vs = vs
        self.material = material

    def intersect(self, ray):
        """Computes the intersection between a ray and this triangle, if it exists.

        Parameters:
          ray : Ray -- the ray to intersect with the triangle
        Return:
          Hit -- the hit data
        """

        a = self.vs[0]
        b = self.vs[1]
        c = self.vs[2]

        a_b = a-b
        a_c = a-c

        p = ray.origin
        d = ray.direction
        a_p = np.atleast_2d(a-p).T
        vs_d = np.concatenate(([a_b], [a_c], [d])).T

        bgt = np.linalg.solve(vs_d, a_p)
        beta = bgt[0][0]
        gamma = bgt[1][0]
        t = bgt[2][0]

        if beta <= 0 or gamma <= 0 or beta + gamma >= 1:
            return no_hit
        else:
            if t < ray.start or t > ray.end:
                return no_hit
            point = ray.origin + t*ray.direction
            normal = normalize(np.cross((b-a), (c-a)))
            return Hit(t, point, normal, self.material)


class Camera:

    def __init__(self, eye=vec([0, 0, 0]), target=vec([0, 0, -1]), up=vec([0, 1, 0]),
                 vfov=90.0, aspect=1.0):
        """Create a camera with given viewing parameters.

        Parameters:
          eye : (3,) -- the camera's location, aka viewpoint (a 3D point)
          target : (3,) -- where the camera is looking: a 3D point that appears centered in the view
          up : (3,) -- the camera's orientation: a 3D vector that appears straight up in the view
          vfov : float -- the full vertical field of view in degrees
          aspect : float -- the aspect ratio of the camera's view (ratio of width to height)
        """
        self.origin = eye
        self.d = target - self.origin
        self.up = up

        self.w = normalize(- self.d)
        self.u = normalize(np.cross(self.up, self.w))
        self.v = normalize(np.cross(self.w, self.u))

        self.aspect = aspect
        self.alpha = vfov

        len_d = np.sqrt(self.d[0]**2 + self.d[1]**2 + self.d[2]**2)

        self.height = len_d*np.tan(self.alpha/2 * np.pi/180)
        self.width = self.aspect*self.height

    def generate_ray(self, img_point):
        """Compute the ray corresponding to a point in the image.

        Parameters:
          img_point : (2,) -- a 2D point in [0,1] x [0,1], where (0,0) is the lower left
                      corner of the image and (1,1) is the upper right
        Return:
          Ray -- The ray corresponding to that image location (not necessarily normalized)
        """
        origin = self.origin

        u = (img_point[0] - 0.5)*2
        v = (img_point[1] - 0.5)*2
        i = u*self.width
        j = v*self.height
        len_d = np.sqrt(self.d[0]**2 + self.d[1]**2 + self.d[2]**2)
        direction = -len_d*self.w + i*self.u + j*self.v
        return Ray(origin, direction)


class PointLight:

    def __init__(self, position, intensity):
        """Create a point light at given position and with given intensity

        Parameters:
          position : (3,) -- 3D point giving the light source location in scene
          intensity : (3,) or float -- RGB or scalar intensity of the source
        """
        self.position = position
        self.intensity = intensity

    def illuminate(self, ray, hit, scene):
        """Compute the shading at a surface point due to this light.

        Parameters:
          ray : Ray -- the ray that hit the surface
          hit : Hit -- the hit data
          scene : Scene -- the scene, for shadow rays
        Return:
          (3,) -- the light reflected from the surface
        """
        l = normalize(self.position - hit.point)  # light ray normalized
        v = -normalize(ray.direction)  # camera ray normalized and negated
        h = (v + l)/math.sqrt((v+l).dot(v+l))
        dist = (self.position-hit.point).dot(self.position-hit.point)
        l_d = max(0, hit.normal.dot(l))/dist * self.intensity
        l_r = hit.material.k_d + hit.material.k_s * \
            (hit.normal.dot(h)**hit.material.p)

        return l_d*l_r


class AmbientLight:

    def __init__(self, intensity):
        """Create an ambient light of given intensity

        Parameters:
          intensity (3,) or float: the intensity of the ambient light
        """
        self.intensity = intensity

    def illuminate(self, ray, hit, scene):
        """Compute the shading at a surface point due to this light.

        Parameters:
          ray : Ray -- the ray that hit the surface
          hit : Hit -- the hit data
          scene : Scene -- the scene, for shadow rays
        Return:
          (3,) -- the light reflected from the surface
        """
        return self.intensity * hit.material.k_a


class Scene:

    def __init__(self, surfs, bg_color=vec([0.2, 0.3, 0.5])):
        """Create a scene containing the given objects.

        Parameters:
          surfs : [Sphere, Triangle] -- list of the surfaces in the scene
          bg_color : (3,) -- RGB color that is seen where no objects appear
        """
        self.surfs = surfs
        self.bg_color = bg_color

    def intersect(self, ray):
        """Computes the first (smallest t) intersection between a ray and the scene.

        Parameters:
          ray : Ray -- the ray to intersect with the scene
        Return:
          Hit -- the hit data
        """
        hit_current = no_hit
        for surf in self.surfs:
            hit_new = surf.intersect(ray)
            if hit_new.t < hit_current.t:
                hit_current = hit_new
        return hit_current


MAX_DEPTH = 4


def shade(ray, hit, scene, lights, depth=0):
    """Compute shading for a ray-surface intersection.

    Parameters:
      ray : Ray -- the ray that hit the surface
      hit : Hit -- the hit data
      scene : Scene -- the scene
      lights : [PointLight or AmbientLight] -- the lights
      depth : int -- the recursion depth so far
    Return:
      (3,) -- the color seen along this ray
    When mirror reflection is being computed, recursion will only proceed to a depth
    of MAX_DEPTH, with zero contribution beyond that depth.
    """

    result = (0, 0, 0)
    for light in lights:
        if hit != no_hit:
            if type(light) != AmbientLight:
                shadRay = Ray(hit.point, light.position -
                              hit.point, ray.start+.0001)
                shadRayHit = scene.intersect(shadRay)
                if shadRayHit.t > hit.t:
                    result = result + light.illuminate(ray, hit, scene)
            else:
                result = result + light.illuminate(ray, hit, scene)
                if(depth < MAX_DEPTH):
                    v = -normalize(ray.direction)
                    r = Ray(hit.point, 2*(hit.normal.dot(v)
                                          * hit.normal)-v, ray.start+.0001)
                    hit_r = scene.intersect(r)
                    if hit_r == no_hit:
                        return result + scene.bg_color * hit.material.k_m
                    else:
                        recursive = vec(
                            shade(r, hit_r, scene, lights, depth+1))
                        return result + recursive * hit.material.k_m
        else:
            return result + scene.bg_color
    return result


def render_image(camera, scene, lights, nx, ny):
    """Render a ray traced image.

    Parameters:
      camera : Camera -- the camera defining the view
      scene : Scene -- the scene to be rendered
      lights : Lights -- the lights illuminating the scene
      nx, ny : int -- the dimensions of the rendered image
    Returns:
      (ny, nx, 3) float32 -- the RGB image
    """
    image = np.zeros((ny, nx, 3), np.float32)
    for x in range(nx):
        for y in range(ny):
            img_point = np.array([(x+0.5)/nx, (y+0.5)/ny])
            ray = camera.generate_ray(img_point)
            hit = scene.intersect(ray)
            if hit != no_hit:
                image[y, x] = shade(ray, hit, scene, lights)
            else:
                image[y, x] = scene.bg_color
    return image
