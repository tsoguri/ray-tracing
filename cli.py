from sys import argv
from os import path
import argparse
import numpy as np
from PIL import Image

from ray import render_image
from utils import *

"""
Command line driver for rendering scenes.

When imported this module parses command line arguments, and then when
the render() function is called, it renders and image and writes it to 
the output file.
"""

default_outFile = path.splitext(argv[0])[0] + '.png'

parser = argparse.ArgumentParser(
    description='Render the scene and write it to an image file.',
    epilog='Renders the scene that is built in the scene file.')
parser.add_argument('--nx', type=int, default=256,
                    help='width of output image')
parser.add_argument('--ny', type=int, help='height of output image')
parser.add_argument('--white', type=float, default=1.0, help='white point')
parser.add_argument('--outFile', type=str,
                    default=default_outFile, help='name of output PNG image')
args = parser.parse_args()


def render(camera, scene, lights):

    ny = args.ny or int(np.round(args.nx / camera.aspect))

    img = render_image(camera, scene, lights, args.nx, ny)

    cam_img_ui8 = to_srgb8(img / args.white)
    Image.fromarray(cam_img_ui8[::-1, :, :], 'RGB').save(args.outFile)
