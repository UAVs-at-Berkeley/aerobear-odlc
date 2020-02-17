import argparse
import cv2
import numpy as np
import math
import random

import generate_targets

# Conversion between orientation and heading angle, used by place_target
orientations = {'N': 0, 'NE': -45, 'E': -90, 'SE': -135, 'S': 180, 'SW': 135, 'W': 90, 'NW': 45}

# Given the target we want as well as some randomized information about its position, place the target onto the mock
# background.
def place_target(target_img, orientation, position, scale):
    # Target preprocessing

    # Since add_noise requires an HSV conversion that doesn't preserve alpha channel, we have to save it first
    alpha_channel = target_img[:, :, 3]
    target_img = add_noise(target_img)
    if args.debug_mode:
        cv2.imshow('Raw target', target_img)
        cv2.waitKey(0)
    transformed_img = affine_transform(target_img, orientations[orientation], scale)
    transformed_alpha = affine_transform(alpha_channel, orientations[orientation], scale)

    if args.debug_mode:
        cv2.imshow('Raw target', transformed_img)
        cv2.waitKey(0)

    # Target placing
    field = alpha_blend(transformed_img, transformed_alpha, position)


# Compute the affine transformed image from a given rotation and scale
def affine_transform(img, rotation, scale):
    rotation_matrix = cv2.getRotationMatrix2D((50, 50), rotation, 1)
    scale_matrix = cv2.getRotationMatrix2D((0, 0), 0, scale)

    new_dsize = (round(img.shape[0] * scale), round(img.shape[1] * scale))
    transformed_img = cv2.warpAffine(img, rotation_matrix, img.shape[:2])
    transformed_img = cv2.warpAffine(transformed_img, scale_matrix, new_dsize)
    return transformed_img


# Reduce image quality and add lighting effects and noise to give targets the feeling of having been photographed
def add_noise(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            # Desaturate
            if img[row, col, 2] == 0:
                # Desaturating black works differently than other colors; instead of lowering S we increase V
                img[row, col, 2] = 255 * (1 - args.lighting_constant)
            img[row, col, 1] *= args.lighting_constant
            # Add a noise value to each of a pixel's saturation, and value
            for i in [1, 2]:
                value = img[row, col, i]
                img[row, col, i] += min(255 - value, max(-int(value), random.randint(-args.noise_intensity, args.noise_intensity)))
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    return img


# Superimpose the modified target image onto the background at the specified offset.
def alpha_blend(img, alpha_channel, offset):

    if args.debug_mode:
        cv2.imshow('Field', field)
        cv2.waitKey(0)

    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            if alpha_channel[row, col] > 0:
                field[row + offset[0], col + offset[1], :3] = img[row, col]

    if args.debug_mode:
        cv2.imshow('Field', field)
        cv2.waitKey(0)

    return field


# Blur the edges of targets by combining a gaussian blur of the field with a edge detection mask
def blur_edges(field):
    field_blur = cv2.GaussianBlur(field, (5, 5), 0)
    field_mask = cv2.cvtColor(cv2.Canny(field, 200, 600), cv2.COLOR_GRAY2BGR)
    blurred_edges = np.where(field_mask==0, field_blur, field)
    return blurred_edges


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--background', type=str, help='The name of the image file to use as a background.')
    parser.add_argument('-t', '--num_targets', type=int, default=5, help='The number of targets to place in this mock field.')
    parser.add_argument('--seed', type=int, default=-1, help='Seed used for random target generation. Random by default.')
    parser.add_argument('-s', '--scale_target', type=float, default=0.35, help='The average scale factor for each target.')
    parser.add_argument('-sv', '--scale_variance', type=float, default=0.1, help='The multiplication factor by which the scale of a single target can vary. Set to 0 for a constant scale.')
    parser.add_argument('-l', '--lighting_constant', type=float, default=0.5, help='The amount to scale each pixel saturation by, simulating natural lighting.')
    parser.add_argument('-n', '--noise_intensity', type=int, default=10, help='The maximum increase or decrease applied to HSV values in random noise generation.')
    parser.add_argument('-c', '--clip_maximum', type=float, default=0, help='The greatest proportion of a target\'s width/height that may be out of bounds. Zero by default, but set higher to allow clipping.')
    parser.add_argument('--debug-mode', type=bool, nargs='?', const=True, default=False, help='Set to true to display intermediate images.')
    args = parser.parse_args()

    field = cv2.imread('./backgrounds/{0}.jpg'.format(args.background))

    if args.seed == -1:
        seed = random.randint(0, 100)
    else:
        seed = args.seed
    random.seed(seed) # Setting the seed insures replicability of results

    for i in range(args.num_targets):
        # Randomize target characteristics
        shape = random.choice(list(generate_targets.shapes.keys()))
        alphanum = random.choice(generate_targets.alphanumerics)
        shape_color, alphanum_color = random.sample(list(generate_targets.colors.keys()), 2)

        if args.debug_mode:
            print(shape, shape_color, alphanum, alphanum_color)

        # Randomize transformations
        orientation = random.choice(list(orientations.keys()))
        scale = random.uniform((1-args.scale_variance)*args.scale_target, (1+args.scale_variance)*args.scale_target)
        pos = (round(random.uniform(-args.clip_maximum*100*scale, field.shape[0]-(1-args.clip_maximum)*100*scale)),
               round(random.uniform(-args.clip_maximum*100*scale, field.shape[1]-(1-args.clip_maximum)*100*scale)))

        raw_target = generate_targets.target(shape, shape_color, alphanum, alphanum_color)
        if args.debug_mode:
            cv2.imshow('Raw target', raw_target)
            cv2.waitKey(0)
        place_target(raw_target, orientation, pos, scale)
    # field = blur_edges(field)
    # place_target(generate_targets.target('circle', 'red', 'V', 'brown'), 'SE', (190, 300), 0.25)
    
    cv2.imwrite('./{0}_{1}.jpg'.format(args.background, seed), field)
