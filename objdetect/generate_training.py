import argparse
import cv2
import numpy as np
import random
import os

import generate_targets

# Conversion between orientation and heading angle, used by place_target
orientations = {'N': 0, 'NE': -45, 'E': -90, 'SE': -135, 'S': 180, 'SW': 135, 'W': 90, 'NW': 45}

# Given the target we want as well as some randomized information about its position, place the target onto the mock
# background.
def place_target(target_img, orientation, position, scale, field, args):
    global bbox
    # Target preprocessing
    alpha_channel = target_img[:, :, 3]  # Since add_noise requites an HSV conversion that doesn't preserve alpha channel, we have to save it first
    target_img = add_noise(target_img, args)
    transformed_img = affine_transform(target_img, orientations[orientation], scale)
    transformed_alpha = affine_transform(alpha_channel, orientations[orientation], scale)

    # Target placing
    alpha_blend(transformed_img, transformed_alpha, position, field)
    y, x = position
    return (x, y, x+int(100*scale), y+int(100*scale))

def darknetify(bbox, imshape):
    """Compute the darknet descriptior for a bounding box."""
    xmin, ymin, xmax, ymax = bbox
    imheight = imshape[0]
    imwidth = imshape[1]
    x = (xmin + xmax)/2 / imwidth
    y = (ymin + ymax)/2 / imheight
    w = (xmax - xmin) / imwidth
    h = (ymax - ymin) / imheight
    clsid = 0 # We're only using 1 class for targets;
              #a separate nn will classify their type
    return ' '.join(str(x) for x in (clsid, x, y, w, h))


# Compute the affine transformed image from a given rotation and scale
def affine_transform(img, rotation, scale):
    rotation_matrix = cv2.getRotationMatrix2D((50, 50), rotation, 1)
    scale_matrix = cv2.getRotationMatrix2D((0, 0), 0, scale)

    new_dsize = (round(img.shape[0] * scale), round(img.shape[1] * scale))
    transformed_img = cv2.warpAffine(img, rotation_matrix, img.shape[:2])
    transformed_img = cv2.warpAffine(transformed_img, scale_matrix, new_dsize)
    return transformed_img


# Reduce image quality and add lighting effects and noise to give targets the feeling of having been photographed
def add_noise(img, args):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            # Desaturate
            img[row, col, 1] *= args.lighting_constant
            # Add a noise value to each of a pixel's saturation, and value
            for i in [1, 2]:
                value = img[row, col, i]
                img[row, col, i] += min(255 - value, max(-int(value), random.randint(-args.noise_intensity, args.noise_intensity)))
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    return img


# Superimpose the modified target image onto the background at the specified offset.
def alpha_blend(img, alpha_channel, offset, field):
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            if alpha_channel[row, col] > 0:
                field[row + offset[0], col + offset[1], :3] = img[row, col]
    return field


# Blur the edges of targets by combining a gaussian blur of the field with a edge detection mask
def blur_edges(field):
    field_blur = cv2.GaussianBlur(field, (5, 5), 0)
    field_mask = cv2.cvtColor(cv2.Canny(field, 200, 600), cv2.COLOR_GRAY2BGR)
    blurred_edges = np.where(field_mask==0, field_blur, field)
    return blurred_edges


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--num_targets', type=int, default=5, help='The number of targets to place in this mock field.')
    parser.add_argument('-s', '--scale_target', type=float, default=0.25, help='The average scale factor for each target.')
    parser.add_argument('-sv', '--scale_variance', type=float, default=0.1, help='The multiplication factor by which the scale of a single target can vary. Set to 0 for a constant scale.')
    parser.add_argument('-l', '--lighting_constant', type=float, default=0.5, help='The amount to scale each pixel saturation by, simulating natural lighting.')
    parser.add_argument('-n', '--noise_intensity', type=int, default=10, help='The maximum increase or decrease applied to HSV values in random noise generation.')
    parser.add_argument('-c', '--clip_maximum', type=float, default=0, help='The greatest proportion of a target\'s width/height that may be out of bounds. Zero by default, but set higher to allow clipping.')
    parser.add_argument('-N', '--num', type=int, default=100, help='Number of training images to generate.')
    parser.add_argument('-d', '--dest', type=str, default='training', help='Directory to output training images.')
    args = parser.parse_args()

    os.makedirs(args.dest + '/images', exist_ok=True)
    os.makedirs(args.dest + '/labels', exist_ok=True)

    allfields = [cv2.imread('fields/' + f) for f in os.listdir('fields')]
    trainfields, valfields = allfields[:-1], allfields[-1:]

    for seed in range(args.num):
        random.seed(seed) # Setting the seed insures replicability of results

        if seed < int(0.8*args.num):
            field = np.copy(random.choice(trainfields))
        else:
            field = np.copy(random.choice(valfields))
        boxes = []
        for i in range(args.num_targets):
            # Randomize one target
            shape = random.choice(list(generate_targets.shapes.keys()))
            alphanum = random.choice(generate_targets.alphanumerics)
            shape_color, alphanum_color = random.sample(list(generate_targets.colors.keys()), 2)

            orientation = random.choice(list(orientations.keys()))
            scale = random.uniform((1-args.scale_variance)*args.scale_target, (1+args.scale_variance)*args.scale_target)
            pos = (round(random.uniform(-args.clip_maximum*100*scale, field.shape[0]-(1-args.clip_maximum)*100*scale)),
		   round(random.uniform(-args.clip_maximum*100*scale, field.shape[1]-(1-args.clip_maximum)*100*scale)))
            bbox = place_target(generate_targets.target(shape, shape_color, alphanum, alphanum_color), orientation, pos, scale, field, args)
            # cv2.rectangle(field, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 2555), 2)

            boxes.append(darknetify(bbox, field.shape))
        field = blur_edges(field)
        # place_target(generate_targets.target('circle', 'red', 'V', 'brown'), 'SE', (190, 300), 0.25)

        cv2.imwrite(args.dest + '/images/field_{}.png'.format(seed), field)
        with open(args.dest + '/labels/field_{}.txt'.format(seed), 'w') as f:
            for box in boxes:
                print(box, file=f)

        if seed % 4 == 0:
            print('\r[' + ('#' * int(seed/4)) + ('.' * int(25-seed/4)) +']', end='')
    print('\r[' + '#'*25 + ']')
