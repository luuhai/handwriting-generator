# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import

import argparse
import sys
import numpy as np
import scipy.misc
import os
import glob
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import json
import collections
import random

prev = 0

def draw_single_char(ch, font, canvas_size=128, x_offset=0, y_offset=0):
    img = Image.new("L", (canvas_size, canvas_size), 255)
    draw = ImageDraw.Draw(img)
    draw.text((x_offset, y_offset), ch, 0, font=font)
    return img


def resize_image(img):
    # pad to square
    pad_size = int(abs(img.shape[0]-img.shape[1]) / 2)
    if img.shape[0] < img.shape[1]:
        pad_dims = ((pad_size, pad_size), (0, 0))
    else:
        pad_dims = ((0, 0), (pad_size, pad_size))
    img = np.lib.pad(img, pad_dims, mode='constant', constant_values=255)
    # resize
    img = scipy.misc.imresize(img, (128, 128))

    assert img.shape == (128, 128)
    return img


def main(path, source_path, ratioA, ratioB, percentage, font_size, offset):
    global prev
    source_font = ImageFont.truetype(source_path, size=font_size)
    f = open(path, "rb")
    directory, name = os.path.split(path)
    random.seed(20171201)
    charlist = []
    bitmaplist = []
    sourcelist = []
    tmp = []
    filename = os.path.basename(path).split('.')[0]
    datafolder = os.path.join(os.path.normpath(directory + os.sep + os.pardir),
			      'datasets',
                              str.join('_', [name.split('.')[0], str(font_size), str(offset), str(ratioA)]))
    if not os.path.exists(datafolder):
        os.makedirs(datafolder)
    trainA_path = os.path.join(datafolder, 'trainA')
    trainB_path = os.path.join(datafolder, 'trainB')
    testA_path = os.path.join(datafolder, 'testA')
    testB_path = os.path.join(datafolder, 'testB')
    folders = [trainA_path,trainB_path, testA_path, testB_path]
    for folder in folders:
        if not os.path.exists(folder):
            os.mkdir(folder)
    while True:
        tmp = f.read(4)
        if len(tmp) is 0:
            break
        else:
            sample_size = np.fromstring(tmp, dtype=np.uint32).item()
            tag_code = np.fromstring(f.read(2), dtype=np.uint16).newbyteorder().item()
            width = np.fromstring(f.read(2), dtype=np.uint16).item()
            height = np.fromstring(f.read(2), dtype=np.uint16).item()
            bitmap = np.fromstring(f.read(width * height), dtype=np.uint8)
            bitmap = bitmap.reshape([height, width])
            bitmap = resize_image(bitmap)
            if (random.randrange(100) <= percentage):
                bitmaplist.append(bitmap)
                ch = bytearray.fromhex(str(hex(tag_code))[2:]).decode('gb2312')
                charlist.append(ch)
                source_img = draw_single_char(ch, font = source_font, x_offset=offset, y_offset=offset)
                sourcelist.append(source_img)

    print("Number of images: {}".format(len(sourcelist)))
    arr = np.arange(len(charlist))
    np.random.shuffle(arr)
    ntrainA = np.floor(float(ratioA) * len(charlist))
    ntrainB = np.floor(float(ratioB) * len(charlist))
    for i, x in enumerate(np.arange(len(arr))):
        ch = charlist[arr[x]]
        bitmap = bitmaplist[arr[x]]
        source_img = sourcelist[arr[x]]
        if arr[x]<=ntrainA and arr[x]<=ntrainB:
            scipy.misc.imsave(os.path.join(trainA_path, str(i+prev) + '.png'), bitmap)
            scipy.misc.imsave(os.path.join(trainB_path, str(i+prev) + '.png'), source_img)
        elif arr[x]>ntrainA and arr[x]<=ntrainB:
            scipy.misc.imsave(os.path.join(testA_path, str(i+prev) + '.png'), bitmap)
            scipy.misc.imsave(os.path.join(trainB_path, str(i+prev) + '.png'), source_img)
        elif arr[x]<=ntrainA and arr[x]>ntrainB:
            scipy.misc.imsave(os.path.join(trainA_path, str(i+prev) + '.png'), bitmap)
            scipy.misc.imsave(os.path.join(testB_path, str(i+prev) + '.png'), source_img)
        else:
            scipy.misc.imsave(os.path.join(testA_path, str(i+prev) + '.png'), bitmap)
            scipy.misc.imsave(os.path.join(testB_path, str(i+prev) + '.png'), source_img)
    prev += len(arr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocessing data")
    parser.add_argument('--source', dest='source', help="input file(s) to process")
    parser.add_argument('--font', dest='font', help="font to process")
    parser.add_argument('--ratioA', dest='ratioA', type=float, default='0.7', help='the split ratio of the training and test data')
    parser.add_argument('--ratioB', dest='ratioB', type=float, default='0.7', help='the split ratio of the training and test data')
    parser.add_argument('--percent', dest='percent', type=int, default='50', help='the wanted percentage of dataset')
    parser.add_argument('--fontSize', dest='fontSize', type=int, default='128', help='the wanted size of font character')
    parser.add_argument('--offset', dest='offset', type=int, default='0', help='the x and y offset of font character image')
    args = parser.parse_args()

    main(args.source, args.font, args.ratioA, args.ratioB, args.percent, args.fontSize, args.offset)

