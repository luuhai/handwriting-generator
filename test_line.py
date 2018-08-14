import os
import torch

from options.test_line_options import TestLineOptions
from data import CreateDataLoader
from data.base_dataset import get_transform
from models import create_model
from util.visualizer import save_image
from util import html
from PIL import Image, ImageFont, ImageDraw
from functools import reduce

def draw_single_char(ch, font, canvas_size=128, x_offset=26, y_offset=36):
    img = Image.new("L", (canvas_size, canvas_size), 255)
    draw = ImageDraw.Draw(img)
    draw.text((x_offset, y_offset), ch, 0, font=font)
    return img.convert('RGB')


def gen_line(text, opt):
    result_img_names = []

    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.display_id = -1  # no visdom display

    font = ImageFont.truetype(opt.font, size=opt.font_size)

    epochs = opt.which_epoch.split(',')
    for epoch in epochs:
        results = []
        inputs = []
        opt.which_epoch = epoch
        model = create_model(opt)
        model.setup(opt)

        for ch in text:
            img = draw_single_char(ch, font, x_offset=opt.offset, y_offset=opt.offset)

            transform = get_transform(opt)

            img = transform(img)

            if opt.input_nc == 1:  # RGB to gray
                tmp = img[0, ...] * 0.299 + img[1, ...] * 0.587 + img[2, ...] * 0.114
                img = tmp.unsqueeze(0)


            img = img.reshape([1, img.shape[0], img.shape[1], img.shape[2]])
            inputs.append(img)

            model.set_input_real_A(img)
            model.test_fake_B()
            results.append(model.fake_B)

        result = reduce((lambda x, y: torch.cat((x, y), -1)), results)
        input_img = reduce((lambda x, y: torch.cat((x, y), -1)), inputs)

        result_img_name = 'result_' + opt.name + '_' + str(epoch) + '_' + text +  '.png'
        input_img_name = 'input_' + opt.name + '_' + text + '.png'
        result_img_names.append(result_img_name)

        save_image(opt.results_dir, result, result_img_name, aspect_ratio=opt.aspect_ratio)
        save_image(opt.results_dir, input_img, input_img_name, aspect_ratio=opt.aspect_ratio)

    return {'input': input_img_name, 'result': result_img_names}

if __name__ == '__main__':
    text = input("Input text: ")

    opt = TestLineOptions().parse()
    gen_line(text, opt)
