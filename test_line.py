import os
import torch

from options.test_options import TestOptions
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


if __name__ == '__main__':
    address = '〒100-8994 東京都中央区八重洲1-5-3'
    font = ImageFont.truetype("TakaoGothic.ttf", size=68)
    results = []

    opt = TestOptions().parse()
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.display_id = -1  # no visdom display

    model = create_model(opt)
    model.setup(opt)
    for ch in address:
        img = draw_single_char(ch, font)

        transform = get_transform(opt)

        # create website
        web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
        webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

        img = transform(img)

        if opt.which_direction == 'BtoA':
            input_nc = opt.output_nc
            output_nc = opt.input_nc
        else:
            input_nc = opt.input_nc
            output_nc = opt.output_nc

        if input_nc == 1:  # RGB to gray
            tmp = img[0, ...] * 0.299 + img[1, ...] * 0.587 + img[2, ...] * 0.114
            img = tmp.unsqueeze(0)


        model.set_input_real_A(img.reshape([1, img.shape[0], img.shape[1], img.shape[2]]))
        model.test_fake_B()
        results.append(model.fake_B)

    result = reduce((lambda x, y: torch.cat((x, y), -1)), results)

    save_image(webpage, result, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
    # test
    #for i, data in enumerate(dataset):
    #    if i >= opt.how_many:
    #        break
    #    model.set_input(data)
    #    model.test()
    #    visuals = model.get_current_visuals()
    #    img_path = model.get_image_paths()
    #    if i % 5 == 0:
    #        print('processing (%04d)-th image... %s' % (i, img_path))
    #    save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)

    #webpage.save()
