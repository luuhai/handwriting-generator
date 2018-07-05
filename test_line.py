import os
from options.test_options import TestOptions
from data import CreateDataLoader
from data.base_dataset import get_transform
from models import create_model
from util.visualizer import save_images
from util import html
from PIL import Image, ImageFont, ImageDraw
from IPython import embed

def draw_single_char(ch, font, canvas_size=128, x_offset=0, y_offset=0):
    img = Image.new("L", (canvas_size, canvas_size), 255)
    draw = ImageDraw.Draw(img)
    draw.text((x_offset, y_offset), ch, 0, font=font)
    return img


if __name__ == '__main__':
    ch = 'ひ'
    font = ImageFont.truetype("/home/hailt/code/github/HCCG-CycleGAN/fonts/font/TakaoGothic.ttf", size=128)
    img = draw_single_char(ch, font)

    opt = TestOptions().parse()
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.display_id = -1  # no visdom display

    transform = get_transform(opt)
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    model = create_model(opt)
    model.setup(opt)

    # create website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

    img = transform(img)
    model.set_input_real_A(img)
    model.test_fake_B()

    if self.opt.which_direction == 'BtoA':
        input_nc = self.opt.output_nc
        output_nc = self.opt.input_nc
    else:
        input_nc = self.opt.input_nc
        output_nc = self.opt.output_nc

    if input_nc == 1:  # RGB to gray
        tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
        A = tmp.unsqueeze(0)

    if output_nc == 1:  # RGB to gray
        tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
        B = tmp.unsqueeze(0)
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
