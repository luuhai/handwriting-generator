from .base_options import BaseOptions

class TestLineOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        self.parser.add_argument('--font', type=str, default='./data/fonts/simhei.ttf', help='the font path')
        self.parser.add_argument('--font_size', type=int, default=128, help="the font's character size")
        self.parser.add_argument('--offset', type=int, default=0, help="the font's character x and y offset")
        self.parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.isTrain = False
