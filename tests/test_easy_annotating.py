import sys, logging, unittest
from pathlib import Path
from easy_annotating import utils, annotator

format = '%(asctime)s,%(msecs)03d %(levelname)-4s [%(filename)s:%(lineno)d] %(message)s'
logging.basicConfig(level=logging.INFO, datefmt='%H:%M:%S', format=format)
if sys.version_info.major == 3 and sys.version_info.minor >= 8:
    logging.basicConfig(level=logging.INFO, datefmt='%H:%M:%S', format=format, force=True)
logging.info('Logger is on -->')

cur_dir = Path().resolve()
dir_examples = cur_dir.parent/'examples'
c = utils.Configurations()
ea = annotator.EasyAnnotating(configs=c, verbose=1)


class TestEasyAnnotating(unittest.TestCase):
    def test_ea_a_u0(self): print(ea.annotate(audio=dir_examples/'test.wav', type_representation='segments'))    # OK
    def test_ea_a_u1(self): print(ea.annotate(audio=dir_examples/'test.wav', type_representation='words'))       # OK
    def test_ea_a_u2(self): print(ea.annotate(audio=dir_examples/'test.wav', type_representation='particles'))   # OK
