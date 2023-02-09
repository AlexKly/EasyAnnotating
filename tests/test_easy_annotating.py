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
    def test_ea_a_u0(self):
        audio = utils.load_audio(path=dir_examples/'cosmos_720.mp4')
        print(ea.annotate(audio=audio, type_representation='words'))

