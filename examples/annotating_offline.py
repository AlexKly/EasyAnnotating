import time
from easy_annotating import utils, annotator

TYPE_REPRESENTATION = 'words'
global CURRENT_TS
CURRENT_TS = 0


def display(tokens, timestamps):
    global CURRENT_TS
    if len(timestamps) > 0:
        for p in zip(tokens, timestamps):
            if p[1] >= CURRENT_TS:
                time.sleep(p[1] - CURRENT_TS)
            else:
                time.sleep(0.01)
            print(p[0], end='')
            CURRENT_TS = p[1]


if __name__ == '__main__':
    configs = utils.Configurations()
    ann = annotator.EasyAnnotating(configs=configs)

    outputs = ann.annotate(audio='test.wav', type_representation=TYPE_REPRESENTATION)

    for output in outputs[0]:
        display(tokens=output['content'], timestamps=output['ts'])
