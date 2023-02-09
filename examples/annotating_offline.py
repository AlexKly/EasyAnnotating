import time
from easy_annotating import utils, annotator

TYPE_REPRESENTATION = 'particles'

if __name__ == '__main__':
    # Initialization objects:
    configs = utils.Configurations()
    ann = annotator.EasyAnnotating(configs=configs, verbose=1)

    # Process audio:
    audio = utils.load_audio(path='cosmos_720.mp4')
    outputs = ann.annotate(audio=audio, type_representation=TYPE_REPRESENTATION)

    # Display text according timestamps:
    for k in outputs[0].keys():
        utils.display(tokens=outputs[0][k]['tokens'], timestamps=outputs[0][k]['timestamps'])
