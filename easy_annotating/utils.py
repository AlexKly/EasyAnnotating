from pathlib import Path
import time, torch, librosa

PROJECT_NAME = 'EasyAnnotating'
PROJECT_SUBDIR = 'easy_annotating'
global CURRENT_TS
CURRENT_TS = 0


def load_audio(path):
    samples, _ = librosa.load(path=path, sr=Configurations().ia_sr)
    return torch.tensor(data=samples, dtype=torch.float32)


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


class Configurations:
    def __init__(
            self,
            name='small',
            device='cpu',
            download_root=None,
            in_memory=False,
            is_multilingual=True,
            task=None,
            do_classification=True,
            segment_len=30,
            use_subtracting=True,
            language='ru',
            type_representation='words',
            no_speech_threshold=0.2
    ):
        # Directories, files and paths:
        cur_path = str(Path().resolve())
        path_to_project = '/'.join(cur_path.split('/')[:cur_path.split('/').index(PROJECT_NAME) + 1])
        self.dir_cfgs = Path(path_to_project)/f'{PROJECT_SUBDIR}/configs'
        # Whisper configurations:
        self.whisper_name = name
        self.whisper_device = device
        self.download_root = download_root
        self.whisper_in_memory = in_memory
        self.whisper_is_multilingual = is_multilingual
        self.whisper_task = task
        # Parameters for audio classification:
        self.ac_do_classification = do_classification
        self.ac_segment_len = segment_len
        self.ac_use_subtracting = use_subtracting
        # Parameters for input audio:
        self.ia_sr = 16000
        self.ia_lang = language
        # Type of outputs representation:
        self.out_no_speech_threshold = no_speech_threshold
