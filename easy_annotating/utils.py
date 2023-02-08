from pathlib import Path

PROJECT_NAME = 'EasyAnnotating'


class Configurations:
    def __init__(
            self,
            name='small',
            device='cpu',
            download_root=None,
            in_memory=False,
            is_multilingual=True,
            task=None,
            segment_len=5,
            use_subtracting=True,
            language=None,
            type_representation='words',
            no_speech_threshold=0.2
    ):
        # Directories, files and paths:
        cur_path = str(Path().resolve())
        path_to_project = '/'.join(cur_path.split('/')[:cur_path.split('/').index(PROJECT_NAME) + 1])
        self.dir_cfgs = Path(path_to_project)/'easy_annotating/configs'
        # Whisper configurations:
        self.whisper_name = name
        self.whisper_device = device
        self.download_root = download_root
        self.whisper_in_memory = in_memory
        self.whisper_is_multilingual = is_multilingual
        self.whisper_task = task
        # Parameters for audio classification:
        self.ac_segment_len = segment_len
        self.ac_use_subtracting = use_subtracting
        # Parameters for input audio:
        self.ia_sr = 16000
        self.ia_lang = language
        # Type of outputs representation:
        self.out_no_speech_threshold = no_speech_threshold
