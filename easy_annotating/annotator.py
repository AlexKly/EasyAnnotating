import numpy as np
import torch, timeit, pathlib, logging, librosa, whisper, whisper.tokenizer, stable_whisper


class EasyAnnotating:
    def __init__(self, configs, dir_cfgs=None, f_classes='class_names.txt', verbose=-1):
        # Directories and files:
        self.dir_cfgs = configs.dir_cfgs if dir_cfgs is None else dir_cfgs
        self.path_classes = self.dir_cfgs/f_classes
        # Common parameters:
        self.verbose = verbose
        # Audio data parameters:
        self.sr = configs.ia_sr
        self.lang = configs.ia_lang
        # Whisper initialization:
        self.model = stable_whisper.load_model(
            name=configs.whisper_name,
            device=configs.whisper_device,
            download_root=configs.download_root,
            in_memory=configs.whisper_in_memory
        )
        self.tokenizer = whisper.tokenizer.get_tokenizer(
            multilingual=configs.whisper_is_multilingual,
            task=configs.whisper_task,
            language=configs.ia_lang
        )
        # Audio Classification parameters:
        self.classes = None
        self.load_classes()
        self.internal_lm_average_logprobs = None
        self.segment_len = configs.ac_segment_len
        self.use_subtracting = configs.ac_use_subtracting
        # Processing data parameters:
        self.no_speech_threshold = configs.out_no_speech_threshold

    def load_classes(self):
        with self.path_classes.open('r') as reader:
            self.classes = [line.strip() for line in reader]

    def load_audio(self, audio):
        if isinstance(audio, str) or isinstance(audio, pathlib.Path):
            samples, _ = librosa.load(path=audio, sr=self.sr)
            return torch.tensor(data=samples, dtype=torch.float32)
        elif isinstance(audio, np.ndarray):
            return torch.tensor(data=audio, dtype=torch.float32)
        elif isinstance(audio, torch.Tensor):
            return audio
        else:
            logging.info(f'Incorrect input format for audio --> [str, Path, numpy array or torch Tensor]')
            return None

    def split_audio(self, samples):
        if self.verbose > -1: t = timeit.default_timer()
        samples_per_segment = self.sr * self.segment_len
        frames = torch.zeros((int(np.ceil(samples.shape[0] / samples_per_segment)), int(samples_per_segment)))
        start_ind, end_ind = 0, int(samples_per_segment)
        for i in range(frames.shape[0]):
            if samples[start_ind:].shape[0] >= int(samples_per_segment):
                frames[i] = samples[start_ind:end_ind]
            else:
                frames[i][:samples[start_ind:].shape[0]] = samples[start_ind:]
            start_ind = end_ind
            end_ind += int(samples_per_segment)

        if self.verbose > -1: logging.info(f'Framing processing time --> {round(timeit.default_timer() - t, 3)}')

        return frames

    def combine_outputs(self, outputs, type_r):
        if self.verbose > -1: t = timeit.default_timer()
        if self.verbose == 1:
            logging.info(f'Start to combine outputs --> ')
            logging.info(f'Chosen representation type: {type_r}')
        results = list()
        for segment in outputs['segments']:
            if type_r == 'segments':
                result = {
                    'content': [segment['text']],
                    'ts': [segment['start']],
                    'no_speech_prob': segment['no_speech_prob'],
                    'is_speech': True if segment['no_speech_prob'] < self.no_speech_threshold else False
                }
            elif type_r == 'words':
                result = {
                    'content': list(),
                    'ts': list(),
                    'no_speech_prob': segment['no_speech_prob'],
                    'is_speech': True if segment['no_speech_prob'] < self.no_speech_threshold else False
                }
                first_run = True
                word_tmp = ''
                for particle in segment['unstable_word_timestamps']:
                    if ' ' in particle['word']:
                        result['ts'] += [particle['timestamps'][0]]
                        if first_run:
                            first_run = False
                        else:
                            result['content'] += [word_tmp]
                            word_tmp = ''
                    word_tmp += particle['word']
                if len(result['content']) < len(result['ts']):
                    result['content'] += [word_tmp]
            elif type_r == 'particles':
                result = {
                    'content': list(),
                    'ts': list(),
                    'no_speech_prob': segment['no_speech_prob'],
                    'is_speech': True if segment['no_speech_prob'] < self.no_speech_threshold else False
                }
                for particle in segment['unstable_word_timestamps']:
                    result['content'] += [particle['word']]
                    result['ts'] += [particle['timestamps'][0]]
            else:
                logging.info(f'Incorrect chosen representation type.')
                return None
            results += [result]

            if self.verbose > -1: logging.info(f'Processing outputs time --> {round(timeit.default_timer() - t, 3)}')

        return results

    @torch.no_grad()
    def calc_audio_features(self, samples):
        if samples is None:
            segment = torch.zeros(
                (whisper.audio.N_MELS, whisper.audio.N_FRAMES),
                dtype=torch.float32
            ).to(self.model.device)
        else:
            padded_samples = whisper.pad_or_trim(array=samples)
            segment = whisper.log_mel_spectrogram(audio=padded_samples).to(self.model.device)

        return self.model.embed_audio(mel=segment.unsqueeze(0))

    @torch.no_grad()
    def calc_average_logprobs(self, audio_features):
        initial_tokens = (
            torch.tensor(self.tokenizer.sot_sequence_including_notimestamps).unsqueeze(0).to(self.model.device)
        )
        eot_token = torch.tensor([self.tokenizer.eot]).unsqueeze(0).to(self.model.device)

        average_logprobs = torch.zeros(len(self.classes))
        for i, class_name in enumerate(self.classes):
            class_name_tokens = (
                torch.tensor(self.tokenizer.encode(" " + class_name)).unsqueeze(0).to(self.model.device)
            )
            input_tokens = torch.cat([initial_tokens, class_name_tokens, eot_token], dim=1)
            logits = self.model.logits(input_tokens, audio_features)  # (1, T, V)
            logprobs = torch.nn.functional.log_softmax(logits, dim=-1).squeeze(0)  # (T, V)
            logprobs = logprobs[len(self.tokenizer.sot_sequence_including_notimestamps) - 1: -1]  # (T', V)
            logprobs = torch.gather(logprobs, dim=-1, index=class_name_tokens.view(-1, 1))  # (T', 1)
            average_logprob = logprobs.mean().item()
            average_logprobs[i] = average_logprob

        return average_logprobs

    def calc_internal_lm_average_logprobs(self):
        audio_features_from_empty_input = self.calc_audio_features(samples=None)
        self.internal_lm_average_logprobs = self.calc_average_logprobs(audio_features=audio_features_from_empty_input)

    def classify(self, frames):
        if self.verbose > -1: t1 = timeit.default_timer()
        if self.use_subtracting:
            self.calc_internal_lm_average_logprobs()
        classes_per_frames = {'audio_class': list(), 'start_ts': list(), 'end_ts': list()}
        for i in range(frames.shape[0]):
            if self.verbose == 1: t2 = timeit.default_timer()
            audio_features = self.calc_audio_features(samples=frames[i])
            average_logprobs = self.calc_average_logprobs(audio_features=audio_features)
            if self.internal_lm_average_logprobs is not None:
                average_logprobs -= self.internal_lm_average_logprobs
            sorted_indices = sorted(range(len(self.classes)), key=lambda ind: average_logprobs[ind], reverse=True)
            classes_per_frames['audio_class'] += [[self.classes[ind] for ind in sorted_indices][0]]
            classes_per_frames['start_ts'] += [self.segment_len * i]
            classes_per_frames['end_ts'] += [self.segment_len * (i + 1)]
            if self.verbose == 1:
                logging.info(f'Average log probabilities fr each class per frames: '
                             f'[{self.segment_len * i} --> {self.segment_len * (i + 1)}]')
                for ind in sorted_indices:
                    logging.info(f'{self.classes[ind]}: {average_logprobs[ind].round(decimals=3)}')
                logging.info(f'Classification audio processing time on frame --> '
                             f'{round(timeit.default_timer() - t2, 3)} seconds')
                logging.info(f'======================================================')
        if self.verbose > -1: logging.info(f'Classification audio processing time --> '
                                           f'{round(timeit.default_timer() - t1, 3)} seconds')

        return classes_per_frames

    def annotate(self, audio, type_representation='segments'):
        # Load samples from file and wrap it to tensor:
        samples = self.load_audio(audio=audio)
        frames = self.split_audio(samples=samples)
        # Perform transcription:
        if self.verbose > -1: t = timeit.default_timer()
        outputs = self.model.transcribe(audio=samples)
        if self.verbose > -1: logging.info(f'Whisper processing time --> {round(timeit.default_timer() - t, 3)}')
        # Combine outputs:
        asr_results = self.combine_outputs(outputs=outputs, type_r=type_representation)
        # Perform Audio Classification:
        cls_results = self.classify(frames=frames)

        return asr_results, cls_results
