import numpy as np
import torch, timeit, logging, whisper, whisper.tokenizer, stable_whisper


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
        self.do_classification = configs.ac_do_classification
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

    def split_audio(self, samples):
        if self.verbose > -1:
            t = timeit.default_timer()
            logging.info(f'Start framing input audio -->')
        samples_per_segment = self.sr * self.segment_len
        if isinstance(samples, np.ndarray):
            samples = torch.tensor(data=samples, dtype=torch.float32)
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

    def apply_model(self, audio):
        if self.verbose > -1:
            t = timeit.default_timer()
            logging.info(f'Start performing ASR Whisper model -->')
        outputs = self.model.transcribe(audio=audio, language=self.lang)
        if self.verbose > -1: logging.info(f'ASR Whisper processing time --> {round(timeit.default_timer() - t, 3)}')

        return outputs

    def restore_data(self, particles, type_r):
        restored_data = dict()
        for segment in particles:
            first_run = True
            word_tmp = ''
            restored_data[f'segment_{segment["id"]}'] = {
                'tokens': list(),
                'timestamps': list(),
                'no_speech_prob': segment['no_speech_prob'],
                'is_speech': True if segment['no_speech_prob'] < self.no_speech_threshold else False,
            }
            for particle in segment['whole_word_timestamps']:
                if type_r == 'words':
                    if ' ' in particle['word']:
                        restored_data[f'segment_{segment["id"]}']['timestamps'] += [particle['timestamp']]
                        if first_run:
                            first_run = False
                        else:
                            restored_data[f'segment_{segment["id"]}']['tokens'] += [word_tmp]
                            word_tmp = ''
                    word_tmp += particle['word']
                elif type_r == 'particles':
                    restored_data[f'segment_{segment["id"]}']['tokens'] += [particle['word']]
                    restored_data[f'segment_{segment["id"]}']['timestamps'] += [particle['timestamp']]
                else:
                    logging.info(f'Chosen incorrect output data representation type --> {type_r}')
                    break
            if type_r == 'words':
                if len(word_tmp) > 0:
                    restored_data[f'segment_{segment["id"]}']['tokens'] += [word_tmp]

        return restored_data

    def form_outputs(self, outputs, type_r):
        if self.verbose > -1:
            t = timeit.default_timer()
            logging.info(f'Start compiling output from Whisper -->')
        particles = stable_whisper.stabilize_timestamps(segments=outputs, top_focus=True)
        restored_data = self.restore_data(particles=particles, type_r=type_r)
        if self.verbose > -1: logging.info(f'Forming outputs processing time --> {round(timeit.default_timer() - t, 3)}')

        return restored_data

    @torch.no_grad()
    def calc_audio_features(self, samples):
        if samples is None:
            segment = torch.zeros((whisper.audio.N_MELS, whisper.audio.N_FRAMES), dtype=torch.float32).to(self.model.device)
        else:
            mel = whisper.log_mel_spectrogram(audio=samples)
            segment = whisper.pad_or_trim(array=mel, length=whisper.audio.N_FRAMES).to(self.model.device)

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

    def classify_frames(self, frames):
        if self.verbose > -1:
            t1 = timeit.default_timer()
            logging.info(f'Start audio classification -->')
        if self.use_subtracting:
            self.calc_internal_lm_average_logprobs()
        classes_per_frames = {'audio_class': list(), 'timestamps': list()}
        for i in range(frames.shape[0]):
            if self.verbose == 1: t2 = timeit.default_timer()
            audio_features = self.calc_audio_features(samples=frames[i])
            average_logprobs = self.calc_average_logprobs(audio_features=audio_features)
            if self.internal_lm_average_logprobs is not None:
                average_logprobs -= self.internal_lm_average_logprobs
            sorted_indices = sorted(range(len(self.classes)), key=lambda ind: average_logprobs[ind], reverse=True)
            classes_per_frames['audio_class'] += [[self.classes[ind] for ind in sorted_indices][0]]
            classes_per_frames['timestamps'] += [self.segment_len * i]
            if self.verbose == 1:
                logging.info(f'Average log probabilities fr each class per frames: '
                             f'[{self.segment_len * i} --> {self.segment_len * (i + 1)}]')
                for ind in sorted_indices:
                    logging.info(f'{self.classes[ind]}: {average_logprobs[ind].round(decimals=3)}')
                logging.info(f'Classification audio processing time on frame --> '
                             f'{round(timeit.default_timer() - t2, 3)} seconds')
                logging.info(f'======================================================')
        if self.verbose > -1: logging.info(f'Audio classification processing time --> '
                                           f'{round(timeit.default_timer() - t1, 3)} seconds')

        return classes_per_frames

    def annotate(self, audio, type_representation):
        predicted_classes = None
        # Perform Whisper ASR model to get text and timestamps:
        outputs = self.apply_model(audio=audio)
        # Form output from Whisper according output data representation type:
        restored_data = self.form_outputs(outputs=outputs, type_r=type_representation)
        # Classify frames:
        if self.do_classification:
            frames = self.split_audio(samples=audio)
            predicted_classes = self.classify_frames(frames=frames)

        return restored_data, predicted_classes
