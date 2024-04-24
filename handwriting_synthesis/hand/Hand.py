import logging
import os

import numpy as np

from handwriting_synthesis import drawing
from handwriting_synthesis.config import prediction_path, checkpoint_path, style_path
from handwriting_synthesis.hand._draw import _draw, _simulate_paragraph_word_counts, _draw_document
from handwriting_synthesis.rnn import RNN


class Hand(object):
    def __init__(self):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        self.nn = RNN(
            log_dir='logs',
            checkpoint_dir=checkpoint_path,
            prediction_dir=prediction_path,
            learning_rates=[.0001, .00005, .00002],
            batch_sizes=[32, 64, 64],
            patiences=[1500, 1000, 500],
            beta1_decays=[.9, .9, .9],
            validation_batch_size=32,
            optimizer='rms',
            num_training_steps=100000,
            warm_start_init_step=17900,
            regularization_constant=0.0,
            keep_prob=1.0,
            enable_parameter_averaging=False,
            min_steps_to_checkpoint=2000,
            log_interval=20,
            logging_level=logging.CRITICAL,
            grad_clip=10,
            lstm_size=400,
            output_mixture_components=20,
            attention_mixture_components=10
        )
        self.nn.restore()

    def write(self, filename, lines, biases=None, styles=None, stroke_colors=None, stroke_widths=None, text_align='center'):
        valid_char_set = set(drawing.alphabet)
        for line_num, line in enumerate(lines):
            if len(line) > 75:
                raise ValueError(
                    (
                        "Each line must be at most 75 characters. "
                        "Line {} contains {}"
                    ).format(line_num, len(line))
                )

            for char in line:
                if char not in valid_char_set:
                    raise ValueError(
                        (
                            "Invalid character {} detected in line {}. "
                            "Valid character set is {}"
                        ).format(char, line_num, valid_char_set)
                    )
                
        strokes = self._sample(lines, biases=biases, styles=styles)
        _draw(strokes, lines, filename, stroke_colors=stroke_colors, stroke_widths=stroke_widths, text_align=text_align)

    def write_document(self, filename, words, biases=None, styles=None, stroke_colors=None, stroke_widths=None):
        valid_char_set = set(drawing.alphabet)
        for word_num, word in enumerate(words):
            if len(word) > 75:
                raise ValueError(
                    (
                        "Each word must be at most 75 characters. "
                        "Word {} contains {}"
                    ).format(word_num, len(word))
                )

            for char in word:
                if char not in valid_char_set:
                    raise ValueError(
                        (
                            "Invalid character {} detected in word {}. "
                            "Valid character set is {}"
                        ).format(char, word_num, valid_char_set)
                    )
            
        strokes = self._sample(lines=words, biases=biases, styles=styles)
        word_counts = _simulate_paragraph_word_counts(strokes, words)

        lines = []
        line = ''
        
        # word_counts is list of int which says how many words in each line
        for word_count in word_counts:
            line = ''
            for _ in range(word_count):
                # if the first word in the line dont add space
                if len(line) == 0:
                    line += words.pop(0)
                else:
                    line += ' ' + words.pop(0)
            lines.append(line)

        biases = [0.35 for i in lines]
        styles = [5 for i in lines]
        # styles from 1 to 10 roll with lines
        # biases = [1 for i in lines]
        # styles = np.linspace(1, 10, len(lines)) # convert to int
        # styles = np.round(styles).astype(int)

        print(styles)

        strokes = self._sample(lines, biases=biases, styles=styles)
        _draw_document(strokes, lines, filename, stroke_colors=stroke_colors, stroke_widths=stroke_widths)

    def _sample(self, lines, biases=None, styles=None):
        num_samples = len(lines)
        max_tsteps = 40 * max([len(i) for i in lines])
        biases = biases if biases is not None else [0.5] * num_samples

        x_prime = np.zeros([num_samples, 1200, 3])
        x_prime_len = np.zeros([num_samples])
        chars = np.zeros([num_samples, 120])
        chars_len = np.zeros([num_samples])

        if styles is not None:
            for i, (cs, style) in enumerate(zip(lines, styles)):
                x_p = np.load(f"{style_path}/style-{style}-strokes.npy")
                c_p = np.load(f"{style_path}/style-{style}-chars.npy").tostring().decode('utf-8')

                c_p = str(c_p) + " " + cs
                c_p = drawing.encode_ascii(c_p)
                c_p = np.array(c_p)

                x_prime[i, :len(x_p), :] = x_p
                x_prime_len[i] = len(x_p)
                chars[i, :len(c_p)] = c_p
                chars_len[i] = len(c_p)

        else:
            for i in range(num_samples):
                encoded = drawing.encode_ascii(lines[i])
                chars[i, :len(encoded)] = encoded
                chars_len[i] = len(encoded)

        [samples] = self.nn.session.run(
            [self.nn.sampled_sequence],
            feed_dict={
                self.nn.prime: styles is not None,
                self.nn.x_prime: x_prime,
                self.nn.x_prime_len: x_prime_len,
                self.nn.num_samples: num_samples,
                self.nn.sample_tsteps: max_tsteps,
                self.nn.c: chars,
                self.nn.c_len: chars_len,
                self.nn.bias: biases
            }
        )
        samples = [sample[~np.all(sample == 0.0, axis=1)] for sample in samples]
        return samples
