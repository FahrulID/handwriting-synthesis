import numpy as np
import svgwrite
import random

from handwriting_synthesis import drawing

def _draw(strokes, lines, filename, stroke_colors=None, stroke_widths=None, text_align='center'):
    text_align = text_align.lower()
    text_align_options = ['left', 'center', 'right']

    stroke_colors = stroke_colors or ['black'] * len(lines)
    stroke_widths = stroke_widths or [2] * len(lines)

    line_height = 60
    view_width = 1000
    view_height = line_height * (len(strokes) + 1)

    A4_width = 210
    A4_height = 297

    A4_margin_top = 30
    A4_margin_left = 40
    A4_margin_right = 30
    A4_margin_bottom = 30

    view_box = '{} {} {} {}'.format(
        A4_margin_left,
        A4_margin_top,
        # A4_width - A4_margin_right,
        # A4_height - A4_margin_bottom
        A4_width*2,
        A4_height*2
    )

    size_width = '{}mm'.format(A4_width)
    size_height = '{}mm'.format(A4_height)
    size = (size_width, size_height)

    # dwg = svgwrite.Drawing(filename=filename, size=size, viewBox=view_box)
    dwg = svgwrite.Drawing(filename=filename, size=size)
    # dwg.viewbox(width=view_width, height=view_height)
    # dwg.add(dwg.rect(insert=(0, 0), size=(view_width, view_height), fill='white'))

    initial_coord = np.array([0, -(3 * line_height / 4)])
    for offsets, line, color, width in zip(strokes, lines, stroke_colors, stroke_widths):

        if not line:
            initial_coord[1] -= line_height
            continue

        offsets[:, :2] *= 1.5
        strokes = drawing.offsets_to_coords(offsets)
        strokes = drawing.denoise(strokes)
        strokes[:, :2] = drawing.align(strokes[:, :2])
        # scaling here
        strokes[:, :2] *= 0.8

        strokes[:, 1] *= -1
        strokes[:, :2] -= strokes[:, :2].min() + initial_coord
        # check if text align is not in options
        if text_align not in text_align_options:
            raise ValueError(
                "Invalid text align option. Options are {}".format(text_align_options)
            )
        
        if text_align == 'center':
            strokes[:, 0] += (view_width - strokes[:, 0].max()) / 2
        elif text_align == 'right':
            strokes[:, 0] += view_width - strokes[:, 0].max()
        elif text_align == 'left':
            strokes[:, 0] += A4_margin_left
        elif text_align == 'justify':
            strokes[:, 0] += 0

        prev_eos = 1.0
        p = "M{},{} ".format(0, 0)
        for x, y, eos in zip(*strokes.T):
            p += '{}{},{} '.format('M' if prev_eos == 1.0 else 'L', x, y)
            prev_eos = eos
        path = svgwrite.path.Path(p)
        path = path.stroke(color=color, width=width, linecap='round').fill("none")
        dwg.add(path)

        initial_coord[1] -= line_height

    dwg.save()

def _draw_document(strokes, words, filename, stroke_colors=None, stroke_widths=None):
    A4_width = 210
    A4_height = 297
    size_width = '{}mm'.format(A4_width)
    size_height = '{}mm'.format(A4_height)
    size = (size_width, size_height)

    stroke_colors = stroke_colors or ['blue'] * len(words)
    stroke_widths = stroke_widths or [2] * len(words)

    dwg = svgwrite.Drawing(filename=filename, size=size)
    
    line_height = 60
    max_x = 700
    initial_coord = np.array([0, -(3 * line_height / 4)])
    for offset, word, color, width in zip(strokes, words, stroke_colors, stroke_widths):
        print(word)

        if not word:
            initial_coord[1] -= line_height
            continue
        
        offset[:, :2] *= 1.5
        strokes = drawing.offsets_to_coords(offset)
        strokes = drawing.denoise(strokes)
        # skewing
        strokes = drawing.skew(strokes, 20)
        strokes[:, :2] = drawing.align(strokes[:, :2])
        # scaling here
        strokes[:, :2] *= 0.8

        strokes[:, 1] *= -1
        strokes[:, :2] -= strokes[:, :2].min() + initial_coord
        strokes[:, 0] += 0

        stroke_size_x = strokes[:, 0].max()
        random_multipler_offset = random.uniform(-0.02, 0.02)
        multiplier = (max_x / stroke_size_x) + random_multipler_offset
        strokes = drawing.stretch(strokes, multiplier, 1)
        prev_eos = 1.0
        p = "M{},{} ".format(0, 0)
        for x, y, eos in zip(*strokes.T):
            p += '{}{},{} '.format('M' if prev_eos == 1.0 else 'L', x, y)
            prev_eos = eos
        path = svgwrite.path.Path(p)
        path = path.stroke(color=color, width=width, linecap='round').fill("none")
        dwg.add(path)

        initial_coord[1] -= line_height

    dwg.save()

def _simulate_paragraph_word_counts(strokes, words):

    strokes_list = []
    strokes_list_size = 0

    A4_width = 210
    A4_height = 297
    size_width = '{}mm'.format(A4_width)
    size_height = '{}mm'.format(A4_height)
    size = (size_width, size_height)
    
    line_height = 60
    max_x = 780
    initial_coord = np.array([0, -(3 * line_height / 4)])

    paragraph_word_counts = []

    for offsets, line in zip(strokes, words):

        if not line:
            initial_coord[1] -= line_height
            continue
        
        offsets[:, :2] *= 1.5
        strokes = drawing.offsets_to_coords(offsets)
        strokes = drawing.denoise(strokes)
        strokes[:, :2] *= 0.8
        size_x = strokes[:, 0].max() - strokes[:, 0].min()
        # size_y = strokes[:, 1].max() - strokes[:, 1].min()

        if strokes_list_size + size_x < max_x - 50:
            strokes_list_size += size_x
            strokes_list.append(strokes)
            # add current stroke to buffer
        else:
            paragraph_word_counts.append(len(strokes_list))
            strokes_list_size = 0
            strokes_list = []

    return paragraph_word_counts