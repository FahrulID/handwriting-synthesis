import numpy as np
from handwriting_synthesis import Hand

full_text = """Bagian ini secara umum berisi latar belakang dan alasan penulis memilih objek penelitian. Uraian dimulai dengan penjelasan mengenai hal yang bersifat umum terkait dengan topik TA, kemudian diarahkan kepada hal yang lebih khusus yaitu judul proposal TA. Objek yang akan diteliti harus dijelaskan secara konkret sebagai pengantar menuju permasalahan, dan sebagai hasil kajian studi terdahulu hasil analisis atas data sekunder, tentang obyek yang akan diteliti dirancang, disertai alasan mengapa masalah tersebut perlu diteliti baik secara teoritis maupun praktis. Bagian ini secara umum berisi latar belakang dan alasan penulis memilih objek penelitian. Uraian dimulai dengan penjelasan mengenai hal yang bersifat umum terkait dengan topik TA, kemudian diarahkan kepada hal yang lebih khusus yaitu judul proposal TA. Objek yang akan diteliti harus dijelaskan secara konkret sebagai pengantar menuju permasalahan, dan sebagai hasil kajian studi terdahulu hasil analisis atas data sekunder, tentang obyek yang akan diteliti dirancang, disertai alasan mengapa masalah tersebut perlu diteliti baik secara teoritis maupun praktis."""
# full_text = """Bagianini secaraumum berisilatar belakangdan alasanpenulis memilihobjek penelitian."""

lines = []

if __name__ == '__main__':
    hand = Hand()

    # split the text into words
    # words = full_text.split()
    # line = ''
    # for word in words:
    #     if len(line) + len(word) + 1 < 75:
    #         line += ' ' + word
    #     else:
    #         lines.append(line)
    #         line = word    

    # print(lines)
    # print(words)

    words = full_text.split()
    
    biases = [0.5 for i in words]
    styles = [8 for i in words]
    
    # biases = [0.6 for i in lines]
    # styles = [8 for i in lines]

    # make biases for each line based on number of lines normalized into 1
    # biases = np.linspace(0.1, 0.9, len(lines))
    # make styles for each line based on number of lines normalized into 1
    # styles = np.linspace(1, 10, len(lines))
    # styles = np.round(styles).astype(int)

    hand.write_document(
        filename='img/test.svg',
        words=words,
        biases=biases,
        styles=styles,
    )
    # hand.write(
    #     filename='img/test.svg',
    #     lines=lines,
    #     biases=biases,
    #     styles=styles,
    #     text_align='left'
    # )
