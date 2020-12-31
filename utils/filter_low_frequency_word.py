import os
import sys

version = '1.2'

corpus = 'data/kothou_keu_nei_v1.2.txt'
corpus_clean = 'data/kothou_keu_nei_clean_v1.2.txt'
corpus_filter_low_freq_words = 'data/kothou_keu_nei_clean_filter_low_freq_word_with_UNK_token_v1.2.txt'

# for dataset version 1.2
chars_to_remove = [
    '!', '(', ')', ',', '-', '.', '1', ':', ';', '<', '?', 'R', 'r', '{', '·',
    'я', 'अ', 'छ', 'ब', 'भ', 'य', 'र', 'ल', 'श', 'ৰ', 'ৱ', '৷', '–', '—', 
    '‘', '’', '“', '”', '…', '।'
]

min_frequency = 10 # frequency bellow which word will be filter

def clean(filter_list, text):
    for ch in filter_list:
        text = text.replace(ch, '')
    text = text.strip()
    return text


# clean-up corpus 
clean_lines = []
with open(corpus, 'r') as fp:
    lines = fp.readlines()
    for line in lines:
        text = clean(chars_to_remove, line)
        if text and text != '':
            clean_lines.append(text)

# write clean corpus to a file
with open(corpus_clean, 'w') as fp:
    fp.write('\n'.join(clean_lines))
    print("clean corpus write at:", corpus_clean)

# count frequency of each words in the corpus
text_frequ = {}
for line in clean_lines:
    words = line.split()
    for word in words:
        word = word.strip()
        if word not in text_frequ:
            text_frequ[word] = 1
        else:
            text_frequ[word] += 1

# filter low frequency words and write lines to a file
clean_lines_filtered = []
for text_line in clean_lines:
    text_words = text_line.split()
    filter_line = []
    for word in text_words:
        word = word.strip()
        if text_frequ[word] >= min_frequency:
            filter_line.append(word)
        else:
            filter_line.append('UNK')
    clean_lines_filtered.append(' '.join(filter_line))

# write filtered lines to a file
with open(corpus_filter_low_freq_words, 'w') as fp:
    fp.write('\n'.join(clean_lines_filtered))
    print("filtered corpus write at:", corpus_filter_low_freq_words)