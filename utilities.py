import numpy as np
import os

def _create_word_id_dict(words_index_path):
    dict = {}
    path = os.path.join(words_index_path)
    with open(path, 'r') as f:
        for line in f.readlines():
            line = line.replace('\n', '').split()
            dict[line[1]] = int(line[0]) + 1
    return dict

def _word_to_id(data, words_index_path):
    vocab = _create_word_id_dict(words_index_path)
    return [[vocab[word] if word in vocab else 0 for word in line] for line in data]

def _load_text(data_path, words_index_path):
    """
    Converts words in each line into a sequence of word ids. Maintains length of each line
    :param data_path: path to text data
    :param word_index_path: path to index-wordlist file
    :return: data with words replaced by word ids, sentence lengths
    """
    with open(data_path, 'r') as f:
        data = []
        slens = []
        elems_to_remove = []
        count = 0
        for line in f.readlines():
            line = line.replace('\n', '').split()
            if len(line) == 0:
                elems_to_remove.append(count)  
            else:
                data.append(line)
                slens.append(len(line))
            count+=1
    data = np.asarray(_word_to_id(data, words_index_path))
    return data, slens, elems_to_remove

def _load_text_eval(data_path, words_index_path):
    """
    Converts words in each line into a sequence of word ids. Maintains length of each line
    :param data_path: path to text data
    :param word_index_path: path to index-wordlist file
    :return: data with words replaced by word ids, sentence lengths
    """
    with open(data_path, 'r') as f:
        data = []
        slens = []
        count = 0
        for line in f.readlines():
            line = line.replace('\n', '').split()
            if len(line) == 0:
                data.append('s')
                slens.append(1)
            else:
                data.append(line)
                slens.append(len(line))
            count+=1
    data = np.asarray(_word_to_id(data, words_index_path))
    return data, slens

def text_to_array(data_path, words_index_path, is_eval=False):
    data, slens, elems_to_remove = _load_text(data_path, words_index_path)

    if is_eval:
        data, slens = _load_text_eval(data_path, words_index_path)
        elems_to_remove = None

    slens = np.asarray(slens, dtype=np.int32)
    processed_data = np.zeros((len(slens), np.max(slens)), dtype=np.int32)
    for i, _ in zip(range(len(data)), slens):
        processed_data[i][0:slens[i]] = data[i]

    return processed_data, slens, elems_to_remove    