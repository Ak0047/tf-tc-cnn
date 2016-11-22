
import xlrd
import re
import string
import numpy as np

def clean_str(string1):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string1 = re.sub(r"[^A-Za-z0-9(),.!?\'\`]", " ", string1)
    string1 = re.sub(r"\'s", " \'s", string1)
    string1 = re.sub(r"\'ve", " \'ve", string1)
    string1 = re.sub(r"n\'t", " n\'t", string1)
    string1 = re.sub(r"\'re", " \'re", string1)
    string1 = re.sub(r"\'d", " \'d", string1)
    string1 = re.sub(r"\'ll", " \'ll", string1)
    string1 = re.sub(r",", " , ", string1)
    string1 = re.sub(r"!", " ! ", string1)
    string1 = re.sub(r"\(", " \( ", string1)
    string1 = re.sub(r"\)", " \) ", string1)
    string1 = re.sub(r"\?", " \? ", string1)
    string1 = re.sub(r"\s{2,}", " ", string1)
    return string1.strip().lower()


def load_data_and_labels(data_file):
    '''
    col0 : input_text_train
    col1 : labels_train
    col2 : input_text_test
    col3 : labels_test
    :param data_file:
    :return:
    '''
    x_train = []
    y_train = []

    x_test = []
    y_test = []

    xl_workbook = xlrd.open_workbook(data_file)
    xl_sheet = xl_workbook.sheet_by_index(0)
    for r in range(xl_sheet.nrows):
        if str(xl_sheet.cell_value(r, 0)) is not "":
            sentence = clean_str(str(xl_sheet.cell_value(r, 0)).lower())
            sentence = sentence.translate(None, string.punctuation)
            x_train.append(sentence)
            y_train.append(xl_sheet.cell_value(r, 1))
        if str(xl_sheet.cell_value(r, 2)) is not "":
            sentence = clean_str(str(xl_sheet.cell_value(r, 2)).lower())
            sentence = sentence.translate(None, string.punctuation)
            x_test.append(sentence)
            y_test.append(xl_sheet.cell_value(r, 3))
    return [x_train, y_train, x_test, y_test]




def get_y_indices_dict(y_set, y_labels):
    """Return the dict mapping elements from label set of indices of the input labels.
        >>> get_y_indices_dict(["1", "2"], ["1", "1", "2"])
        {'1': [0, 1], '2': [2]}
    """
    y_index_dict = {}
    for label in y_set:
        indices = [i for i, y in enumerate(y_labels) if y == label]
        y_index_dict[label] = indices

    return y_index_dict


def vecTransform(X, model):
    vectors = []
    index2word_set = set(model.index2word)
    num_features = model.layer1_size
    for words in X:
        featureVec = np.zeros((num_features,), dtype="float32")
        nwords = 0
        for word in words.split():
            if word in index2word_set:
                nwords = nwords + 1.
                featureVec = np.add(featureVec, model[word])
        if nwords > 0:
            featureVec = np.divide(featureVec, nwords)
        vectors.append(featureVec)
    return vectors


class DataParser(object):
    """
    This is a parsing class of the csv dataset
    Holds the raw input text data
    """

    def __init__(self, data_file):
        self.x_text, self.y_labels, self.x_text_test, self.y_labels_test = load_data_and_labels(data_file)
        self.y_set = list(set(self.y_labels))
        y_indices_dict = get_y_indices_dict(self.y_set, self.y_labels)
        self.y_x_mapping = {y: map(lambda x: self.x_text[x], y_indices_dict[y]) for y in self.y_set}
        self.batch_iter = 0
        self.max_sentence_length = max([len(x.split(" ")) for x in self.x_text])

    def num_classes(self):
        return len(self.y_set)

    def num_examples(self):
        return len(self.x_text)

    def get_all_words_from_data(self):
        return [word for line in self.x_text for word in line.split()]

    def get_y_vector(self, y_label):
        vector = np.zeros(len(self.y_set))
        vector[self.y_set.index(y_label)] = 1
        return vector

    def get_y_vectors(self, y_labels):
        return [self.get_y_vector(y) for y in y_labels]

    def get_vectorize_train_data_mean(self, model, start_index, end_index):
        return vecTransform(self.x_text[start_index:end_index], model), \
               self.get_y_vectors(self.y_labels[start_index:end_index])

    def get_vectorize_test_data_mean(self, model):
        return vecTransform(self.x_text_test, model), \
               self.get_y_vectors(self.y_labels_test)

    def get_word_vector(self, word, model):
        if word in set(model.index2word):
            return np.array(model[word])
        else:
            return np.zeros(model.layer1_size)

    def get_vectorize_train_data_conv(self, model, start_index, end_index):
        y_data = self.y_labels[start_index: end_index]
        x_data = self.x_text[start_index: end_index]
        return self.get_conv_data(model, x_data), self.get_y_vectors(y_data)

    def get_conv_data(self, model, x_data):
        vectors = []
        for sentence in x_data:
            words = sentence.split()
            extra_words = self.max_sentence_length - len(words)
            sentence_vector = np.array([])
            for word in words:
                sentence_vector = np.concatenate((sentence_vector, self.get_word_vector(word, model)), axis=0)
            for i in range(extra_words):
                zero_vec = np.zeros(model.layer1_size)
                sentence_vector = np.concatenate((sentence_vector, zero_vec), axis=0)
            vectors.append(sentence_vector)
        return np.array(vectors)

    def get_vectorize_test_data_conv(self, model):
        return self.get_conv_data(model, self.x_text_test), self.get_y_vectors(self.y_labels_test)

    def next_batch_conv(self, batch_size, model):
        if self.batch_iter + batch_size > self.num_examples():
            start_index = self.batch_iter
            self.batch_iter = 0
            return self.get_vectorize_train_data_conv(model, start_index, -1)
        else:
            start_index = self.batch_iter
            self.batch_iter += batch_size
            return self.get_vectorize_train_data_conv(model, start_index, start_index + batch_size)


    def next_batch(self, batch_size, model):
        if self.batch_iter + batch_size > self.num_examples():
            start_index = self.batch_iter
            self.batch_iter = 0
            return self.get_vectorize_train_data_mean(model, start_index, -1)
        else:
            start_index = self.batch_iter
            self.batch_iter += batch_size
            return self.get_vectorize_train_data_mean(model, start_index, start_index + batch_size)

if __name__ == '__main__':
    import doctest
    doctest.testmod()