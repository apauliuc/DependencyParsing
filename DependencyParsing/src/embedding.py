from collections import defaultdict, Counter
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from bokeh.models import ColumnDataSource, LabelSet
from bokeh.plotting import figure, show, output_file
from bokeh.palettes import d3
from bokeh.io import output_notebook

import numpy as np

from sentence import Sentence

output_notebook()

en_path = "../resources/annotated_data/en/"
en_dev_filename = "en-ud-dev.conllu"
en_test_filename = "en-ud-test.conllu"
en_train_filename = "en-ud-train.conllu"

ro_path = "../resources/annotated_data/ro/"
ro_dev_filename = "ro-ud-dev.conllu"
ro_test_filename = "ro-ud-test.conllu"
ro_train_filename = "ro-ud-train.conllu"


def en_train_sentences():
    return read_sentences(en_path, en_train_filename)


def en_dev_sentences():
    return read_sentences(en_path, en_dev_filename)


def en_test_sentences():
    return read_sentences(en_path, en_test_filename)


def ro_train_sentences():
    return read_sentences(ro_path, ro_train_filename)


def ro_dev_sentences():
    return read_sentences(ro_path, ro_dev_filename)


def ro_test_sentences():
    return read_sentences(ro_path, ro_test_filename)


def read_sentences(path, filename):
    sentences = []
    sentence_lines = []
    file = open(path + filename, 'r')
    for line in file:
        if line == '\n':
            sentences.append(Sentence.from_lines(sentence_lines))
            sentence_lines = []
        else:
            sentence_lines.append(line)

    return sentences


def write_sentences(sentences, path, filename):
    with open(path + filename, 'w+') as file:
        for sentence in sentences:
            file.write(str(sentence) + "\n")


def emb_scatter(data, names, N=20, perplexity=30.0):
    """
    Function for plotting word_embeddings and words using TSNE.
    TSNE finds a way to plot multidimensional data to a
    bidimensional plane. It assures that data close in the
    multidimensionalspace will also be close in budimensional
    place, but not the other way around.
    """
    # try to find some clusters
    print("finding clusters")
    kmeans = KMeans(n_clusters=N)
    kmeans.fit(data)
    klabels = kmeans.labels_

    # get a tsne fit
    print("fitting tsne")
    tsne = TSNE(n_components=2, perplexity=perplexity)
    emb_tsne = tsne.fit_transform(data)

    # plot the tsne of the word_embeddings with bokeh
    # source: https://github.com/oxford-cs-deepnlp-2017/practical-1
    p = figure(tools="pan,wheel_zoom,reset,save",
               toolbar_location="above",
               title="T-SNE for most common words")

    # set colormap as a list
    colormap = d3['Category20'][N]
    colors = [colormap[i] for i in klabels]

    source = ColumnDataSource(data=dict(x1=emb_tsne[:, 0],
                                        x2=emb_tsne[:, 1],
                                        names=names,
                                        colors=colors))

    p.scatter(x="x1", y="x2", size=8, source=source, color='colors')

    labels = LabelSet(x="x1", y="x2", text="names", y_offset=6,
                      text_font_size="8pt", text_color="#555555",
                      source=source, text_align='center')
    p.add_layout(labels)

    show(p)


def get_gensim_sentences(sentences):
    """
    This function receives a list of Sencences and return
    the word and POS sentences in the format Gensim needs,
    as well as Counters for words, POS tags and dependency
    relation labels.
    Format: [['i', 'like', 'custard'],...]
    All words are lower-cased
    """
    gensim_word_sentences = []
    word_counts = Counter()
    gensim_POS_sentences = []
    POS_counts = Counter()
    label_counts = Counter()
    for sentence in sentences:
        gensim_word_sentence = []
        gensim_POS_sentence = []
        for word in sentence.words:
            gensim_word_sentence.append(word.FORM.lower())
            word_counts[word.FORM.lower()] += 1
            gensim_POS_sentence.append(word.UPOSTAG.lower())
            POS_counts[word.UPOSTAG.lower()] += 1
            # some labels comprize multiple words(i.e: "nsubj:poss"), so we take only the first label
            label_counts[word.DEPREL.split(":")[0].lower()] += 1
        gensim_word_sentences.append(gensim_word_sentence)
        gensim_POS_sentences.append(gensim_POS_sentence)
    return gensim_word_sentences, word_counts, gensim_POS_sentences, POS_counts, label_counts


# def get_all_pre_trained_word_embeddings():
#     file = open("../resources/glove.6B.50d.txt")
#     pre_trained_tokens = {}
#     for line in file:
#         tokens = line.split()
#         pre_trained_tokens[tokens[0]] = tokens[1:]
#     return pre_trained_tokens


# def write_all_our_words():
#     unknown_representation = []
#     with open("../resources/glove50d.txt", 'w') as file:
#         for w in word_vocabulary:
#             if w in all_pre_trained_words:
#                 file.write(w + " ")
#                 for item in all_pre_trained_word_embeddings[w]:
#                     file.write(item + " ")
#                 file.write("\n")
#                 if word_counts_train[w] == 1:
#                     unknown_representation.append(all_pre_trained_word_embeddings[w])
#
#         unknown_representation = np.sum(np.array(unknown_representation).astype(np.double), axis=0) / len(
#             unknown_representation)
#         file.write("<UNK> ")
#
#         for item in unknown_representation:
#             file.write(str(item) + " ")


def get_glove_pre_trained_tag_model():
    """
    PLOT TWIST: we get the pos tag word_embeddings from training
    """
    model = {}
    word2vect = Word2Vec(gensim_POS_sentences_train, size=50, window=3, min_count=2, workers=4)
    for key in word2vect.wv.vocab.keys():
        model[key] = word2vect.wv[key]
    return model


def get_glove_pre_trained_word_model():
    file = open("../resources/glove50d.txt")
    pre_trained_tokens = {}
    for line in file:
        tokens = line.split()
        if tokens[0] != "<UNK>":
            pre_trained_tokens[tokens[0]] = tokens[1:]
        else:
            unknown_representation = tokens[1:]
    return defaultdict(lambda: unknown_representation, pre_trained_tokens)


gensim_word_sentences_train, word_counts_train, gensim_POS_sentences_train, POS_counts_train, label_counts_train = get_gensim_sentences(
    en_train_sentences())
gensim_word_sentences_dev, word_counts_dev, gensim_POS_sentences_dev, POS_counts_dev, label_counts_dev = get_gensim_sentences(
    en_dev_sentences())
gensim_word_sentences_test, word_counts_test, gensim_POS_sentences_test, POS_counts_test, label_counts_test = get_gensim_sentences(
    en_test_sentences())

# all_pre_trained_word_embeddings = get_all_pre_trained_word_embeddings()
# all_pre_trained_words = all_pre_trained_word_embeddings.keys()

pre_trained_word_model = get_glove_pre_trained_word_model()
pre_trained_POS_model = get_glove_pre_trained_tag_model()

w2i = defaultdict(lambda: len(w2i))
i2w = dict()
i2w[w2i["<UNK>"]] = "<UNK>"  # word with index 0 are the words that are unknown.
word_vocabulary = pre_trained_word_model.keys()
for word in word_vocabulary:
    i2w[w2i[word]] = word
w2i = defaultdict(lambda: 0, w2i)

t2i = defaultdict(lambda: len(t2i))
i2t = dict()
pos_tag_vocabulary = pre_trained_POS_model.keys()
for tag in pos_tag_vocabulary:
    i2t[t2i[tag]] = tag  # trick

l2i = defaultdict(lambda: len(l2i))
i2l = dict()
label_vocabulary = label_counts_train.keys()
for label in label_vocabulary:
    i2l[l2i[label]] = label  # trick


def word_embeddings():
    glove_pre_trained_word_model = get_glove_pre_trained_word_model()
    embeddings = []
    for index in i2w.keys():
        embeddings.append(glove_pre_trained_word_model[i2w[index]])
    return embeddings


def tag_embeddings():
    glove_pre_trained_pos_model = get_glove_pre_trained_tag_model()
    embeddings = []
    for index in i2t.keys():
        embeddings.append(glove_pre_trained_pos_model[i2t[index]])
    return embeddings


if __name__ == '__main__':
    output_notebook()

    pre_trained_word_model = get_glove_pre_trained_word_model()
    pre_trained_POS_model = get_glove_pre_trained_tag_model()

    word_vector = []
    word_list = []
    for word, count in word_counts_train.most_common(1000):
        word_list.append(word)
        word_vector.append(pre_trained_word_model[word])
    emb_scatter(word_vector, word_list, N=20)

    top_POS = [k for k, v in POS_counts_train.most_common(1000)]

    top_POS_vecs = []
    for pos in top_POS:
        top_POS_vecs.append(pre_trained_POS_model[pos])
    emb_scatter(top_POS_vecs, top_POS, N=17)
