import random
from collections import defaultdict, Counter
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from bokeh.models import ColumnDataSource, LabelSet
from bokeh.plotting import figure, show, output_file
from bokeh.palettes import d3

from sentence import Sentence
from word import Word

ROOT_TAG = Word.ROOT_TAG
ROOT_WORD = Word.ROOT_WORD
ROOT_LABEL = Word.ROOT_LABEL
UNKNOWN_WORD = Word.UNKNOWN_WORD
UNKNOWN_TAG = Word.UNKNOWN_TAG
UNKNOWN_LABEL = Word.UNKNOWN_LABEL

# output_notebook()

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
    file = open(path + filename, 'r', encoding="utf-8")
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
    This function receives a list of Sentences and return
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
        gensim_word_sentence = [ROOT_WORD]
        gensim_POS_sentence = [ROOT_TAG]
        for word in sentence.words:
            gensim_word_sentence.append(word.FORM.lower())
            word_counts[word.FORM.lower()] += 1
            gensim_POS_sentence.append(word.UPOSTAG.lower())
            POS_counts[word.UPOSTAG.lower()] += 1

            # some labels comprise multiple words(i.e: "nsubj:poss"), so we take only the first label
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
#     with open("../resources//embedding50d_word.txt", 'w') as file:
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

def get_random_representation(length=50):
    return [random.random() for _ in range(0, length)]


def get_pre_trained_tag_model(language):
    file = open("../resources/embeddings/%s/embedding50d_pos.txt" % language, "r", encoding="utf-8")

    pre_trained_pos_tags = {}
    for line in file:
        tokens = line.split()
        if tokens[0] == UNKNOWN_TAG:
            unknown_representation = tokens[1:]
        else:
            pre_trained_pos_tags[tokens[0]] = tokens[1:]

    return defaultdict(lambda: unknown_representation, pre_trained_pos_tags)


def get_pre_trained_word_model(language):
    file = open("../resources/embeddings/%s/embedding50d_word.txt" % language, "r", encoding="utf-8")
    pre_trained_tokens = {}
    for line in file:
        tokens = line.split()
        if tokens[0] == UNKNOWN_WORD:
            unknown_representation = tokens[1:]
        else:
            pre_trained_tokens[tokens[0]] = tokens[1:]

    return defaultdict(lambda: unknown_representation, pre_trained_tokens)


gensim_word_sentences_train, word_counts_train, gensim_POS_sentences_train, POS_counts_train, label_counts_train = get_gensim_sentences(en_train_sentences())
gensim_word_sentences_dev, word_counts_dev, gensim_POS_sentences_dev, POS_counts_dev, label_counts_dev = get_gensim_sentences(en_dev_sentences())
gensim_word_sentences_test, word_counts_test, gensim_POS_sentences_test, POS_counts_test, label_counts_test = get_gensim_sentences(en_test_sentences())

ro_gensim_word_sentences_train, ro_word_counts_train, ro_gensim_POS_sentences_train, ro_POS_counts_train, ro_label_counts_train = get_gensim_sentences(ro_train_sentences())
ro_gensim_word_sentences_dev, ro_word_counts_dev, ro_gensim_POS_sentences_dev, ro_POS_counts_dev, ro_label_counts_dev = get_gensim_sentences(ro_dev_sentences())
ro_gensim_word_sentences_test, ro_word_counts_test, ro_gensim_POS_sentences_test, ro_POS_counts_test, ro_label_counts_test = get_gensim_sentences(ro_test_sentences())

pre_trained_word_model = get_pre_trained_word_model("en")
pre_trained_POS_model = get_pre_trained_tag_model("en")

ro_pre_trained_word_model = get_pre_trained_word_model("ro")
ro_pre_trained_POS_model = get_pre_trained_tag_model("ro")

w2i = {}
t2i = {}
l2i = {}
i2w = {}
i2t = {}
i2l = {}

w2i["en"] = defaultdict(lambda: len(w2i["en"]))
i2w["en"] = dict()
i2w["en"][w2i["en"][UNKNOWN_WORD]] = UNKNOWN_WORD  # word with index 0 are the words that are unknown.
word_vocabulary = pre_trained_word_model.keys()
for word in word_vocabulary:
    i2w["en"][w2i["en"][word]] = word  # trick
w2i["en"] = defaultdict(lambda: 0, w2i["en"])  # change the default behaviour so that it returns the index of unknown words(i.e. 0 )

# FOR ROMANIAN
w2i["ro"] = defaultdict(lambda: len(w2i["ro"]))
i2w["ro"] = dict()
i2w["ro"][w2i["ro"][UNKNOWN_WORD]] = UNKNOWN_WORD  # word with index 0 are the words that are unknown.
word_vocabulary = ro_pre_trained_word_model.keys()
for word in word_vocabulary:
    i2w["ro"][w2i["ro"][word]] = word  # trick
w2i["ro"] = defaultdict(lambda: 0, w2i["ro"])  # change the default behaviour so that it returns the index of unknown words(i.e. 0 )

t2i["en"] = defaultdict(lambda: len(t2i["en"]))
i2t["en"] = dict()
i2t["en"][t2i["en"][UNKNOWN_TAG]] = UNKNOWN_TAG  # tags with index 0 are the tags that are unknown.
pos_tag_vocabulary = pre_trained_POS_model.keys()
for tag in pos_tag_vocabulary:
    i2t["en"][t2i["en"][tag]] = tag  # trick
t2i["en"] = defaultdict(lambda: 0, t2i["en"])  # change the default behaviour so that it returns the index of unknown tags(i.e. 0 )

# FOR ROMANIAN
t2i["ro"] = defaultdict(lambda: len(t2i["ro"]))
i2t["ro"] = dict()
i2t["ro"][t2i["ro"][UNKNOWN_TAG]] = UNKNOWN_TAG  # tags with index 0 are the tags that are unknown.
ro_pos_tag_vocabulary = ro_pre_trained_POS_model.keys()
for tag in ro_pos_tag_vocabulary:
    i2t["ro"][t2i["ro"][tag]] = tag  # trick
t2i["ro"] = defaultdict(lambda: 0, t2i["ro"])  # change the default behaviour so that it returns the index of unknown tags(i.e. 0 )

l2i["en"] = defaultdict(lambda: len(l2i["en"]))
i2l["en"] = dict()
label_vocabulary = label_counts_train.keys()
for label in label_vocabulary:
    i2l["en"][l2i["en"][label]] = label  # trick
l2i["en"] = dict(l2i["en"])  # remove the default behaviour
l2i["ro"] = l2i["en"]
i2l["ro"] = i2l["en"]


def word_embeddings(language):
    glove_pre_trained_word_model = get_pre_trained_word_model(language)
    embeddings = []
    indices = len(i2w[language].keys())
    for index in range(0, indices):
        embeddings.append(glove_pre_trained_word_model[i2w[language][index]])
    return embeddings


def tag_embeddings(language):
    glove_pre_trained_pos_model = get_pre_trained_tag_model(language)
    embeddings = []
    indices = len(i2t[language].keys())
    for index in range(0, indices):
        embeddings.append(glove_pre_trained_pos_model[i2t[language][index]])
    return embeddings


if __name__ == '__main__':
    '''
    
    # models that have a default behaviour and return a representation for unknown words.
    pre_trained_word_model = get_glove_pre_trained_word_model()
    pre_trained_POS_model = get_glove_pre_trained_tag_model()


    # dictionaries for conversion from "elements to index" and from "indices to elements".
    w2i  # word to index
    t2i  # tag to index
    l2i  # label to index

    i2w  # index to index
    i2t  # index to tag
    i2l  # index to label


    # counter for words, tags and labels for training, development and test set.
    word_counts_train
    word_counts_dev
    word_counts_test

    POS_counts_train
    POS_counts_dev
    POS_counts_test

    label_counts_train
    label_counts_dev
    label_counts_test


    # lists of sentences; each sentence is a list of words(or tags); so you have a list of lists.
    gensim_word_sentences_train
    gensim_word_sentences_dev
    gensim_word_sentences_test

    gensim_POS_sentences_train
    gensim_POS_sentences_dev
    gensim_POS_sentences_test
    
    '''
