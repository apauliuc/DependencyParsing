import shutil
import time
import numpy as np
import torch
from torch import autograd
import matplotlib.pyplot as plt

import embedding as em
from BiLSTM import longTensor

import logging
logging.basicConfig(filename='../resources/logs/training_{}.log'.format(time.strftime('%d-%m-%Y%_H:%M:%S')),
                    level=logging.DEBUG)


def prepare_sequence(sequence, element2index):
    """
    :param sequence: sequence of elements
    :param element2index: dictionary to map elements to index
    :return: autograd.Variable(torch.LongTensor(X)), where "X" is the sequence of indexes.
    """
    indexes = [element2index[element] for element in sequence]
    tensor = longTensor(indexes)
    return autograd.Variable(tensor)


def plot_matrix(matrix_variable):
    plt.clf()
    numpy_a = matrix_variable.data.numpy()
    plt.imshow(numpy_a)
    plt.show()


def save_checkpoint(checkpoint, latest_path, best_path, best):
    torch.save(checkpoint, latest_path)
    if best:
        shutil.copyfile(latest_path, best_path)


def train_model(model, optimizer, loss_function, conllu_sentences, language):
    train_loss = 0
    arc_scores = None
    label_scores = None
    for conllu_sentence in conllu_sentences:
        sentence = conllu_sentence.get_word_list()
        tags = conllu_sentence.get_pos_list()
        labels = conllu_sentence.get_label_list()
        head_representation = conllu_sentence.get_head_representation()

        # Step 1. Remember that PyTorch accumulates gradients.
        # We need to clear them out before each instance
        optimizer.zero_grad()

        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Variables of word indices.
        sentence_in = prepare_sequence(sentence, em.w2i[language])
        post_tags_in = prepare_sequence(tags, em.t2i[language])
        labels_in = prepare_sequence(labels, em.l2i[language])

        # Step 3. Run our forward pass.
        arc_scores, label_scores = model(
            sentence_word_indices=sentence_in,
            sentence_pos_indices=post_tags_in,
            mode='train',
            heads=head_representation.tolist())

        # Step 4. Compute the loss, gradients, and update the parameters by calling optimizer.step()
        target_arcs = autograd.Variable(torch.from_numpy(np.array(head_representation, dtype=np.long))).type(
            longTensor)
        loss_arcs = loss_function(arc_scores.permute(1, 0), target_arcs)
        loss_labels = loss_function(label_scores, labels_in)
        loss = loss_arcs + loss_labels

        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]

    train_loss /= len(conllu_sentences)
    print('Training loss: {}'.format(train_loss))
    logging.info('Training loss: {}'.format(train_loss))
    return train_loss, arc_scores, label_scores


def validate_model(model, loss_function, conllu_sentences, language):
    validate_loss = 0
    arc_scores = None
    label_scores = None
    for conllu_sentence in conllu_sentences:
        sentence = conllu_sentence.get_word_list()
        tags = conllu_sentence.get_pos_list()
        labels = conllu_sentence.get_label_list()
        head_representation = conllu_sentence.get_head_representation()

        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Variables of word indices.
        sentence_in = prepare_sequence(sentence, em.w2i[language])
        post_tags_in = prepare_sequence(tags, em.t2i[language])
        labels_in = prepare_sequence(labels, em.l2i[language])

        # Step 3. Run our forward pass.
        arc_scores, label_scores = model(
            sentence_word_indices=sentence_in,
            sentence_pos_indices=post_tags_in,
            heads=head_representation.tolist(),
            mode='validate')

        # Step 4. Compute the loss
        target_arcs = autograd.Variable(torch.from_numpy(np.array(head_representation, dtype=np.long))).type(
            longTensor)
        loss_arcs = loss_function(arc_scores.permute(1, 0), target_arcs)
        loss_labels = loss_function(label_scores, labels_in)
        loss = loss_arcs + loss_labels

        validate_loss += loss.data[0]

    validate_loss /= len(conllu_sentences)
    print("Validation loss: {}".format(validate_loss))
    logging.info("Validation loss: {}".format(validate_loss))
    return validate_loss, arc_scores, label_scores


def test_model(model, loss_function, conllu_sentences, language):
    test_loss = 0
    arc_scores = None
    label_scores = None
    for conllu_sentence in conllu_sentences:
        sentence = conllu_sentence.get_word_list()
        tags = conllu_sentence.get_pos_list()
        labels = conllu_sentence.get_label_list()
        head_representation = conllu_sentence.get_head_representation()

        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Variables of word indices.
        sentence_in = prepare_sequence(sentence, em.w2i[language])
        post_tags_in = prepare_sequence(tags, em.t2i[language])
        labels_in = prepare_sequence(labels, em.l2i[language])

        # Step 3. Run our forward pass.
        arc_scores, label_scores = model(
            sentence_word_indices=sentence_in,
            sentence_pos_indices=post_tags_in,
            sentence=sentence,
            mode='test')

        # Step 4. Compute the loss
        target_arcs = autograd.Variable(torch.from_numpy(np.array(head_representation, dtype=np.long))).type(
            longTensor)
        loss_arcs = loss_function(arc_scores.permute(1, 0), target_arcs)
        loss_labels = loss_function(label_scores, labels_in)
        loss = loss_arcs + loss_labels

        test_loss += loss.data[0]

    test_loss /= len(conllu_sentences)
    print("Test loss: {}".format(test_loss))
    logging.info("Test loss: {}".format(test_loss))
    return test_loss, arc_scores, label_scores


def predict(model, conllu_sentence, language):
    sentence = conllu_sentence.get_word_list()
    tags = conllu_sentence.get_pos_list()
    labels = conllu_sentence.get_label_list()
    head_representation = conllu_sentence.get_head_representation()

    # Step 2. Get our inputs ready for the network, that is, turn them into
    # Variables of word indices.
    sentence_in = prepare_sequence(sentence, em.w2i[language])
    post_tags_in = prepare_sequence(tags, em.t2i[language])
    labels_in = prepare_sequence(labels, em.l2i[language])

    # Step 3. Run our forward pass.
    arc_scores, label_scores, predicted_arcs = model(
        sentence_word_indices=sentence_in,
        sentence_pos_indices=post_tags_in,
        mode='predict',
        heads=head_representation,
        sentence=sentence
    )
    predicted_labels = np.argmax(label_scores.data.numpy(), axis=1)

    return predicted_arcs, predicted_labels
