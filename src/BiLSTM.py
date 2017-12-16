import torch
import torch.autograd as autograd
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import math
import shutil
import time
import argparse
import logging

import embedding as em
import edmonds as ed

logging.basicConfig(filename='../resources/logs/training_{}.log'.format(time.strftime('%d-%m-%Y%_H:%M:%S')),
                    level=logging.DEBUG)

NUM_EPOCHS = 50

NUM_LAYERS = 3

HIDDEN_DIMENSION = 300

INPUT_SIZE = 100

LEARNING_RATE = 0.001

MLP_ARC_OUTPUT = 400

MLP_LABEL_OUTPUT = 100

if torch.cuda.is_available():
    floatTensor = torch.cuda.FloatTensor
    longTensor = torch.cuda.LongTensor
else:
    floatTensor = torch.FloatTensor
    longTensor = torch.LongTensor


class BiLSTMTagger(nn.Module):
    def __init__(self, input_size, hidden_dim, num_layers,
                 mlp_arc_dimension, mlp_label_dimension, n_labels,
                 word_embeddings, pos_embeddings):
        super(BiLSTMTagger, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.n_labels = n_labels

        self.word_embeddings = nn.Embedding(word_embeddings.data.shape[0], word_embeddings.data.shape[1])
        self.word_embeddings.weight = nn.Parameter(word_embeddings.data)
        self.word_embeddings.requires_grad = True

        self.pos_embeddings = nn.Embedding(pos_embeddings.data.shape[0], pos_embeddings.data.shape[1])
        self.pos_embeddings.weight = nn.Parameter(pos_embeddings.data)
        self.pos_embeddings.requires_grad = True

        self.bilstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        self.num_directions = 2 if self.bilstm.bidirectional else 1

        # One linear layer Wx + b, input dim 100 output dim 100
        self.mlp_arc_head = nn.Linear(hidden_dim, mlp_arc_dimension)

        # One linear layer Wx + b, input dim 100 output dim 100
        self.mlp_arc_dependent = nn.Linear(hidden_dim, mlp_arc_dimension)

        # activation function  h(Wx + b)
        self.ReLU = nn.ReLU()

        # One linear layer for the first arc tranformation from D&M
        self.transform_H_dependent = nn.Linear(mlp_arc_dimension, mlp_arc_dimension, bias=True)

        # One linear layer Wx + b for labels head, input dim 100 output dim 100
        self.mlp_label_head = nn.Linear(hidden_dim, mlp_label_dimension)

        # One linear layer Wx + b for labels dependent, input dim 100 output dim 100
        self.mlp_label_dependent = nn.Linear(hidden_dim, mlp_label_dimension)

        # Bilinear for first term
        self.label_bilinear = nn.Bilinear(mlp_label_dimension, mlp_label_dimension, n_labels, bias=False)

        # Normal linear for 2nd
        self.label_transform = nn.Linear(2 * mlp_label_dimension, n_labels, bias=True)

        if torch.cuda.is_available():
            self.cuda()
        else:
            self.cpu()

    def forward(self, sentence_word_indices, sentence_pos_indices, mode, heads=None, sentence=None):
        if (mode == 'train' or mode == 'validate') and heads is None:
            raise ValueError('Train mode requires golden head representation')
        if mode == 'predict' and sentence is None:
            raise ValueError('Predict mode requires sentence')

        # get embeddings for sentence
        embedded_sentence = torch.cat(
            (self.word_embeddings(sentence_word_indices), self.pos_embeddings(sentence_pos_indices)), 1)
        sentence_length = len(embedded_sentence)
        inputs = embedded_sentence.view(1, sentence_length, -1)

        # pass through the biLstm layer
        lstm_out = self.bilstm(inputs)[0]

        # compute head and dependent representations
        R = lstm_out.view(sentence_length, -1)
        H_head = self.ReLU(self.mlp_arc_head(R))
        H_dependent = self.ReLU(self.mlp_arc_dependent(R))

        # calculate scores on formula H_head * (U1 * H_dep + u2)
        H_dep_transformed = self.transform_H_dependent(H_dependent)
        scores = torch.mm(H_head, torch.transpose(H_dep_transformed, 0, 1))

        L_head = self.ReLU(self.mlp_label_head(R))
        L_dependent = self.ReLU(self.mlp_label_dependent(R))

        if mode == 'train' or mode == 'validate':
            Ryi = L_head[tuple(heads),]
        elif mode == 'predict':
            scores_data = scores.data.cpu().numpy()
            root = np.argmax(scores_data[0, 1:])  # get the true root node
            mst = ed.edmonds_list(cost_matrix=scores_data[1:, 1:], sentence=sentence[1:], root=root)
            heads = np.zeros(len(sentence), dtype=np.int)

            heads[1] = root
            for pair in mst:
                head = pair[0]
                dep = pair[1]
                # the first element should always point to zero because mst does not know about the root ROOT stuff
                heads[dep + 1] = head + 1
            Ryi = L_head[tuple(heads.tolist()),]
        else:
            raise ValueError('Unknown mode: {}.'.format(mode))

        first_term = self.label_bilinear(Ryi, L_dependent)
        second_term = self.label_transform(torch.cat((Ryi, L_dependent), dim=1))

        label_scores = first_term + second_term

        scores.type(floatTensor)

        return scores, label_scores


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
    numpy_A = matrix_variable.data.numpy()
    plt.imshow(numpy_A)
    plt.show()


def save_checkpoint(checkpoint, best):
    torch.save(checkpoint, LATEST_CHECKPOINT_RELATIVE_PATH)
    if best:
        shutil.copyfile(LATEST_CHECKPOINT_RELATIVE_PATH, BEST_CHECKPOINT_RELATIVE_PATH)


def train_model(model, optimizer, loss_function, conllu_sentences, language):
    train_loss = 0
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
            heads=head_representation.tolist(),
            mode='train')

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
    prediction_loss = 0
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
            mode='validate')

        # Step 4. Compute the loss
        target_arcs = autograd.Variable(torch.from_numpy(np.array(head_representation, dtype=np.long))).type(
            longTensor)
        loss_arcs = loss_function(arc_scores.permute(1, 0), target_arcs)
        loss_labels = loss_function(label_scores, labels_in)
        loss = loss_arcs + loss_labels

        prediction_loss += loss.data[0]

    prediction_loss /= len(conllu_sentences)
    print("Prediction loss: {}".format(validate_loss))
    logging.info("Prediction loss: {}".format(validate_loss))
    return prediction_loss, arc_scores, label_scores


if __name__ == '__main__':
    # parse command-line arguments
    parser = argparse.ArgumentParser(description='Start anew or resume training')
    parser.add_argument('-m', '--mode', type=str, choices=['start', 'resume', 'test'], required=True,
                        help='start from scratch or resume training: [start, resume]')
    parser.add_argument('-l', '--language', type=str, choices=['en', 'ro'], required=True,
                        help='which language to model: [en, ro]')
    args = parser.parse_args()

    if args.mode == 'start':
        print('Started training at {}.'.format(time.strftime('%d-%m-%Y, %H:%M:%S')))
        logging.info('Started training at {}.'.format(time.strftime('%d-%m-%Y, %H:%M:%S')))
    elif args.mode == 'resume':
        print('Resumed training at {}.'.format(time.strftime('%d-%m-%Y, %H:%M:%S')))
        logging.info('Resumed training at {}.'.format(time.strftime('%d-%m-%Y, %H:%M:%S')))

    global LATEST_CHECKPOINT_RELATIVE_PATH
    LATEST_CHECKPOINT_RELATIVE_PATH = '../resources/checkpoints/{}/latest_checkpoint.tar'.format(args.language)
    global BEST_CHECKPOINT_RELATIVE_PATH
    BEST_CHECKPOINT_RELATIVE_PATH = '../resources/checkpoints/{}/best_checkpoint.tar'.format(args.language)

    word_embeddings = autograd.Variable(
        torch.from_numpy(np.array(em.word_embeddings(args.language), dtype=np.float))).type(
        floatTensor)
    pos_embeddings = autograd.Variable(
        torch.from_numpy(np.array(em.tag_embeddings(args.language), dtype=np.float))).type(
        floatTensor)

    model = BiLSTMTagger(input_size=INPUT_SIZE, hidden_dim=HIDDEN_DIMENSION, num_layers=NUM_LAYERS,
                         mlp_arc_dimension=MLP_ARC_OUTPUT, mlp_label_dimension=MLP_LABEL_OUTPUT,
                         n_labels=len(em.i2l[args.language].keys()),
                         word_embeddings=word_embeddings, pos_embeddings=pos_embeddings)
    model.train(True)

    parameters = model.parameters()
    optimizer = torch.optim.Adam(parameters, lr=LEARNING_RATE, weight_decay=1e-6, betas=(0.9, 0.9))
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, verbose=True)

    losses = {
        'train': {
            'min': {
                'value': math.inf,
                'epoch': 0
            },
            'history': []
        },
        'validate': {
            'min': {
                'value': math.inf,
                'epoch': 0
            },
            'history': []
        },
    }

    if args.mode == 'resume':
        checkpoint = torch.load(LATEST_CHECKPOINT_RELATIVE_PATH)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        model.word_embeddings.weight = nn.Parameter(
            torch.from_numpy(np.array(checkpoint['word_embeddings'], dtype=np.float))).type(floatTensor)
        model.pos_embeddings.weight = nn.Parameter(
            torch.from_numpy(np.array(checkpoint['pos_embeddings'], dtype=np.float))).type(floatTensor)
        losses = checkpoint['losses']
    elif args.mode == 'test':
        checkpoint = torch.load(BEST_CHECKPOINT_RELATIVE_PATH)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        model.word_embeddings.weight = nn.Parameter(
            torch.from_numpy(np.array(checkpoint['word_embeddings'], dtype=np.float))).type(floatTensor)
        model.pos_embeddings.weight = nn.Parameter(
            torch.from_numpy(np.array(checkpoint['pos_embeddings'], dtype=np.float))).type(floatTensor)
        losses = checkpoint['losses']

    loss_function = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        loss_function.cuda()

    if args.language == 'en':
        conllu_sentences_train = em.en_train_sentences()
        conllu_sentences_dev = em.en_dev_sentences()
        conllu_sentences_test = em.en_test_sentences()
    elif args.language == 'ro':
        conllu_sentences_train = em.ro_train_sentences()
        conllu_sentences_dev = em.ro_dev_sentences()
        conllu_sentences_test = em.ro_test_sentences()
    else:
        raise ValueError('Specified language {} is not supported.'.format(args.language))

    train_loss = math.inf
    validate_loss = math.inf
    if args.mode == 'start' or args.mode == 'resume':
        for epoch in range(NUM_EPOCHS):
            print("Epoch [%d/%d]..." % (epoch + 1, NUM_EPOCHS))
            logging.info("Epoch [%d/%d]..." % (epoch + 1, NUM_EPOCHS))

            is_best_model = False

            # train
            train_loss, train_arc_scores, train_label_scores = train_model(model, optimizer, loss_function,
                                                                           conllu_sentences_train, args.language)

            # validate
            validate_loss, validate_arc_scores, validate_label_scores = validate_model(model, loss_function,
                                                                                       conllu_sentences_dev, args.language)
            # check for best model
            if train_loss < losses['train']['min']['value']:
                print('Minimum training loss found in epoch {}'.format(epoch + 1))
                logging.info('Minimum training loss found in epoch {}'.format(epoch + 1))

                losses['train']['min']['value'] = train_loss
                losses['train']['min']['epoch'] = epoch
            if validate_loss < losses['validate']['min']['value']:
                print('Minimum validation loss found in epoch {}'.format(epoch + 1))
                logging.info('Minimum validation loss found in epoch {}'.format(epoch + 1))

                losses['validate']['min']['value'] = validate_loss
                losses['validate']['min']['epoch'] = epoch
                is_best_model = True

            # track loss and adjust learning rate if necessary
            scheduler.step(validate_loss)

            # track losses history
            losses['train']['history'].append(train_loss)
            losses['validate']['history'].append(validate_loss)

            model.cpu()
            # always save latest checkpoint after an epoch, and flag if best checkpoint
            save_checkpoint({
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'losses': losses,
                'word_embeddings': model.word_embeddings.weight.data.numpy(),
                'pos_embeddings': model.pos_embeddings.weight.data.numpy(),
                'optimizer': optimizer.state_dict(),
            }, is_best_model)

            if torch.cuda.is_available():
                model.cuda()

            if validate_loss > losses['validate']['min']['value'] and epoch - losses['validate']['min']['epoch'] > 10:
                print('Ten epochs with no improvement have passed. Stopping training...')
                logging.info('Ten epochs with no improvement have passed. Stopping training...')

                break

        print('Finished training at {}.'.format(time.strftime('%d-%m-%Y, %H:%M:%S')))
        logging.info('Finished training at {}.'.format(time.strftime('%d-%m-%Y, %H:%M:%S')))
    elif args.mode == 'test':
        test_loss, validate_arc_scores, validate_label_scores = test_model(model, loss_function, conllu_sentences_test, args.language)
        print(test_loss)
        # TODO: do some more stuff here probably

    # DEBUG
    # print([em.i2l[l] for l in np.argmax(nn.Softmax()(train_label_scores).data.numpy(), axis=1)])
    # print([em.i2l[l] for l in np.argmax(nn.Softmax()(validate_label_scores).data.numpy(), axis=1)])
    # print(conllu_sentences_train.get_label_list())
    # plot_matrix(nn.Softmax()(train_label_scores))
    # plot_matrix(nn.Softmax()(validate_label_scores))
    # plot_matrix(nn.Softmax()(train_arc_scores.permute(1, 0)).permute(1, 0))
    # plot_matrix(nn.Softmax()(validate_arc_scores.permute(1, 0)).permute(1, 0))
