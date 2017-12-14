import torch
import torch.autograd as autograd
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

import embedding as em

POS_EMBEDDINGS_RELATIVE_PATH = "../resources/parameters/pos_embeddings"

WORD_EMBEDDINGS_RELATIVE_PATH = "../resources/parameters/word_embeddings"

OPTIMISER_MODEL_RELATIVE_PATH = "../resources/parameters/optimiser_weights"

LSTM_MODEL_WEIGHTS_RELATIVE_PATH = "../resources/parameters/model_weights"

NUM_EPOCHS = 200

NUM_LAYERS = 3

HIDDEN_DIMENSION = 400

INPUT_SIZE = 100

LEARNING_RATE = 0.001

MLP_ARC_OUTPUT = 500

if torch.cuda.is_available():
    floatTensor = torch.cuda.FloatTensor
    longTensor = torch.cuda.LongTensor
else:
    floatTensor = torch.FloatTensor
    longTensor = torch.LongTensor


class BiLSTMTagger(nn.Module):
    def __init__(self, input_size, hidden_dim, num_layers, mlp_arc_dimension, word_embeddings, pos_embeddings):
        super(BiLSTMTagger, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

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

        self.hidden = self.init_hidden(0)

        # One linear layer Wx + b, input dim 100 output dim 100
        self.mlp_arc_head = nn.Linear(hidden_dim, mlp_arc_dimension)
        # self.mlp_head2 = nn.Linear(input_size, input_size)

        # One linear layer Wx + b, input dim 100 output dim 100
        self.mlp_arc_dependent = nn.Linear(hidden_dim, mlp_arc_dimension)
        # self.mlp_dependent2 = nn.Linear(input_size, input_size)

        # # One linear layer Wx + b for labels head, input dim 100 output dim 100
        # self.mlp_label_head = nn.Linear(hidden_dim, input_size)
        #
        # # One linear layer Wx + b for labels dependent, input dim 100 output dim 100
        # self.mlp_label_dependent = nn.Linear(hidden_dim, input_size)

        # activation function  h(Wx + b)
        self.ReLU = nn.ReLU()

        # One linear layer for the first tranformation from Dozat & Manning
        self.transform_H_dependent = nn.Linear(mlp_arc_dimension, mlp_arc_dimension, bias=True)

        if torch.cuda.is_available():
            self.cuda()
        else:
            self.cpu()

    def init_hidden(self, length=1):
        # Before we've done anything, we don't have any hidden state.
        # Refer to the PyTorch documentation to see exactly why they have this dimensionality.
        # The axes semantics are (num_layers*num_directions, mini-batch_size, hidden_dim)
        return (
            autograd.Variable(torch.zeros(self.num_layers * self.num_directions, length, self.hidden_dim // 2)).type(
                floatTensor),
            autograd.Variable(torch.zeros(self.num_layers * self.num_directions, length, self.hidden_dim // 2)).type(
                floatTensor))

    def forward(self, sentence_word_indices, sentence_pos_indices):
        # get embeddings for sentence
        embedded_sentence = torch.cat(
            (self.word_embeddings(sentence_word_indices), self.pos_embeddings(sentence_pos_indices)), 1)
        length = len(embedded_sentence)
        inputs = embedded_sentence.view(length, 1, -1)

        # pass through the biLstm layer
        lstm_out, self.hidden = self.bilstm(inputs, self.hidden)

        # compute head and dependent representations
        R = lstm_out.view(length, -1)
        H_head = self.ReLU(self.mlp_arc_head(R))
        H_dependent = self.ReLU(self.mlp_arc_dependent(R))

        # calculate scores on formula H_head * (U1 * H_dep + u2)
        H_dep_transformed = self.transform_H_dependent(H_dependent)
        scores = torch.mm(H_head, torch.transpose(H_dep_transformed, 0, 1))

        return scores


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


if __name__ == '__main__':
    word_embeddings = autograd.Variable(torch.from_numpy(np.array(em.word_embeddings(), dtype=np.float))).type(
        floatTensor)
    pos_embeddings = autograd.Variable(torch.from_numpy(np.array(em.tag_embeddings(), dtype=np.float))).type(
        floatTensor)

    model = BiLSTMTagger(input_size=INPUT_SIZE, hidden_dim=HIDDEN_DIMENSION, num_layers=NUM_LAYERS,
                         mlp_arc_dimension=MLP_ARC_OUTPUT,
                         word_embeddings=word_embeddings, pos_embeddings=pos_embeddings)
    model.train(True)

    parameters = model.parameters()
    optimizer = torch.optim.Adam(parameters, lr=LEARNING_RATE, weight_decay=1e-6)
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, verbose=True)

    continue_train = False
    save_model = False

    if continue_train:
        model.load_state_dict(torch.load(LSTM_MODEL_WEIGHTS_RELATIVE_PATH))
        optimizer.load_state_dict(torch.load(OPTIMISER_MODEL_RELATIVE_PATH))
        model.word_embeddings.weight = nn.Parameter(
            torch.from_numpy(np.array(torch.load(WORD_EMBEDDINGS_RELATIVE_PATH), dtype=np.float))).type(floatTensor)
        model.pos_embeddings.weight = nn.Parameter(
            torch.from_numpy(np.array(torch.load(POS_EMBEDDINGS_RELATIVE_PATH), dtype=np.float))).type(floatTensor)

    loss_function = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        loss_function.cuda()

    conllu_sentences = em.en_train_sentences()

    for epoch in range(NUM_EPOCHS):
        print("Epoch [%d/%d]\tLoss:" % (epoch + 1, NUM_EPOCHS), end="", flush=True)
        epoch_loss = 0
        for conllu_sentence in conllu_sentences[19:20]:
            sentence = conllu_sentence.get_word_list()
            tags = conllu_sentence.get_pos_list()
            # labels = [em.l2i(l) for l in conllu_sentence.get_label_list()]

            # Step 1. Remember that PyTorch accumulates gradients.
            # We need to clear them out before each instance
            optimizer.zero_grad()

            # Also, we need to clear out the hidden state of the LSTM,
            # detaching it from its history on the last instance.
            model.hidden = model.init_hidden(len(sentence))

            # Step 2. Get our inputs ready for the network, that is, turn them into
            # Variables of word indices.
            sentence_in = prepare_sequence(sentence, em.w2i)
            post_tags_in = prepare_sequence(tags, em.t2i)

            # Step 3. Run our forward pass.
            arc_scores = model(sentence_in, post_tags_in)
            # arc_scores, label_scores = model(sentence_in, post_tags_in)

            # Step 4. Compute the loss, gradients, and update the parameters by calling optimizer.step()
            target_arcs = autograd.Variable(torch.from_numpy(np.array(conllu_sentence.get_head_representation(), dtype=np.long))).type(longTensor)
            loss_arcs = loss_function(arc_scores.permute(1, 0), target_arcs)
            loss = loss_arcs
            loss.backward()
            optimizer.step()

            # target_labels = autograd.Variable(torch.from_numpy(np.array(labels, dtype=np.long))).type(longTensor)
            # loss_labels = loss_function(label_scores, target_labels)
            # loss = loss_arcs + loss_labels

            epoch_loss += loss.data[0]

        epoch_loss /= len(conllu_sentences)
        print(":%f" % epoch_loss)

    plot_matrix(nn.Softmax()(arc_scores.permute(1, 0)).permute(1, 0))

    if save_model:
        model.cpu()
        torch.save(optimizer.state_dict(), OPTIMISER_MODEL_RELATIVE_PATH)
        torch.save(model.state_dict(), LSTM_MODEL_WEIGHTS_RELATIVE_PATH)
        torch.save(model.word_embeddings.weight.numpy(), WORD_EMBEDDINGS_RELATIVE_PATH)
        torch.save(model.pos_embeddings.weight.numpy(), POS_EMBEDDINGS_RELATIVE_PATH)
