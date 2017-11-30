import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import embedding as em

torch.manual_seed(1)


class BiLSTMTagger(nn.Module):
    def __init__(self, input_size, hidden_dim, num_layers, word_embeddings, pos_embeddings):
        super(BiLSTMTagger, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.word_embeddings = nn.Embedding(word_embeddings.data.shape[0], word_embeddings.data.shape[1])
        self.word_embeddings.weight = nn.Parameter(word_embeddings.data)
        self.word_embeddings.requires_grad = False

        self.pos_embeddings = nn.Embedding(pos_embeddings.data.shape[0], pos_embeddings.data.shape[1])
        self.pos_embeddings.weight = nn.Parameter(pos_embeddings.data)
        self.pos_embeddings.requires_grad = False

        self.bilstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        self.hidden = self.init_hidden()

        # One linear layer Wx + b, input dim 100 output dim 100
        self.mlp_head = nn.Linear(input_size, input_size)

        # One linear layer Wx + b, input dim 100 output dim 100
        self.mlp_dependent = nn.Linear(input_size, input_size)

        # One bi-linear layer x1∗W1∗x2+b, input1 dim 100,input2 dim 100, output dim 1
        self.bi_linear = nn.Bilinear(input_size, input_size, 1)

        self.softmax = nn.Softmax()

        # more layers for MLP and OUTPUT

    def init_hidden(self):
        # Before we've done anything, we don't have any hidden state.
        # Refer to the PyTorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers*num_directions, mini-batch_size, hidden_dim)
        return (autograd.Variable(torch.zeros(self.num_layers * 2, 1, self.hidden_dim // 2)),
                autograd.Variable(torch.zeros(self.num_layers * 2, 1, self.hidden_dim // 2)))

    def forward(self, x, x_pos):
        # get embeddings for sentence
        embedded_sentence = torch.cat((self.word_embeddings(x), self.pos_embeddings(x_pos)), 1)
        inputs = embedded_sentence.view(len(embedded_sentence), 1, -1)

        # pass through the biLstm layer
        lstm_out, self.hidden = self.bilstm(inputs, self.hidden)

        # do further processing in MLP
        # for each pair v_i, v_j, go with v_1 through mlp_head and with v_j to mlp_dependent
        matrix = []
        for v_i in lstm_out:
            matrix_row = []
            for v_j in lstm_out:
                v_i_head = self.mlp_head(v_i)
                v_j_dependent = self.mlp_dependent(v_j)

                # for each pair, of v_i_head and v_j_dependent go through bi_linear, so that we have a score
                score = self.bi_linear(v_i_head, v_j_dependent)

                # append to matrix_row the score
                matrix_row.append(score)

            matrix_row = torch.cat(matrix_row, 1)

            print(matrix_row)
            matrix_row = self.softmax(matrix_row)
            print(matrix_row)

            # append to matrix  the
            matrix.append(matrix_row)

        matrix = torch.cat(matrix, 0)

        return matrix


def prepare_sequence(sequence, element2index):
    """
    :param sequence: sequence of elements
    :param element2index: dictionary to map elements to index
    :return: autograd.Variable(torch.LongTensor(X)), where "x" is the sequence of indexes.
    """
    indexes = [element2index[element] for element in sequence]
    tensor = torch.LongTensor(indexes)
    return autograd.Variable(tensor)


if __name__ == '__main__':
    word_embeddings = autograd.Variable(torch.from_numpy(np.array(em.word_embeddings()).astype(np.float)).float())
    pos_embeddings = autograd.Variable(torch.from_numpy(np.array(em.tag_embeddings()).astype(np.float)).float())

    model = BiLSTMTagger(input_size=100, hidden_dim=100, num_layers=2, word_embeddings=word_embeddings,
                         pos_embeddings=pos_embeddings)

    loss_function = nn.NLLLoss()

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.SGD(parameters, lr=0.1)

    sentences = em.gensim_word_sentences_train
    pos_tags = em.gensim_POS_sentences_train

    for epoch in range(1):
        counter = 0
        # for i in np.arange(0, len(sentences)):
        for i in np.arange(0, 1):
            print(counter)
            counter += 1

            sentence = sentences[i]
            tags = pos_tags[i]

            print(sentence)

            # Step 1. Remember that PyTorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Also, we need to clear out the hidden state of the LSTM,
            # detaching it from its history on the last instance.
            model.hidden = model.init_hidden()

            # Step 2. Get our inputs ready for the network, that is, turn them into
            # Variables of word indices.
            sentence_in = prepare_sequence(sentence, em.w2i)
            post_tags_in = prepare_sequence(tags, em.t2i)

            # Step 3. Run our forward pass.
            tag_scores = model(sentence_in, post_tags_in)

            print(tag_scores)


            #
            # # Step 4. Compute the loss, gradients, and update the parameters by
            # #  calling optimizer.step()
            # loss = loss_function(tag_scores, targets)
            #
            # loss.backward()
            # optimizer.step()

            # # See what the scores are after training
            # inputs = prepare_sequence(sentences[0][0], word_to_ix)
            # tag_scores = model(inputs)
            # # The sentence is "the dog ate the apple".  i,j corresponds to score for tag j
            # #  for word i. The predicted tag is the maximum scoring tag.
            # # Here, we can see the predicted sequence below is 0 1 2 0 1
            # # since 0 is index of the maximum value of row 1,
            # # 1 is the index of maximum value of row 2, etc.
            # # Which is DET NOUN VERB DET NOUN, the correct sequence!
            # print(tag_scores)
