import torch
import torch.nn as nn
import numpy as np
import time
import logging

import edmonds as ed

logging.basicConfig(filename='../resources/logs/training_{}.log'.format(time.strftime('%d-%m-%Y%_H:%M:%S')),
                    level=logging.DEBUG)

NUM_EPOCHS = 50
NUM_LAYERS = 3
HIDDEN_DIMENSION = 400
INPUT_SIZE = 100
LEARNING_RATE = 0.001
MLP_ARC_OUTPUT = 500
MLP_LABEL_OUTPUT = 100

if torch.cuda.is_available():
    floatTensor = torch.cuda.FloatTensor
    longTensor = torch.cuda.LongTensor
else:
    floatTensor = torch.FloatTensor
    longTensor = torch.LongTensor


def get_true_root(arc_scores):
    for index, column in enumerate(arc_scores.data.cpu().numpy()[:, 1:].T):
        if np.argmax(column) == 0:
            return index + 1
    return np.argmax(nn.Softmax(dim=1)(arc_scores.permute(1, 0)).permute(1, 0).data.numpy()[0, 1:]) + 1


class BiLSTMTagger(nn.Module):
    def __init__(self, input_size, hidden_dim, num_layers,
                 mlp_arc_dimension, mlp_label_dimension, n_labels,
                 word_embeddings, pos_embeddings):
        super(BiLSTMTagger, self).__init__()

        self.hidden_dim = hidden_dim * 2
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
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        self.Dropout = nn.Dropout(p=0.33)

        # One linear layer Wx + b, input dim 800 output dim 100
        self.mlp_arc_head = nn.Linear(self.hidden_dim, mlp_arc_dimension)

        # One linear layer Wx + b, input dim 800 output dim 100
        self.mlp_arc_dependent = nn.Linear(self.hidden_dim, mlp_arc_dimension)

        # activation function  h(Wx + b)
        self.ReLU = nn.ReLU()

        # One linear layer for the first arc tranformation from D&M
        self.transform_H_dependent = nn.Linear(mlp_arc_dimension, mlp_arc_dimension, bias=True)

        # One linear layer Wx + b for labels head, input dim 100 output dim 100
        self.mlp_label_head = nn.Linear(self.hidden_dim, mlp_label_dimension)

        # One linear layer Wx + b for labels dependent, input dim 100 output dim 100
        self.mlp_label_dependent = nn.Linear(self.hidden_dim, mlp_label_dimension)

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
        if (mode == 'predict' or mode == 'test') and sentence is None:
            raise ValueError('Predict mode requires sentence')
        predicted_arcs = None

        # get embeddings for sentence
        embedded_sentence = torch.cat(
            (self.Dropout(self.word_embeddings(sentence_word_indices)),
             self.Dropout(self.pos_embeddings(sentence_pos_indices))), 1)
        sentence_length = len(embedded_sentence)
        inputs = embedded_sentence.view(1, sentence_length, -1)

        # pass through the biLstm layer
        lstm_out = self.Dropout(self.bilstm(inputs)[0])

        # compute head and dependent representations
        R = lstm_out.view(sentence_length, -1)
        H_head = self.Dropout(self.ReLU(self.mlp_arc_head(R)))
        H_dependent = self.Dropout(self.ReLU(self.mlp_arc_dependent(R)))

        # calculate scores on formula H_head * (U1 * H_dep + u2)
        H_dep_transformed = self.transform_H_dependent(H_dependent)
        arc_scores = torch.mm(H_head, torch.transpose(H_dep_transformed, 0, 1))

        L_head = self.Dropout(self.ReLU(self.mlp_label_head(R)))
        L_dependent = self.Dropout(self.ReLU(self.mlp_label_dependent(R)))

        if mode == 'train' or mode == 'validate':
            Ryi = L_head[tuple(heads), ]
        elif mode == 'test' or mode == 'predict':
            root = get_true_root(arc_scores)
            # softmax_arc_scores_matrix = nn.Softmax(dim=1)(arc_scores).data.cpu().numpy()
            arc_scores_matrix = arc_scores.data.cpu().numpy()
            mst = ed.edmonds_list(cost_matrix=arc_scores_matrix[1:, 1:], sentence=sentence[1:], root=root - 1)
            predicted_arcs = np.zeros(len(sentence), dtype=np.int)
            predicted_arcs[root] = 0
            for pair in mst:
                head = pair[0]
                dep = pair[1]
                predicted_arcs[dep + 1] = head + 1

            Ryi = L_head[tuple(predicted_arcs.tolist()), ]
        else:
            raise ValueError('Unknown mode: {}.'.format(mode))

        first_term = self.label_bilinear(Ryi, L_dependent)
        second_term = self.label_transform(torch.cat((Ryi, L_dependent), dim=1))

        label_scores = first_term + second_term

        arc_scores.type(floatTensor)

        if mode in ['train', 'validate', 'test']:
            return arc_scores, label_scores
        elif mode in ['predict']:
            return arc_scores, label_scores, predicted_arcs
