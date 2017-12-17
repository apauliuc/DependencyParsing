import argparse
import time
import math
import numpy as np

import torch
from torch import nn
from torch import autograd
from torch.optim import lr_scheduler

from helpers import train_model, validate_model, test_model, predict, save_checkpoint
import embedding as em
from BiLSTM import BiLSTMTagger, NUM_EPOCHS, NUM_LAYERS, HIDDEN_DIMENSION, INPUT_SIZE, LEARNING_RATE, MLP_ARC_OUTPUT, \
    MLP_LABEL_OUTPUT

import logging

logging.basicConfig(filename='../resources/logs/training_{}.log'.format(time.strftime('%d-%m-%Y%_H:%M:%S')),
                    level=logging.DEBUG)

LATEST_CHECKPOINT_RELATIVE_PATH = ''
BEST_CHECKPOINT_RELATIVE_PATH = ''
RESULTS_RELATIVE_PATH = ''
FORMATTED_TEST_RELATIVE_PATH = ''

if torch.cuda.is_available():
    CUDA = True
    floatTensor = torch.cuda.FloatTensor
    longTensor = torch.cuda.LongTensor
else:
    CUDA = False
    floatTensor = torch.FloatTensor
    longTensor = torch.LongTensor


def set_paths():
    global LATEST_CHECKPOINT_RELATIVE_PATH
    LATEST_CHECKPOINT_RELATIVE_PATH = '../resources/checkpoints/{}/latest_checkpoint.tar'.format(args.language)
    global BEST_CHECKPOINT_RELATIVE_PATH
    BEST_CHECKPOINT_RELATIVE_PATH = '../resources/checkpoints/{}/best_checkpoint.tar'.format(args.language)
    global RESULTS_RELATIVE_PATH
    RESULTS_RELATIVE_PATH = '../resources/results/{}-ud-predict.conllu'.format(args.language)
    global FORMATTED_TEST_RELATIVE_PATH
    FORMATTED_TEST_RELATIVE_PATH = '../resources/results/{}-ud-formatted_test.conllu'.format(args.language)


def load_model(a):
    word_embeddings = autograd.Variable(
        torch.from_numpy(np.array(em.word_embeddings(a.language), dtype=np.float))).type(
        floatTensor)
    pos_embeddings = autograd.Variable(
        torch.from_numpy(np.array(em.tag_embeddings(a.language), dtype=np.float))).type(
        floatTensor)
    loaded_model = BiLSTMTagger(input_size=INPUT_SIZE, hidden_dim=HIDDEN_DIMENSION, num_layers=NUM_LAYERS,
                                mlp_arc_dimension=MLP_ARC_OUTPUT, mlp_label_dimension=MLP_LABEL_OUTPUT,
                                n_labels=len(em.i2l[a.language].keys()),
                                word_embeddings=word_embeddings, pos_embeddings=pos_embeddings)
    loaded_optimizer = torch.optim.Adam(loaded_model.parameters(), lr=LEARNING_RATE, weight_decay=1e-6,
                                        betas=(0.9, 0.9))
    loaded_scheduler = lr_scheduler.ReduceLROnPlateau(loaded_optimizer, mode='min', patience=5, verbose=True)
    loaded_losses = {
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
    loaded_loss_function = nn.CrossEntropyLoss()
    if CUDA:
        loaded_loss_function.cuda()

    if a.mode == 'resume':
        if CUDA:
            checkpoint = torch.load(LATEST_CHECKPOINT_RELATIVE_PATH)
        else:
            checkpoint = torch.load(LATEST_CHECKPOINT_RELATIVE_PATH, map_location=lambda storage, loc: storage)
        loaded_model.load_state_dict(checkpoint['model'])
        loaded_optimizer.load_state_dict(checkpoint['optimizer'])
        loaded_model.word_embeddings.weight = nn.Parameter(
            torch.from_numpy(np.array(checkpoint['word_embeddings'], dtype=np.float))).type(floatTensor)
        loaded_model.pos_embeddings.weight = nn.Parameter(
            torch.from_numpy(np.array(checkpoint['pos_embeddings'], dtype=np.float))).type(floatTensor)
        loaded_losses = checkpoint['losses']
    elif a.mode == 'test' or a.mode == 'predict':
        if CUDA:
            checkpoint = torch.load(LATEST_CHECKPOINT_RELATIVE_PATH)
        else:
            checkpoint = torch.load(LATEST_CHECKPOINT_RELATIVE_PATH, map_location=lambda storage, loc: storage)
        loaded_model.load_state_dict(checkpoint['model'])
        loaded_optimizer.load_state_dict(checkpoint['optimizer'])
        loaded_model.word_embeddings.weight = nn.Parameter(
            torch.from_numpy(checkpoint['word_embeddings'])).type(floatTensor)
        loaded_model.pos_embeddings.weight = nn.Parameter(
            torch.from_numpy(checkpoint['pos_embeddings'])).type(floatTensor)
        loaded_losses = checkpoint['losses']

    return loaded_model, loaded_losses, loaded_optimizer, loaded_scheduler, loaded_loss_function


def load_data(a):
    loaded_conllu_sentences = {}
    if a.language == 'en':
        loaded_conllu_sentences['en'] = {}
        if a.mode in ['start', 'resume']:
            loaded_conllu_sentences['en']['train'] = em.en_train_sentences()[0:50]
            loaded_conllu_sentences['en']['dev'] = em.en_dev_sentences()[0:9]
        elif a.mode in ['test', 'predict']:
            loaded_conllu_sentences['en']['test'] = em.en_train_sentences()[0:2]
    elif a.language == 'ro':
        if a.mode in ['start', 'resume']:
            loaded_conllu_sentences['ro']['train'] = em.ro_train_sentences()
            loaded_conllu_sentences['ro']['dev'] = em.ro_dev_sentences()
        elif a.mode in ['test', 'predict']:
            loaded_conllu_sentences['ro']['test'] = em.ro_test_sentences()
    else:
        raise ValueError('Specified language {} is not supported.'.format(a.language))

    return loaded_conllu_sentences


if __name__ == '__main__':
    # parse command-line arguments
    parser = argparse.ArgumentParser(description='Start anew or resume training')
    parser.add_argument('-m', '--mode', type=str, choices=['start', 'resume', 'test', 'predict'], required=True,
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

    # set paths for checkpoints, and results
    set_paths()

    # initialize a new model if starting, load latest if validating, load best if predicting/testing
    model, losses, optimizer, scheduler, loss_function = load_model(args)

    # load data as dictionary depending on language and mode, prevent useless loading which takes time
    conllu_sentences = load_data(args)

    train_loss = math.inf
    validate_loss = math.inf
    if args.mode == 'start' or args.mode == 'resume':
        for epoch in range(NUM_EPOCHS):
            print("Epoch [%d/%d]..." % (epoch + 1, NUM_EPOCHS))
            logging.info("Epoch [%d/%d]..." % (epoch + 1, NUM_EPOCHS))

            is_best_model = False

            # train
            train_loss, train_arc_scores, train_label_scores = train_model(model, optimizer, loss_function,
                                                                           conllu_sentences[args.language]['train'],
                                                                           args.language)
            # validate
            validate_loss, validate_arc_scores, validate_label_scores = validate_model(model, loss_function,
                                                                                       conllu_sentences[args.language][
                                                                                           'dev'],
                                                                                       args.language)
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

            # always save latest checkpoint after an epoch, and flag if best checkpoint
            if epoch + 1 % 10 == 0:
                model.cpu()
                save_checkpoint({
                    'epoch': epoch + 1,
                    'model': model.state_dict(),
                    'losses': losses,
                    'word_embeddings': model.word_embeddings.weight.data.numpy(),
                    'pos_embeddings': model.cpu().pos_embeddings.weight.data.numpy(),
                    'optimizer': optimizer.state_dict(),
                }, LATEST_CHECKPOINT_RELATIVE_PATH, BEST_CHECKPOINT_RELATIVE_PATH, is_best_model)
                if CUDA:
                    model.cuda()

            if validate_loss > losses['validate']['min']['value'] and epoch - losses['validate']['min']['epoch'] > 10:
                print('Ten epochs with no improvement have passed. Stopping training...')
                logging.info('Ten epochs with no improvement have passed. Stopping training...')

                break

        print('Finished training at {}.'.format(time.strftime('%d-%m-%Y, %H:%M:%S')))
        logging.info('Finished training at {}.'.format(time.strftime('%d-%m-%Y, %H:%M:%S')))
    elif args.mode == 'test':
        test_loss, test_arc_scores, test_label_scores = test_model(model, loss_function,
                                                                   conllu_sentences[args.language]['test'],
                                                                   args.language)
        print(test_loss)
    elif args.mode == 'predict':
        prediction_file = open(RESULTS_RELATIVE_PATH, mode='a', encoding='UTF-8')
        formatted_test_file = open(FORMATTED_TEST_RELATIVE_PATH, mode='a', encoding='UTF-8')
        for conllu_sentence in conllu_sentences[args.language]['test']:
            # save formatted version fo test file
            formatted_test_file.write(str(conllu_sentence))
            formatted_test_file.flush()
            # predict arc scores and labels
            predicted_arcs, predicted_labels = predict(model, conllu_sentence, args.language)
            # generate predicted sentence
            for word in conllu_sentence.words:
                word.HEAD = str(predicted_arcs[int(word.ID)])
                word.DEPREL = str(em.i2l[args.language][predicted_labels[int(word.ID)]])
                word.DEPS = word.HEAD + ':' + word.DEPREL

            prediction_file.write(str(conllu_sentence))
            prediction_file.flush()

            # DEBUG
            # print([em.i2l[l] for l in np.argmax(nn.Softmax()(train_label_scores).data.numpy(), axis=1)])
            # print([em.i2l[l] for l in np.argmax(nn.Softmax()(validate_label_scores).data.numpy(), axis=1)])
            # print(conllu_sentences_train.get_label_list())
            # plot_matrix(nn.Softmax()(train_label_scores))
            # plot_matrix(nn.Softmax()(validate_label_scores))
            # plot_matrix(nn.Softmax()(train_arc_scores.permute(1, 0)).permute(1, 0))
            # plot_matrix(nn.Softmax()(validate_arc_scores.permute(1, 0)).permute(1, 0))
