import pylab
import embedding as em


if __name__ == '__main__':
    en_metrics = {
        'average_sent_len': 0,
        'average_dep_len': 0
    }
    en_train_set = em.en_train_sentences()
    count_sentences = len(en_train_set)
    count_words = 0
    for sentence in en_train_set:
        en_metrics['average_sent_len'] += len(sentence.words)
        for index, head in enumerate(sentence.get_head_representation()):
            en_metrics['average_dep_len'] += abs(head - index)
            count_words += len(sentence.get_word_list()[1:])
    en_metrics['average_sent_len'] /= count_sentences
    en_metrics['average_dep_len'] /= count_words

    print(en_metrics['average_sent_len'])
    print(en_metrics['average_dep_len'])
