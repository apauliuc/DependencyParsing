from word import Word

import numpy as np


class Sentence:
    """Sentence class that maps all details for a sentence from a *.conllu file. """

    ROOT_TAG = Word.ROOT_TAG
    ROOT_WORD = Word.ROOT_WORD
    ROOT_LABEL = Word.ROOT_LABEL
    UNKNOWN_WORD = Word.UNKNOWN_WORD
    UNKNOWN_TAG = Word.UNKNOWN_TAG
    UNKNOWN_LABEL = Word.UNKNOWN_LABEL

    def __init__(self, newdoc_id, send_id, text, words):
        self.new_doc_id = newdoc_id
        self.send_id = send_id
        self.text = text
        self.words = words

    def __str__(self):
        string = ''
        if self.new_doc_id is not None:
            string += "# newdoc id = "
            string += str(self.new_doc_id)
            string += "\n"
        string += "# send_id = "
        string += str(self.send_id)
        string += "\n"
        string += "# text = "
        string += str(self.text)
        string += "\n"
        sentence_length = len(self.words)
        for index, word in enumerate(self.words):
            string += str(word)
            if index < sentence_length - 1:
                string += "\n"
        return string

    @staticmethod
    def from_lines(lines):
        new_doc_id = ''
        send_id = ''
        text = ''
        words = []

        for line in lines:
            if line.startswith("#"):  # misc properties
                prop = line.split("=")[1]
                if line.startswith("newdoc", 2):
                    new_doc_id = prop
                    continue
                if line.startswith("sent_id", 2):
                    send_id = prop
                    continue
                if line.startswith("text", 2):
                    text = prop
                    continue
                continue
            else:  # words
                if line.split()[0].isdigit():  # index is an integer
                    words.append(Word.from_line(line))

        return Sentence(new_doc_id, send_id, text, words)

    def get_matrix_representation(self):
        """
        For some info: https://github.com/bastings/nlp1-2017-projects/blob/master/dep-parser/adjacency.gif
        """
        matrix = np.zeros((len(self.words) + 1, len(self.words) + 1), dtype=np.int)
        for word in self.words:
            matrix[int(word.HEAD)][int(word.ID)] = 1
        matrix[0][0] = 1
        return matrix

    def get_head_representation(self):
        """
        For some info: https://github.com/bastings/nlp1-2017-projects/blob/master/dep-parser/adjacency.gif
        :return: returns the true distributions; this is used by the torch.nn.CrossEntropyLoss.
        """
        list = np.zeros(len(self.words) + 1, dtype=np.int)
        for index, word in enumerate(self.words):
            list[index + 1] = int(word.HEAD)
        list[0] = 0
        return list

    def get_word_list(self):
        sentence = [self.ROOT_WORD]
        for word in self.words:
            sentence.append(word.FORM.lower())
        return sentence

    def get_pos_list(self):
        sentence = [self.ROOT_TAG]
        for word in self.words:
            sentence.append(word.UPOSTAG.lower())
        return sentence


if __name__ == '__main__':
    lines = ["# sent_id = weblog-blogspot.com_gettingpolitical_20030906235000_ENG_20030906_235000-0003",
             "# text = Today's incident proves that Sharon has lost his patience and his hope in peace.",
             "1	Today	today	NOUN	NN	Number=Sing	3	nmod:poss	3:nmod:poss	SpaceAfter=No",
             "2	's	's	PART	POS	_	1	case	1:case	_",
             "3	incident	incident	NOUN	NN	Number=Sing	4	nsubj	4:nsubj	_",
             "4	proves	prove	VERB	VBZ	Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin	0	root	0:root	_",
             "5	that	that	SCONJ	IN	_	8	mark	8:mark	_",
             "6	Sharon	Sharon	PROPN	NNP	Number=Sing	8	nsubj	8:nsubj	_",
             "7	has	have	AUX	VBZ	Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin	8	aux	8:aux	_",
             "8	lost	lose	VERB	VBN	Tense=Past|VerbForm=Part	4	ccomp	4:ccomp	_",
             "9	his	he	PRON	PRP$	Gender=Masc|Number=Sing|Person=3|Poss=Yes|PronType=Prs	10	nmod:poss	10:nmod:poss	_",
             "10	patience	patience	NOUN	NN	Number=Sing	8	obj	8:obj	_",
             "11	and	and	CCONJ	CC	_	13	cc	13:cc	_",
             "12	his	he	PRON	PRP$	Gender=Masc|Number=Sing|Person=3|Poss=Yes|PronType=Prs	13	nmod:poss	13:nmod:poss	_",
             "13	hope	hope	NOUN	NN	Number=Sing	10	conj	10:conj	_",
             "14	in	in	ADP	IN	_	15	case	15:case	_",
             "15	peace	peace	NOUN	NN	Number=Sing	13	nmod	13:nmod	SpaceAfter=No",
             "16	.	.	PUNCT	.	_	4	punct	4:punct	_"]
    sentence = Sentence.from_lines(lines)
    print("1:\n", str(sentence), "\n")
    print("2:\t", sentence.text, "\n")
    print("3:\t", sentence.words[0], "\n")
    print("4:\n", sentence.get_matrix_representation(), "\n")
    print("5:\t", sentence.get_head_representation(), "\n")
    print("6:\t", sentence.get_word_list(), "\n")
    print("7:\t", sentence.get_pos_list(), "\n")
