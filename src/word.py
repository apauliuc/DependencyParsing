class Word:
    """Word class that maps all details for a word from a *.conllu file. """

    ROOT_TAG = "<ROOT_TAG>"
    ROOT_WORD = "<ROOT_WORD>"
    ROOT_LABEL = "<ROOT_LABEL>"
    UNKNOWN_WORD = "<UNK_WORD>"
    UNKNOWN_TAG = "<UNK_TAG>"
    UNKNOWN_LABEL = "<UNK_LABEL>"

    def __init__(self, ID, FORM, LEMMA, UPOSTAG, XPOSTAG, FEATS, HEAD, DEPREL, DEPS, MISC):
        self.ID = ID  # Word idx, int starting at 1 for each new sentence; or a range for multiword tokens
        self.FORM = FORM  # Word form or punctuation symbol.
        self.LEMMA = LEMMA  # Lemma or stem of word form.
        self.UPOSTAG = UPOSTAG  # Universal part-of-speech tag.
        self.XPOSTAG = XPOSTAG  # Language-specific part-of-speech tag; underscore if not available.
        self.FEATS = FEATS  # List of morphological features from the universal feature inventory
        self.HEAD = HEAD  # Head of the current word, which is either a value of ID or zero (0).
        self.DEPREL = DEPREL  # Univ. dep. rel to the HEAD (root iff HEAD = 0) or a defined lang-specific subtype
        self.DEPS = DEPS  # Enhanced dependency graph in the form of a list of head-deprel pairs.
        self.MISC = MISC  # Any other annotation.

    def __str__(self):
        string = ''

        string += self.ID if self.ID is not None else '_'
        string += " "
        string += self.FORM  # if self.FORM is not None else '_'   -> Words can not be None. Here, '_' simply means an underscore.
        string += " "
        string += self.LEMMA  # if self.LEMMA is not None else '_'  -> Words can not be None. Here, '_' simply means an underscore.
        string += " "
        string += self.UPOSTAG if self.UPOSTAG is not None else '_'
        string += " "
        string += self.XPOSTAG if self.XPOSTAG is not None else '_'
        string += " "
        string += self.FEATS if self.FEATS is not None else '_'
        string += " "
        string += self.HEAD if self.HEAD is not None else '_'
        string += " "
        string += self.DEPREL if self.DEPREL is not None else '_'
        string += " "
        string += self.DEPS if self.DEPS is not None else '_'
        string += " "
        string += self.MISC if self.MISC is not None else '_'

        return string

    @staticmethod
    def from_line(line):
        tokens = line.split()

        assert len(tokens) == 10

        ID = tokens[0] if tokens[0] != '_' else None
        FORM = tokens[1]  # if tokens[1] != '_' else None
        LEMMA = tokens[2]  # if tokens[2] != '_' else None
        UPOSTAG = tokens[3] if tokens[3] != '_' else None
        XPOSTAG = tokens[4] if tokens[4] != '_' else None
        FEATS = tokens[5] if tokens[5] != '_' else None
        HEAD = tokens[6] if tokens[6] != '_' else None
        DEPREL = tokens[7] if tokens[7] != '_' else None
        DEPS = tokens[8] if tokens[8] != '_' else None
        MISC = tokens[9] if tokens[9] != '_' else None

        return Word(ID, FORM, LEMMA, UPOSTAG, XPOSTAG, FEATS, HEAD, DEPREL, DEPS, MISC)


if __name__ == '__main__':
    test_text = "4	proves	prove	VERB	VBZ	Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin	0	root	0:root	_"
    test_word = Word.from_line(test_text)
    print(test_word)
    print(test_word.FORM)
