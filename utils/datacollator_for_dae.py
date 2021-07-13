import numpy as np
from copy import deepcopy
class DataCollatorForDAE:
    def __init__(self,tokenizer,word_mask=0.15,word_dropout=0.15,word_shuffle=1):
        self.mask_index = tokenizer.mask_token_id
        self.pad_index = tokenizer.pad_token_id
        self.eos_index = tokenizer.sep_token_id
        self.sos_index = tokenizer.cls_token_id
        self.word_mask_factor = word_mask
        self.word_dropout_factor = word_dropout
        self.is_word_shuffle = word_shuffle

    def get_before_pad(self,x):
        '''Obtain length till pad'''
        try:
            return np.argwhere(x==self.pad_index)[0][0]
        except:
            return len(x)

    def word_mask(self, x, l):
        """ Randomly mask input words """
        # define droppable word indices
        if self.word_mask_factor == 0:
            return x

        no_mask = int(l*self.word_mask_factor)
        mask_seq = np.random.randint(2,l,no_mask)
        
        x2 = deepcopy(x)
        for i in mask_seq:
            x2[i] = self.mask_index
        return x2

    def word_dropout(self, x, l):
        """ Randomly drop input words """
        if self.word_dropout_factor == 0:
            return x

        # define droppable word indices
        no_drops = int(l*self.word_dropout_factor)
        # drop_seq = np.random.randint(1,l,no_drops)  #leave start token #gives duplicates so nope
        drop_seq = np.random.choice(range(1,l), no_drops, replace=False)

        x2 = deepcopy(x)
        x2 = np.delete(x2,drop_seq,0)
        # print("dropped",len(x)-len(x2),no_drops,x2,drop_seq,x)
        x2 = np.concatenate([x2,[self.pad_index]*no_drops],0)
        return x2

    def word_shuffle(self, x, l):
        """ Randomly shuffle input words. """
        if self.is_word_shuffle == 0:
            return x
        
        # Choose a subsequence to shuffle
        shuffl_seq = np.random.randint(2,l,2) #leave start token, get start and end of shuffl_seq
        shuffl_seq.sort()

        x2 = deepcopy(x)
        x2 = np.concatenate([x[:shuffl_seq[0]],np.random.permutation(x[shuffl_seq[0]:shuffl_seq[1]]),x[shuffl_seq[1]:]],axis=0)
        return x2

    def add_noise(self, words):
        """
        Add noise to the encoder input.
        """
        length = self.get_before_pad(words)
        # print("init",len(words))
        words = self.word_shuffle(words, length)
        # print("shuffle",len(words))
        words = self.word_dropout(words, length)
        # print("after drop",len(words))
        length = self.get_before_pad(words)
        # print(tokenizer.convert_ids_to_tokens(words),length)
        words = self.word_mask(words, length)
        # print("mask",len(words))
        return np.asarray(words)
    
    def add_noise_dataset(self,ds):
        ds["input_ids"] = self.add_noise(ds["input_ids"])
        return ds