import os
import json
import torch
from math import floor, ceil
from random import shuffle, seed
from collections import Counter

def read_dataset(filename,bos='<bos>',eos='<eos>'):
    """
    Generator for sentences in a wikitext style corpus
    :param filename: a string
    :param bos: the begin of sentence token
    :param eos: the end of sentence token
    :yields: a list of tokens (strings) one at a time
    """
    istream = open(filename)
    for line in istream:#istream:
        #line = line.split('=')[0]  #wiki103 format
        line = line.strip()
        if line and not line.isspace():
            tokens = [bos] + line.split() #+ [eos] #Gulordava style contains at then end [eos]
            yield tokens
    #istream.close()

def vocabulary(corpus, max_size=None, unk=None,pad=None):
    """
    Generates the encoding string to int and vice-versa
    for a whole corpus.

    The encoding is frequency sensitive: the indexing is a frequency rank
    where most frequent tokens have lowest indices (except for special tokens)

    :param corpus:an iterable of sentences. A sentence is a list of strings
    :param max_size : int max number of elements in the vocab
    :return: a couple. a dict mapping strings to int and a list mapping int to strings
    """
    vocab = Counter()
    for sentence in corpus:
         vocab.update(sentence)

    idx2str = [pad]
    if unk and unk not in vocab:
        idx2str.append(unk)
    idx2str.extend(tok for tok, _ in vocab.most_common(max_size))
    #idx2str.extend(tok for tok,count in vocab.most_common(max_size))
    str2idx = {token:idx for (idx,token) in enumerate(idx2str)}
    print('Vocabulary size = %d'%(len(idx2str)))
    #return (str2idx,idx2str)
    return str2idx, idx2str

def pad(sentence,pad_size,pad_token):

    return sentence + [pad_token] * (pad_size-len(sentence))

class Dataset:

        def __init__(self,filename,bos='<bos>',eos='<eos>',unk='<unk>',parentencoding=None,max_vocab_size=None, mask_stream=None):

            self.sentences = []
            # reads sentences and destructively performs truncation (attempts to avoid memory explosion)
            if filename:
                self.sentences = list(read_dataset(filename, bos, eos))


            if mask_stream:
                # in json all keys are strings, but we want integers (keys are representing indices in a tensor)
                self.masks = [{int(k): v for k, v in json.loads(line).items()} for line in mask_stream]
            else:
                self.masks = None

            if type(parentencoding) == str:
                istream = open(os.path.join(parentencoding, 'tokcodes'))
                unk = istream.readline().strip()
                parentencoding = [line.strip() for line in istream]
                istream.close()
            if type(parentencoding) == list:
                self.pad_token = parentencoding[0]
                self.unk_token = unk
                self.idx2tok = parentencoding
                self.tok2idx = {token:idx for idx,token in enumerate(self.idx2tok)}
            else:
                self.pad_token = '<pad>'
                self.unk_token = unk
                self.tok2idx, self.idx2tok = vocabulary(self.sentences, pad=self.pad_token, unk=self.unk_token, max_size=max_vocab_size)
            self.pad_idx = self.tok2idx[self.pad_token]
            self.unk_idx = self.tok2idx[self.unk_token]

        @property
        def encoding(self):
            return self.idx2tok

        def save(self,dirname):
            """
            Saves the dataset *encoding* to file
            :param dirname:
            :return:
            """
            ostream = open(os.path.join(dirname,'tokcodes'),'w')
            print(self.unk_token,file=ostream) ## write unk_token as the first line of file "tokcodes"
            ostream.write('\n'.join(self.encoding))
            ostream.close()

        def vocab_size(self):
            return len(self.idx2tok)

        def num_batches(self,batch_size,bptt_len=10000,world_size=1):
            """
            An ~expected number of 'physical' batches, this includes duplications of long batches for trunctated backprop.
            A single gpu will process exactly this number of batches during every epoch.

            Args:
                batch_size(int): size of a batch in sentences
            Returns: an int
            """
            B = 0
            N = len(self.sentences)
            idxes = list(range(N))
            idxes.sort(key=lambda x: len(self.sentences[x]))
            for sidx in range(0,N,batch_size):
                eidx = min(sidx + batch_size, N)
                batchlen = max([len(self.sentences[idx]) - 1 for idx in idxes[sidx:eidx]])
                B += ceil(batchlen/bptt_len)
            return floor(B/world_size)


        def init_epoch(self,init_seed,batch_size,keep_order,worker_id,world_size):
            """
            Performs the shuffling and distribution of a dataset in multiprocessing context.
            Args:
                init_seed  (int): a seed to init the random generator with
                batch_size (int): the size of a full batch
            :return:
            """
            seed(init_seed)
            if type(worker_id) == str:
                worker_id =  int(worker_id[-1])
            N = len(self.sentences)
            self.idxes  = list(range(N))
            self.sidxes = list(range(0 + worker_id, N, batch_size * world_size))  # batch start idxes
            if not keep_order:
                shuffle(self.idxes)
                self.idxes.sort(key=lambda x: len(self.sentences[x]))
                shuffle(self.sidxes)

        @staticmethod
        def is_masked(token):
            """
            Returns True if the token is masked (surrounded by brackets)
            :param token:
            :return:
            """
            return token[0] == '[' and token[-1] == ']'

        @staticmethod
        def strip_mask(token):
            """
            Removes the brackets from a token if masked
            :param token:
            :return:
            """
            if Dataset.is_masked(token):
                return token[1:-1]
            else:
                return token


        def generate_batch(self,batch_size,
                           worker_id = 0,
                           world_size = 1,
                           init_seed=0,
                           bptt_len=10000,
                           keep_order=False,
                           incremental=True,
                           masked=False,
                           include_mask=False):
            """
            Generates a batch of data.
            The generator ensures that every GPU receives the exact same number of batches
            This conservative method prevents deadlocks in multiprocessing.

            Args:
                batch_size (int):size of generated batches
                init_seed  (int):random number seed
                keep_order (int):shuffles the data (or not)
                bptt_len   (int):max number of tokens in a given batch chunk
                incremental(bool):shift X, Y positions for incremental language modelling
                masked     (bool): interprets tokens of the form [string] as tokens to mask
            Returns a subset of the data as a triple (X,Y,first)
            where first is true if the batch is the first for a set of sentences and false otherwise
            """
            self.init_epoch(init_seed,batch_size,keep_order,worker_id,world_size)
            nbatches = self.num_batches(batch_size, world_size=world_size, bptt_len=bptt_len)
            start_indexes = iter(self.sidxes)
            B    = 0
            N    = len(self.sentences)

            while True:
               try:
                   start_index = next(start_indexes)
               except StopIteration:
                   start_indexes = iter(self.sidxes)
                   start_index = next(start_indexes)

               end_index = min(start_index + batch_size, N)

               #shift X, Y positions for incremental language modelling
               if incremental:
                   batchlen = max([len(self.sentences[idx]) - 1 for idx in self.idxes[start_index:end_index]])
                   X = [self.sentences[idx][:-1] for idx in self.idxes[start_index:end_index] ]
                   Y = [self.sentences[idx][1:] for idx in self.idxes[start_index:end_index] ]
               else:
                   batchlen = max([len(self.sentences[idx]) for idx in self.idxes[start_index:end_index]])
                   X = [self.sentences[idx] for idx in self.idxes[start_index:end_index]]
                   Y = [self.sentences[idx] for idx in self.idxes[start_index:end_index]]
               #padding
               X = [pad(x,batchlen,pad_token = self.pad_token) for x in X]
               Y = [pad(y,batchlen,pad_token = self.pad_token) for y in Y]

               # generate masks
               masking = None
               if self.masks:
                   from itertools import chain, repeat
                   inf = float("inf")
                   masking = (torch.triu(torch.ones(batchlen, batchlen)) == 1).transpose(0, 1)
                   masking = masking.float().masked_fill(masking == 0, -inf).masked_fill(masking == 1, float(0.0))
                   #import pdb;pdb.set_trace()
                   print(start_index)
                   extra_pos = [self.masks[idx] for idx in self.idxes[start_index:end_index]]
                   assert len(extra_pos) == 1
                   extra_pos = extra_pos[0]
                   for target_word, word_index in chain.from_iterable(zip(repeat(key), extra_pos[key]) for key in extra_pos):
                       target_word = int(target_word)
                       if target_word > masking.shape[0]:
                           print("error: target_word = ",target_word, "masking.shape= ",masking.shape())
                       masking[target_word, word_index] = -inf
               #import pdb;pdb.set_trace()



               #masking
               if masked:
                  mask    = [ [Dataset.is_masked(token) for token in x ] for x in X]
                  X       = [ [Dataset.strip_mask(token) for token in x ] for x in X]
                  Y       = [ [Dataset.strip_mask(token) for token in y ] for y in Y]

               #coding
               xinput  = [ [self.tok2idx.get(token,self.tok2idx[self.unk_token]) for token in x] for x in X]
               youtput = [ [self.tok2idx.get(token,self.tok2idx[self.unk_token]) for token in y] for y in Y]

               #transpose matrices (batch,seq) -> (seq,batch)
               xinput  = list(zip(*xinput))
               youtput = list(zip(*youtput))
               if masked:
                   mask    = list(zip(*mask))


               #truncates long batches and returns
               first = True
               for idx in range(0, batchlen, bptt_len):
                  xchunk = xinput[idx:idx + bptt_len]
                  ychunk = youtput[idx:idx + bptt_len]
                  if masked:
                      mchunk = mask[idx:idx + bptt_len]
                      yield (xchunk,ychunk, mchunk, first)
                  elif include_mask:
                      yield xchunk, ychunk, first, masking
                  else:
                      yield (xchunk, ychunk, first)
                  B += 1
                  first = False
                  if B >= nbatches: # <= exit loop here
                      return


#Example usage
if __name__ == '__main__':
    """
    trainset = Dataset('train96.tokens',max_vocab_size=50000)
    validset = Dataset('test64.tokens',unk=trainset.unk_token, parentencoding=trainset.encoding)
    #testset  = Dataset('wiki.test.tokens',unk=trainset.unk_token,  parentencoding=trainset.encoding)
    for tup in trainset.generate_batch(32,incremental=False,masked=True):
        for idx in range(len(tup[0])) :
            x = tup[0][idx]
            y = tup[1][idx]
            print("x numéro ",idx, [trainset.idx2tok[i] for i in x])
            print ("y numéro ",idx,[trainset.idx2tok[i] for i in y])
            print("mchunk: ",tup[2][idx])
            #print("first: ",tup[3])
        break
    """
    from io import StringIO

    #input_mask = StringIO(json.dumps({7: [1, 2]}))
    input_mask = open("debug_old.json")
    #trainset = Dataset("obj-pp_test.txt",mask_stream=input_mask)
    testdata = Dataset('debug.txt', parentencoding="TM/", mask_stream=input_mask)
    #print(testdata.idx2tok)
    for X in testdata.generate_batch(1,keep_order=True, include_mask=True):
        pass
        #print(X)
        #print(Y)
    input_mask.close()


