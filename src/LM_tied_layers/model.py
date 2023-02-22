import os
import math
import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.distributions import Categorical
import pandas as pa
from data import Dataset
from pytorch_mod import *
from random import randint
import warnings



warnings.filterwarnings("ignore")

class LanguageModel(nn.Module):

    def __init__(self,encoder,
                      context_arch,
                      embedding_size,
                      hidden_size,
                      nlayers,
                      nheads=12,
                      ffn_hidden=2048,
                      dropout=0.5,
                      tie_weights=False,
                      tie_layers=False,
                      positional=True,
                      verbose=False):
        """
        Args:
            encoder    (Dataset): a dataset whose str <-> int encodings are used by the model
            context_arch   (str): A string in the set {'RNN','LSTM','GPT'} that specifies the kind of model is used.
            embedding_size (int): the size of the word embeddings
            hidden_size    (int): the size of the hidden layer for LSTM and RNN. For GPT and RNN models with tied weights must be equal to embedding size.
            nlayers        (int): number of layers in the model
            nheads         (int): number of heads used by the GPT model
            ffn_hidden     (int): size of the FFN hidden vector size for GPT models
            dropout      (float): amount of dropout used all around the place
            tie_weights   (bool): whether decoder and encoder share the same parameters
            device         (str): a string specifying the computer device where to store the model for performing computations, typically cpu etc.
            positional     (bool): whether to add positional embeddings or not
        """
        super(LanguageModel, self).__init__()
        if context_arch.upper() == 'LSTM':
            self.context_model = RnnContextModel(encoder,context_arch,embedding_size,hidden_size, nlayers,dropout,'cpu',tie_weights)
        elif context_arch.upper() == 'RNN':
            self.context_model = nn.RNN(encoder,context_arch,embedding_size,hidden_size, nlayers,dropout,'cpu',tie_weights)
        elif context_arch.upper() == 'GPT':
            self.context_model = IncrementalTransformerContextModel(encoder, embedding_size, nlayers, nheads, ffn_hidden, dropout, dropout, 'cpu', tie_weights = tie_weights,tie_layers=tie_layers, positional=positional, verbose=verbose)

    def load_params(self, dirname,cpu=False):
        if cpu:
            self.load_state_dict(torch.load(os.path.join(dirname, 'lm_params.pt'),map_location='cpu'))
        else:
            self.load_state_dict(torch.load(os.path.join(dirname, 'lm_params.pt')))

    def save_params(self,dirname):
        torch.save(self.state_dict(), os.path.join(dirname, 'lm_params.pt'))

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def freeze(self):
        #note that the optimizer should not try to optimize frozen params:
        #call it like this: optimizer = optim.SGD(filter(lambda p: p.requires_grad, lm.parameters()))
        #or freeze the model params *after* the optimizer has been called.
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, input, bptt_state=None,raw_out=False):
        if isinstance(self.context_model, IncrementalTransformerContextModel):
            return self.context_model.forward(input,raw_out=raw_out)
        else:
            return self.context_model.forward(input,bptt_state,raw_out)

    def train_model(self,trainset,validset,batch_size,chunk_size,epochs,warmup_epochs,warmup_batch_size,warmup_cycles,logger,devicelist,lr=0.001,grad_clip=1.0,modeldir='.'):
        """
        The training procedure implements Truncated BackProp Through Time (T-BPTT).
        Training requires to take care of a proper gradient descent, but also of limited memory constraints.
        The batch_size and chunk_size can be used to control the amount of memory used during training.
        Batch size is the number of sequences in a batch, and chunk_size is the max number of successive tokens used in a single backprop pass
        Args:
            trainset (DataSet): a dataset object on which to train the model
            validset (DataSet): a datsaset object on which to validate the model
            batch_size   (int): the size of a batch (number of sequences)
            chunk_size   (int): the size of a batch chunk (max number of tokens in the sequences)
            epochs       (int): the number of training epochs
        KwArgs:
            lr         (float): the Adam Learning Rate
            device       (str): the device ID on which to run the computations (typically cuda:int)
            grad_clip  (float): gradient clipping factor
            batch_group  (int): the number of batch to group together for running backprop
            modeldir     (str): directory where to save the params
        """
        #Transformer optimization is distinct from RNN optimization:
        #Transformer optimization implements SGD with warm restarts
        #while RNN is a standard Adam
        #@see https://arxiv.org/pdf/1809.10853.pdf, sec.4.5. for explanations

        init_seed = randint(-10000,10000)
        num_gpus= len(devicelist)
        mp.spawn(train_multi, args=(num_gpus,self,trainset,validset,batch_size,chunk_size,epochs,warmup_epochs,warmup_batch_size,warmup_cycles,init_seed,lr,grad_clip,modeldir),
                 nprocs=num_gpus,
                 join=True)
        self.load_state_dict(torch.load(os.path.join(modeldir,'lm_params.pt')))


    def generate(self,dataencoder,context,max_length=200,bos='<bos>',eos='<eos>',device='cuda'):
        """
        This generates a random text given a query. It stops when it encounters the <eos> token (or a maximal bound)
        Args:
            dataencoder (Dataset): a dataset containing the model's encodings
            context: A list of strings. The context tokens
        KwArgs:
            max_length (int): the maximum length of a generated sequence.
            device     (str): the device where we perform computations
        Returns
            A list of strings (the generated text sequence)
        """
        eos_idx = dataencoder.tok2idx[eos] #Get the index of the eos token
        self.eval() # Set the model to evaluation mode
        with torch.no_grad(): # Disable gradient computation for inference
            # index mapping, convert context tokens to indices
            xcontext    = [dataencoder.tok2idx.get(token, dataencoder.tok2idx[dataencoder.unk_token]) for token in [bos]+context]
            gseq        = xcontext
            # converts xcontext to a tensor with a fake batch dimension. len==7 becomes size (7,1)
            xcontext    = torch.LongTensor(xcontext).unsqueeze(1).to(device) #adds a fake batch dimension -> (seq,batch)
            if isinstance(self.context_model,RnnContextModel):
                Yhat,_      = self.forward(xcontext)
            elif isinstance(self.context_model, IncrementalTransformerContextModel):
                Yhat        = self.forward(xcontext) # (n,1,vocab_size)
            # creates a Categorical distribution with the last item in Yhat
            # Yhat[-1] take the last row; view(-1) convert it to a 1-D tensor,
            # #-1 means that the size of the 1-D tensor will be inferred from the size of the original tensor.
            # F.softmax() rescales the 1-D tensor into probability distribution
            dist      = Categorical(probs=F.softmax(Yhat[-1].view(-1)))
            # samples the next word from the distribution.
            # generates a random sample from the categorical distribution
            # item() method is then used to convert this tensor to a Python integer.
            next_word = dist.sample().item()
            gseq.append(next_word)

            for _ in range(max_length-1):
                # checks if the next word is the eos token,if so breaks out of the loop
                if next_word == eos_idx:
                    break
                if isinstance(self.context_model, IncrementalTransformerContextModel):
                    xinput =  torch.LongTensor(gseq).unsqueeze(1).to(device)
                    Yhat = self.forward(xinput)
                elif isinstance(self.context_model,RnnContextModel):
                    xinput = torch.LongTensor([[next_word]]).to(device)
                    Yhat,_ = self.forward(xinput)
                dist = Categorical(probs=F.softmax(Yhat.view(-1)))
                next_word = dist.sample().item()
                gseq.append(next_word)
        # converts the token indexes in gseq to tokens
        return [dataencoder.idx2tok[tok_idx] for tok_idx in gseq]

    def representations(self,datagenerator,batch_size,device='cuda'):
        for (xinput, youtput, first) in datagenerator.generate_batch(batch_size, keep_order=True):
            X = torch.LongTensor(xinput).to(device)  # (seq,batch,emb)
            if isinstance(self.context_model, RnnContextModel):
                Yhat, _ = self.forward(X,raw_out=True)
            elif isinstance(self.context_model, IncrementalTransformerContextModel):
                _, Yhat = self.forward(X,raw_out=True)
            yield X,Yhat

    def predict_top_k(self,datagenerator,batch_size,top_k=3,device='cuda'):
        """
        Returns the model predictions on a text.
        :param datagenerator:
        :param batch_size:
        :param top_k: top 3 by default
        :param device:
        :yield: a decoded batch (as pandas DataFrame), one sentence at a time, where each xtoken is coded as a tuple
                (xtoken,pred_next_token1,pred_next_token2,pred_next_token3) on a dataframe line
        """
        self.eval()
        for (xinput,youtput,first) in datagenerator.generate_batch(batch_size,keep_order=True):
            with torch.no_grad():
                X = torch.LongTensor(xinput).to(device)   # (seq,batch,emb)
                Y = torch.LongTensor(youtput).to(device)  # (seq,batch,emb)
                if isinstance(self.context_model,RnnContextModel):
                    Yhat, _ = self.forward(X)
                    Yhat    = F.log_softmax(Yhat,dim=2)
                elif isinstance(self.context_model, IncrementalTransformerContextModel):
                   Yhat = F.log_softmax(self.forward(X),dim=2)
                Yhat = Yhat.transpose(0,1)                # (batch,seq,emb)
                X    = X.transpose(0,1)                   # (batch,seq,emb)
                Y    = Y.transpose(0,1)                   # (batch,seq,emb)

                (prob_pred,pred_token) = torch.topk(Yhat,3,dim=2)
                prob1, prob2, prob3 = torch.chunk(prob_pred, 3, dim=2)
                token1, token2, token3 = torch.chunk(pred_token, 3, dim=2)
                #prob_ref = torch.gather(Yhat,2,Y.unsqueeze(2))

                #result has dim (batch_size,seq_len,5columns)
                result = torch.stack((X.float(),Y.float(),token1.squeeze(dim=2).float(),token2.squeeze(dim=2).float(),token3.squeeze(dim=2).float()),dim=2)
                #decoding the results on strings
                result = result.to('cpu').tolist()
                for sentence in result:
                    for token in sentence:
                        token[0] = datagenerator.idx2tok[int(token[0])]
                        token[1] = datagenerator.idx2tok[int(token[1])]
                        token[2] = datagenerator.idx2tok[int(token[2])]
                        token[3] = datagenerator.idx2tok[int(token[3])]
                        token[4] = datagenerator.idx2tok[int(token[4])]
                    records = [tuple(token) for token in sentence if token[0] != datagenerator.pad_token]
                    yield pa.DataFrame.from_records(records,columns=['token', 'ref_next', 'pred1', 'pred2', 'pred3'])


    def predict(self,datagenerator,batch_size,device='cuda'):
        """
        Returns the model predictions on a text.
        :param datagenerator:
        :param batch_size:
        :param device:
        :yield: a decoded batch (as pandas DataFrame), one sentence at a time, where each xtoken is coded as a tuple
                (xtoken,ref_next_token,pred_next_token,prob_ref,prob_pred) on a dataframe line
        """
        self.eval()
        for (xinput,youtput,first) in datagenerator.generate_batch(batch_size,keep_order=True):
            with torch.no_grad():
                X = torch.LongTensor(xinput).to(device)   # (seq,batch,emb)
                Y = torch.LongTensor(youtput).to(device)  # (seq,batch,emb)
                if isinstance(self.context_model,RnnContextModel):
                    Yhat, _ = self.forward(X)
                    Yhat    = F.log_softmax(Yhat,dim=2)
                elif isinstance(self.context_model, IncrementalTransformerContextModel):
                   Yhat = F.log_softmax(self.forward(X),dim=2)
                Yhat = Yhat.transpose(0,1)                # (batch,seq,emb)
                X    = X.transpose(0,1)                   # (batch,seq,emb)
                Y    = Y.transpose(0,1)                   # (batch,seq,emb)

                (prob_pred,pred_token) = torch.max(Yhat,dim=2)
                prob_ref = torch.gather(Yhat,2,Y.unsqueeze(2))

                #result has dim (batch_size,seq_len,5columns)
                result = torch.stack((X.float(),Y.float(),pred_token.float(),prob_ref.squeeze(2),prob_pred),dim=2)
                #decoding the results on strings
                result = result.to('cpu').tolist()
                for sentence in result:
                    for token in sentence:
                        token[0] =  datagenerator.idx2tok[int(token[0])]
                        token[1] =  datagenerator.idx2tok[int(token[1])]
                        token[2] =  datagenerator.idx2tok[int(token[2])]
                    records = [tuple(token) for token in sentence if token[0] != datagenerator.pad_token]
                    yield pa.DataFrame.from_records(records,columns=['token', 'ref_next', 'pred_next', 'ref_prob', 'pred_prob'])

    def attention_viz(self,sentence,dataencoder,bos='<bos>',eos='<eos>',device='cuda',candidates=None):
        """
        Returns the model attention matrices for a single sentence, using the BertViz format
        Args:
            sentence (list)             : a list of strings, the sentence
            dataencoder (DataSet)       : a Dataset that contains the string->int mapping
            candidates (list of strings): a list of words for which we want to know the probabilities
        Returns:
            a tuple of attention_weights 4 order tensors (1 x nheads x seq_len x seq_len)
            the there is one element in the tuple for every layer (element 0 is the lowest layer)
        """
        self.eval()
        sentence = [bos] + sentence
        with torch.no_grad():
            xcontext = [dataencoder.tok2idx.get(token, dataencoder.tok2idx[dataencoder.unk_token]) for token in sentence]
            X        = torch.LongTensor(xcontext).unsqueeze(1).to(device)
            Yhat     = F.log_softmax(self.forward(X), dim=2)
        if candidates:
            logprobs = [(candidate,Yhat[-1,0,dataencoder.tok2idx.get(candidate,dataencoder.tok2idx[dataencoder.unk_token])].item()) for candidate in candidates]
            return tuple(layer.attn_weights for layer in self.context_model.transformer_encoder.layers), sentence, logprobs
        else:
            return tuple(layer.attn_weights for layer in self.context_model.transformer_encoder.layers), sentence

    def validate(self, datagenerator, batch_size, chunk_size, device='cuda'):
        """
        Returns the loss and perplexity on a validation set.
        The perplexity does not take into account unk symbols.
        :param datagenerator:
        :param batch_size:
        :param device:
        :return:
        """
        self.eval()
        self.to(device)
        criterion    = nn.CrossEntropyLoss(reduction='none')
        total_loss   = 0.
        total_tokens = 0. #totals the number of true tokens in the dataset (here we remove <pad> tokens but not <eos>)

        for (xinput, youtput,first) in datagenerator.generate_batch(batch_size,bptt_len=chunk_size,worker_id=device):
            with torch.no_grad():
                if first:
                    bptt_state = RnnContextModel.zero_bptt()

                X = torch.LongTensor(xinput).to(device)   # (seq,batch,emb)
                Y = torch.LongTensor(youtput).to(device)  # (seq,batch,emb)
                seq_len, batch_len = Y.shape
                if isinstance(self.context_model, IncrementalTransformerContextModel):
                    Yhat = self.forward(X).view(batch_len * len(X), -1)
                elif isinstance(self.context_model, RnnContextModel):
                    Yhat, bptt_state = self.forward(X, bptt_state)
                    Yhat = Yhat.view(batch_len * len(X), -1)
                Y = Y.view(batch_len * len(Y))
                loss = criterion(Yhat, Y)

                # masking special tokens for metrics computation
                unk_mask = (Y != datagenerator.unk_idx)
                pad_mask = (Y != datagenerator.pad_idx)
                loss_mask = unk_mask * pad_mask
                total_loss += (loss_mask * loss).sum().item()
                total_tokens += loss_mask.sum().item()

        ppl = math.exp(total_loss/total_tokens)
        return (total_loss/total_tokens, ppl)

def cosine_scheduler(optimizer,warmup_steps,training_steps,ncycles):

    def warmup(x):
        return (x / warmup_steps)

    def cosine(x, nsteps=1):
        return 0.5*(math.cos(math.pi*x / nsteps) + 1)

    lr_list = [warmup(i) for i in range(1,warmup_steps+1)]
    for _ in range(ncycles):
        steps = int( training_steps/ncycles )
        lr_list.extend([cosine(i,steps) for i in range(0,steps)])
    return optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=lambda x: lr_list[x] if x < len(lr_list) else lr_list[-1])

def train_multi(gpu,num_gpu,model,trainset,validset,batch_size,chunk_size,epochs,warmup_epochs,warmup_batch_size,warmup_cycles,init_seed,lr,grad_clip,modeldir):
        """
        The training procedure implements Truncated BackProp Through Time (T-BPTT).
        Training requires to take care of a proper gradient descent, but also of limited memory constraints.
        The batch_size and chunk_size can be used to control the amount of memory used during training.
        Batch size is the number of sequences in a batch, and chunk_size is the max number of successive tokens used in a single backprop pass
        Args:
            trainset (DataSet): a dataset object on which to train the model
            validset (DataSet): a datsaset object on which to validate the model
            batch_size   (int): the size of a batch (number of sequences)
            chunk_size   (int): the size of a batch chunk (max number of tokens in the sequences)
            epochs       (int): the number of training epochs
        KwArgs:
            lr         (float): the Adam Learning Rate
            device       (str): the device ID on which to run the computations (typically cuda:int)
            grad_clip  (float): gradient clipping factor
            modeldir     (str): directory where to save the params
        """
        #Transformer optimization is distinct from RNN optimization:
        #Transformer optimization implements SGD with warm restarts
        #while RNN is a standard Adam
        #@see https://arxiv.org/pdf/1809.10853.pdf, sec.4.5. for explanations

        dist.init_process_group("nccl", rank=gpu, world_size=num_gpu)
        model=model.to(gpu)
        parallel_model = DDP(model.context_model, device_ids=[gpu],output_device=gpu)
        criterion = nn.CrossEntropyLoss(ignore_index=trainset.pad_idx,reduction='none')
        min_ppl = 100000
        if isinstance(model.context_model, IncrementalTransformerContextModel):
            optimizer = optim.SGD(parallel_model.parameters(), lr,momentum=0.99,nesterov=True)
            scheduler = cosine_scheduler(optimizer,
                                         warmup_epochs * trainset.num_batches(warmup_batch_size,bptt_len=chunk_size,world_size=num_gpu),
                                         epochs        * trainset.num_batches(batch_size,bptt_len=chunk_size,world_size=num_gpu),
                                         warmup_cycles)
        else:
            optimizer = optim.Adam(parallel_model.parameters(),lr)
            scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer,lr_lambda=lambda x:1)

        for e in range(epochs+warmup_epochs):
            parallel_model.train()
            total_loss   = torch.tensor([0.]).to(gpu)
            total_tokens = torch.tensor([0.]).to(gpu)
            eseed        = init_seed + e
            pbar         = None

            cbatch_size = batch_size
            if isinstance(model.context_model, IncrementalTransformerContextModel):
                cbatch_size = batch_size if e >= warmup_epochs else warmup_batch_size
            nbatches = trainset.num_batches(cbatch_size, bptt_len=chunk_size, world_size=num_gpu)
            for (xinput, youtput,first) in trainset.generate_batch(cbatch_size, init_seed=eseed, worker_id=gpu,world_size=num_gpu,bptt_len=chunk_size,keep_order = False):
                parallel_model.zero_grad()
                if first:
                    bptt_state = RnnContextModel.zero_bptt()

                X = torch.LongTensor(xinput).to(gpu)  #(seq,batch,emb)
                Y = torch.LongTensor(youtput).to(gpu) #(seq,batch,emb)
                seq_len, batch_len = Y.shape
                if isinstance(model.context_model, IncrementalTransformerContextModel):
                    optimizer.zero_grad()
                    tmp = parallel_model.forward(X)
                    Yhat = tmp.view(batch_len * len(X), -1)
                elif isinstance(model.context_model,RnnContextModel):
                    bptt_state = RnnContextModel.truncate_backprop(bptt_state)
                    Yhat,bptt_state = parallel_model.forward(X,bptt_state)
                Yhat = Yhat.view(batch_len * len(X), -1)
                Y = Y.view(batch_len * len(Y))
                loss = criterion(Yhat, Y)
                loss.sum().backward()

                # masking special tokens for metrics computation
                unk_mask      = (Y != trainset.unk_idx)
                pad_mask      = (Y != trainset.pad_idx)
                loss_mask     = unk_mask * pad_mask
                total_loss   += (loss_mask * loss).sum().item()
                total_tokens += loss_mask.sum().item()

                #update
                torch.nn.utils.clip_grad_norm_(parallel_model.parameters(), grad_clip)
                optimizer.step()
                scheduler.step()

                if dist.get_rank() == 0:
                    if not pbar:
                        pbar = tqdm.tqdm(total = nbatches * dist.get_world_size(),ncols=80)
                    pbar.update(dist.get_world_size())

            torch.distributed.all_reduce(total_loss)
            torch.distributed.all_reduce(total_tokens)
            if dist.get_rank() == 0:
                pbar.close()
                print('Epoch %d'%(e+1))
                nll = (total_loss.item() / total_tokens.item())
                ppl = math.exp(nll)
                print('  train mean NLL = %.5f   train ppl = %.5f   learning rate : %.8f'%(nll,ppl,scheduler.get_lr()[0]))
                (vloss,vppl) = model.validate(validset,batch_size,chunk_size,gpu)
                print('  valid mean NLL = %.5f   valid ppl = %.5f'%(vloss,vppl))
                if vppl < min_ppl:
                    min_ppl = vppl
                    torch.save(model.state_dict(), os.path.join(modeldir,'lm_params.pt'))
                if e%10 == 0:
                    torch.save(model.state_dict(), os.path.join(modeldir, 'params_'+str(e)+'.pt'))

class RnnContextModel(nn.Module):

    def __init__(self,encoder,arch,embedding_size,hidden_size,nlayers,dropout,device,tie_weights=True):

        super(RnnContextModel, self).__init__()

        self._embedding_size = embedding_size
        self.encoder = nn.Embedding(encoder.vocab_size(), embedding_size, padding_idx=encoder.pad_idx).to(device)
        self.decoder = nn.Linear(hidden_size, encoder.vocab_size()).to(device)

        if arch.upper() == 'LSTM':
            self.model = nn.LSTM(embedding_size, hidden_size, nlayers, dropout=dropout).to(device)
        elif arch.upper() == 'RNN':
            self.model = nn.RNN(embedding_size,hidden_size,nlayers,dropout=dropout).to(device)
        else:
            raise ValueError('Unknown model specified',arch)
        if tie_weights:
            if embedding_size != hidden_size:
                raise ValueError('When using the tied flag, LSTM or RNN embedding size must be equal to its hidden size')
            self.decoder.weight = self.encoder.weight

        self.drop = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
            initrange = 0.01
            nn.init.uniform_(self.encoder.weight, -initrange, initrange)
            nn.init.zeros_(self.decoder.weight)
            nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    @property
    def embedding_size(self):
        return self._embedding_size

    @staticmethod
    def zero_bptt():
        return None

    @staticmethod
    def truncate_backprop(bptt_state):
        if type(bptt_state) == tuple and bptt_state is not None:
            (h, c) = bptt_state
            return (h.detach(), c.detach())
        elif bptt_state is not None:
            return bptt_state.detach()
        else:
            return bptt_state

    def forward(self,input,bptt_state,raw_out=False):
        emb = self.drop(self.encoder(input))
        output, last_state = self.model.forward(emb, bptt_state)
        output = self.drop(output)
        if raw_out:
            return output, last_state
        else:
            return self.decoder(output),last_state

class IncrementalTransformerContextModel(nn.Module):

    def __init__(self,data_encoder,embedding_size,nlayers,nheads,ffn_hidden_size,positional_dropout,layer_dropout,device,tie_weights=True,tie_layers=False,positional=True,verbose=False):

        super(IncrementalTransformerContextModel, self).__init__()
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(embedding_size, device,positional_dropout)
        self.encoder = nn.Embedding(data_encoder.vocab_size(), embedding_size, padding_idx=data_encoder.pad_idx).to(device)
        if verbose:
            encoder_layers = TransformerEncoderLayerMod(embedding_size, nheads, ffn_hidden_size, layer_dropout).to(device)
        else:
            encoder_layers = TransformerEncoderLayer(embedding_size, nheads, ffn_hidden_size, layer_dropout).to(device)
        if tie_layers:
            self.transformer_encoder = TiedTransformerEncoder(encoder_layers, nlayers).to(device)
        else:
            self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers).to(device)
        self._embedding_size = embedding_size
        self.decoder       = nn.Linear(embedding_size, data_encoder.vocab_size()).to(device)
        self.drop = nn.Dropout(layer_dropout)
        if tie_weights:
            self.decoder.weight = self.encoder.weight
        self.init_weights()
        self.positional = positional
        self.verbose    = verbose
        if not self.positional:
            print('I will ignore positional embeddings')
        else:
            print('I will use positional embeddings')

    @property
    def embedding_size(self):
        return self._embedding_size

    def init_weights(self):
        initrange = 0.001
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        #nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self,xinput,has_mask=True,raw_out=False):
        """
        :param xinput: a tensor with shape (seq, batch, emb)
        :return: a transformed tensor with shape (seq, batch, emb)
        """
        xinput = self.drop(self.encoder(xinput))
        if has_mask:
            device = xinput.device
            if self.src_mask is None or self.src_mask.size(0) != len(xinput):
                mask = self._generate_square_subsequent_mask(len(xinput)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        if self.positional:
            xinput = xinput * math.sqrt(self.embedding_size)
            xinput = self.pos_encoder.forward(xinput)

        raw_output = self.transformer_encoder(xinput, self.src_mask)
        output = self.decoder(self.drop(raw_output))
        if raw_out:
            return (output,raw_output)
        else:
            return output



class TiedTransformerEncoder(nn.Module):
    """
    This is a variant of the pytorch TransformerLayer where weights are tied accross layers
    Usage is exactly the same as the pytorch library class
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TiedTransformerEncoder, self).__init__()
        self.layer = encoder_layer
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask = None, src_key_padding_mask = None):
        r"""Pass the input iteratively through the encoder layer.
        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src
        for _ in range(self.num_layers):
            output = self.layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)
        return output


#class PositionalEncoding(nn.Module):

#    def __init__(self, d_model, device, dropout=0.1, max_len=5000):
#        """
#        Args:
#            d_model: the embed dim (required).
#            dropout: the dropout value (default=0.1).
#            max_len: the max. length of the incoming sequence (default=5000).
#        """
#        super(PositionalEncoding, self).__init__()
#        self.dropout = nn.Dropout(p=dropout)
#
#        pe = torch.zeros(max_len, d_model,device=device)
#        position = torch.arange(0, max_len, dtype=torch.float,device=device).unsqueeze(1)
#        div_term = torch.exp(torch.arange(0, d_model, 2,device=device).float() * (-math.log(10000.0) / d_model))
#        pe[:, 0::2] = torch.sin(position * div_term)
#        pe[:, 1::2] = torch.cos(position * div_term)
#        pe = pe.unsqueeze(0).transpose(0, 1)
#        self.register_buffer('pe', pe)

#    def forward(self, x):
#        """Inputs of forward function
#        Args:
#            x: the sequence fed to the positional encoder model (required).
#        """
#        x = x + self.pe[:x.size(0),:]
#        return self.dropout(x)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, device, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_parameter('pe', nn.Parameter(pe, requires_grad=False))

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


if __name__ == '__main__':

     trainset = Dataset('wiki.train.tokens',max_vocab_size=50000)
     validset = Dataset('wiki.valid.tokens',unk=trainset.unk_token, parentencoding=trainset.encoding)
     #trainset.save('wiki103')
     testset = Dataset('wiki.test.tokens', parentencoding='wiki103')
     lm = LanguageModel(trainset,'LSTM',768,768,1,nheads=12,ffn_hidden=2048, dropout=0.1, tie_weights=True,device='cuda:1')
     lm.train_model(trainset,validset,32,100,15,device='cuda:1',modeldir='wiki103')
     #lm.load_params('wiki103')
     pa.set_option('display.max_rows', None)
     #for df in lm.predict(testset,16,200,device='cuda:1'):
     #   print(df)
     #   print()
