import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import pandas as pa
import torch.nn.functional as F

from model import *


class MLMLanguageModel(nn.Module):

    def __init__(self,encoder,
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
        super(MLMLanguageModel, self).__init__()
        self.context_model = MLMTransformerContextModel(encoder, embedding_size, nlayers, nheads, ffn_hidden,
                                                                dropout, dropout, 'cpu', tie_weights=tie_weights,tie_layers=tie_layers,
                                                                positional=positional, verbose=verbose)

    def load_params(self, dirname,cpu=False):
        if cpu:
            self.load_state_dict(torch.load(os.path.join(dirname, 'lm_params.pt'),map_location='cpu'))
        else:
            self.load_state_dict(torch.load(os.path.join(dirname, 'lm_params.pt')))

    def save_params(self,dirname):
        torch.save(self.state_dict(), os.path.join(dirname, 'lm_params.pt'))

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, input,raw_out=False):
        return self.context_model.forward(input,raw_out)

    def representations(self,datagenerator,batch_size,device='cuda'):
        for (xinput, youtput, first) in datagenerator.generate_batch(batch_size, keep_order=True):
            X = torch.LongTensor(xinput).to(device)  # (seq,batch,emb)
            _,Yhat = self.forward(X,raw_out=True)
            yield X,Yhat

    def predict(self,datagenerator,batch_size=32, device='cuda'):
        """
        Performs predictions on a test set by masking the list of tokens specified in masked_idx
        It outputs the log probabilities of the predicted words for the
        Args:
            datagenerator (DataSet): a test set
            batch_size   (int):
            device       (str):
        Return:
               a pandas DataFrame
        """
        for (X,Y, mask, first) in datagenerator.generate_batch(batch_size,incremental=False,masked=True):

            X = torch.LongTensor(X).to(device)
            #Y = torch.LongTensor(Y).to(device)
            mask = torch.tensor(mask,dtype=torch.bool).to(device)

            X_masked = X.masked_fill(mask,datagenerator.pad_idx)
            Yhat =  F.log_softmax(self.forward(X_masked),dim=2)

            Yhat = Yhat.transpose(0, 1)  # (batch,seq,emb)
            X = X.transpose(0, 1)        # (batch,seq,emb)
            #Y = Y.transpose(0, 1)        # (batch,seq,emb)
            mask = mask.transpose(0,1)   # (batch,seq,emb)
            (prob_pred, pred_token) = torch.max(Yhat, dim=2)
            prob_ref = torch.gather(Yhat, 2, Y.unsqueeze(2))

            # result has dim (batch_size,seq_len,5)
            result = torch.stack((X.float(), mask.float(), pred_token.float(), prob_ref.squeeze(2), prob_pred), dim=2)
            # decoding the results on strings
            result = result.to('cpu').tolist()
            for sentence in result:
                for token in sentence:
                    token[0] = datagenerator.idx2tok[int(token[0])]
                    token[1] = (token[1] == True) #casts to bool
                    token[2] = datagenerator.idx2tok[int(token[2])]
                records = [tuple(token) for token in sentence if token[0] != datagenerator.pad_token]
                yield pa.DataFrame.from_records(records,columns=['token', 'masked', 'pred', 'ref_prob', 'pred_prob'])

            #Convert to numpy and extract the values

    def validate(self, datagenerator, batch_size, chunk_size, device='cuda'):
        """
        Returns the average loss on a validation set.
        Note that validate still performs masking
        :param datagenerator:
        :param batch_size:
        :param device:
        :return:
        """
        self.eval()
        self.to(device)
        criterion = nn.CrossEntropyLoss(reduction='none')
        total_loss = 0.
        total_tokens = 0.  # totals the number of true tokens in the dataset (here we remove <pad> tokens but not <eos>)

        for (xinput, youtput, first) in datagenerator.generate_batch(batch_size, bptt_len=chunk_size, worker_id=device,incremental=False):

            with torch.no_grad():
                X = torch.LongTensor(xinput).to(device)  # (seq,batch,emb)
                Y = torch.LongTensor(youtput).to(device)  # (seq,batch,emb)
                X, Y = MLMTransformerContextModel.apply_MLM_mask(X, Y, datagenerator.pad_idx, datagenerator.vocab_size())
                seq_len, batch_len = Y.shape
                Yhat = self.forward(X).view(batch_len * len(X), -1)
                Yhat = Yhat.view(batch_len * len(X), -1)
                Y = Y.view(batch_len * len(Y))
                loss = criterion(Yhat, Y)

                # masking special tokens for metrics computation
                loss_mask = (Y != datagenerator.pad_idx)
                total_loss += (loss_mask * loss).sum().item()
                total_tokens += loss_mask.sum().item()

        return (total_loss / total_tokens)

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
        mp.spawn(train_MLM,args=(num_gpus,self,trainset,validset,batch_size,chunk_size,epochs,warmup_epochs,warmup_batch_size,warmup_cycles,init_seed,lr,grad_clip,modeldir),
                 nprocs=num_gpus,
                 join=True)
        self.load_state_dict(torch.load(os.path.join(modeldir,'lm_params.pt')))

class MLMTransformerContextModel(nn.Module):

    def __init__(self, data_encoder, embedding_size, nlayers, nheads, ffn_hidden_size, positional_dropout,
                 layer_dropout, device, tie_weights=True,tie_layers=False, positional=True, verbose=False):

        super(MLMTransformerContextModel, self).__init__()
        self.pos_encoder         = PositionalEncoding(embedding_size, device, positional_dropout)
        self.encoder             = nn.Embedding(data_encoder.vocab_size(), embedding_size, padding_idx=data_encoder.pad_idx).to(device)
        encoder_layers           = TransformerEncoderLayer(embedding_size, nheads, ffn_hidden_size, layer_dropout).to(device)
        if tie_layers:
            self.transformer_encoder = TiedTransformerEncoder(encoder_layers, nlayers).to(device)
        else:
            self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers).to(device)
        self._embedding_size = embedding_size
        self.decoder = nn.Linear(embedding_size, data_encoder.vocab_size()).to(device)
        self.drop = nn.Dropout(layer_dropout)
        if tie_weights:
            self.decoder.weight = self.encoder.weight
        self.init_weights()
        self.positional = positional
        self.verbose = verbose

    @property
    def embedding_size(self):
        return self._embedding_size

    def init_weights(self):
        initrange = 0.001
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    #Masking
    @staticmethod
    def apply_MLM_mask(X,Y,pad_idx,vocab_size,mask_prob=0.15,keep_prob=0.1,repl_prob=0.1):
        """
        X and Y are respectively src and target matrices (sequence,batch)
        0.15 words are unmasked in Y
        0.15 words are masked in X, among which:
            0.8 are indeed masked
            0.1 are replaced by a random word
            0.1 are left unmasked # so pretraining is supposed to predict the unmasked word itself
        :param X:
        :param Y:
        :return: X and Y with modified values:masked values are pad_idx valued
        """
        assert (X.shape == Y.shape)
        X,Y = X.float(),Y.float()
        R  = torch.rand_like(X) # creates a tensor with random values that has the same shape as a given input tensor

        #Y masking
        target_mask = (R < mask_prob)        #index is true for words that are to be masked
        Y.masked_fill_(~target_mask,pad_idx) #pad token used for masking # fills the selected word with pad_index

        #X masking
        padding_mask = (X != pad_idx)      #creating a boolean mask: which is true for words that are not padded.
        srcR         = torch.rand_like(X) # creates another random tensor
        keep_masked  = ~(srcR < (keep_prob+repl_prob)) # true for words that are not to be masked
        src_mask     = target_mask & keep_masked & padding_mask #cannot replace padded item # true for words that are to be masked
        X.masked_fill_(src_mask,pad_idx) # fills the selected word with pad_idx

        repl_mask    = (~src_mask) & target_mask & (srcR < repl_prob)  # true for words that are to be replaced by random words
        repl_values  = torch.randint_like(X,vocab_size) # with random integers between 0 and vocab_size
        X.masked_scatter_(repl_mask,repl_values) # replaces the selected word

        return X.long(),Y.long()

    def forward(self,xinput,raw_out=False):
        """
        :param xinput: a tensor with shape (seq, batch, emb)
        :return: a transformed tensor with shape (seq, batch, emb)
        """
        xinput = self.drop(self.encoder(xinput))

        if self.positional:
            xinput = xinput * math.sqrt(self.embedding_size)
            xinput = self.pos_encoder.forward(xinput)

        # Add padding mask here ?
        # Masking is taken into account at the embedding level
        # this would also remove positional embeddings
        raw_output = self.transformer_encoder(xinput)
        output = self.decoder(self.drop(raw_output))

        return (output,raw_output) if raw_out else output

def train_MLM(gpu, num_gpu, model, trainset, validset, batch_size, chunk_size, epochs, warmup_epochs,
                warmup_batch_size, warmup_cycles, init_seed, lr, grad_clip, modeldir):
    """
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
    # Transformer optimization is distinct from RNN optimization:
    # Transformer optimization implements SGD with warm restarts
    # while RNN is a standard Adam
    # @see https://arxiv.org/pdf/1809.10853.pdf, sec.4.5. for explanations

    dist.init_process_group("nccl", rank=gpu, world_size=num_gpu)
    model = model.to(gpu)
    parallel_model = DDP(model.context_model, device_ids=[gpu], output_device=gpu)
    criterion = nn.CrossEntropyLoss(ignore_index=trainset.pad_idx, reduction='none')
    min_loss = 10000000000
    optimizer = optim.SGD(parallel_model.parameters(), lr, momentum=0.99, nesterov=True)
    scheduler = cosine_scheduler(optimizer,
                                 warmup_epochs * trainset.num_batches(warmup_batch_size, bptt_len=chunk_size,world_size=num_gpu),
                                 epochs * trainset.num_batches(batch_size, bptt_len=chunk_size, world_size=num_gpu),
                                 warmup_cycles)

    for e in range(epochs + warmup_epochs):

        parallel_model.train()
        total_loss = torch.tensor([0.]).to(gpu)
        total_tokens = torch.tensor([0.]).to(gpu)
        eseed = init_seed + e
        pbar = None

        cbatch_size = batch_size if e >= warmup_epochs else warmup_batch_size
        nbatches = trainset.num_batches(cbatch_size, bptt_len=chunk_size, world_size=num_gpu)
        for (xinput, youtput, first) in trainset.generate_batch(cbatch_size, init_seed=eseed, worker_id=gpu,
                                                                world_size=num_gpu, bptt_len=chunk_size,
                                                                keep_order=False,incremental=False):
            parallel_model.zero_grad()
            X = torch.LongTensor(xinput).to(gpu)  # (seq,batch,emb)
            Y = torch.LongTensor(youtput).to(gpu)  # (seq,batch,emb)
            X,Y = MLMTransformerContextModel.apply_MLM_mask(X,Y,trainset.pad_idx,trainset.vocab_size())
            seq_len, batch_len = Y.shape
            Yhat = parallel_model.forward(X).view(batch_len * len(X), -1)
            Yhat = Yhat.view(batch_len * len(X), -1)
            Y = Y.view(batch_len * len(Y))
            loss = criterion(Yhat, Y)
            loss.sum().backward()

            # masking special tokens for metrics computation
            loss_mask     = (Y != trainset.pad_idx)
            total_loss   += (loss_mask * loss).sum().item()
            total_tokens += loss_mask.sum().item()

            # update
            torch.nn.utils.clip_grad_norm_(parallel_model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()

            if dist.get_rank() == 0:
                if not pbar:
                    pbar = tqdm.tqdm(total=nbatches * dist.get_world_size(), ncols=80)
                pbar.update(dist.get_world_size())

        torch.distributed.all_reduce(total_loss)
        torch.distributed.all_reduce(total_tokens)
        if dist.get_rank() == 0:
            pbar.close()
            print('Epoch %d' % (e + 1))
            nll = (total_loss.item() / total_tokens.item())
            print('  train mean NLL = %.5f  learning rate : %.8f' % (nll, scheduler.get_lr()[0]))
            vloss = model.validate(validset, batch_size, chunk_size, gpu)
            print('  valid mean NLL = %.5f' % (vloss,))
            if vloss < min_loss:
                min_loss = vloss
                torch.save(model.state_dict(), os.path.join(modeldir, 'lm_params.pt'))

