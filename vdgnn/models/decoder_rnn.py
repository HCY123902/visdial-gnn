import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import random
import numpy as np
import ipdb

class Decoder_RNN(nn.Module):

    '''
    Max likelyhood for decoding the utterance
    input_size is the size of the input vocabulary
    Attention module should satisfy that the decoder_hidden size is the same as 
    the Context encoder hidden size
    '''

    def __init__(self, args, encoder):
        super(Decoder_RNN, self).__init__()
        self.output_size = args.vocab_size
        self.hidden_size = args.rnn_hidden_size
        self.word_embed = encoder.word_embed
        self.embed_size = args.embed_size

        # number of layers should be 2
        self.gru = nn.GRU(self.embed_size, self.hidden_size,
                          num_layers=args.num_layers,
                          batch_first=True, 
                          dropout=args.dropout)
        self.out = nn.Linear(self.hidden_size, output_size)

        # attention on context encoder
        # self.attn = Attention(hidden_size)

        self.init_weight()

    def init_weight(self):
        init.xavier_normal_(self.gru.weight_hh_l0)
        init.xavier_normal_(self.gru.weight_ih_l0)
        self.gru.bias_ih_l0.data.fill_(0.0)
        self.gru.bias_hh_l0.data.fill_(0.0)

    def forward(self, inpt, last_hidden):
        # inpt: [batch_size], last_hidden: [2, batch, hidden_size]
        # encoder_outputs: [turn_len, batch, hidden_size]

        # batch first: [batch, ans_len, vocab size]
        embedded = self.word_embed(inpt)
        # key = last_hidden.sum(axis=0)    # [batch, hidden_size]

        # [batch, 1, seq_len]
        # attn_weights = self.attn(key, encoder_outputs)
        # context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        # context = context.transpose(0, 1)    # [1, batch, hidden]

        # rnn_input = torch.cat([embedded, context], 2)   # [1, batch, embed+hidden]

        # output: [batch, ans_len, hidden_size], hidden: [2, batch, hidden_size]
        output, _ = self.gru(embedded, last_hidden)


        # output = output.squeeze(0)    # [batch, hidden_size]


        # context = context.squeeze(0)  # [batch, hidden]
        # output = torch.cat([output, context], 1)    # [batch, 2 * hidden]


        output = self.out(output)     # [batch, ans_len, output_size]
        output = F.log_softmax(output, dim=2)
        return output
        