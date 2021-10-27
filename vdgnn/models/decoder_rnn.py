import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import random
import numpy as np
import ipdb
from .layers import *

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
        self.gru = nn.GRU(self.embed_size + self.hidden_size, self.hidden_size,
                          num_layers=args.num_layers,
                          batch_first=True, 
                          dropout=args.dropout)
        self.out = nn.Linear(self.hidden_size, self.output_size)

        # attention on context encoder
        self.attn = Attention(self.hidden_size)

        self.init_weight()

    def init_weight(self):
        init.xavier_normal_(self.gru.weight_hh_l0)
        init.xavier_normal_(self.gru.weight_ih_l0)
        self.gru.bias_ih_l0.data.fill_(0.0)
        self.gru.bias_hh_l0.data.fill_(0.0)

    def forward(self, inpt, last_hidden, encoder_outputs, ans_len=20, is_train=True):
        # inpt: [batch_size, ans_len], last_hidden: [2, batch, hidden_size]
        # encoder_outputs: [batch, turn_len, hidden_size]

        if is_train:
            
#             key = last_hidden.sum(axis=0)    # [batch, hidden_size]

#             # [batch, 1, seq_len]
#             attn_weights = self.attn(key, encoder_outputs)
#             context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
#             context = context.transpose(0, 1)    # [1, batch, hidden]
            ans_len = inpt.size(1)
            embedded = self.word_embed(inpt)    # [batch_size, ans_len, embed_size]
            key = last_hidden.sum(dim=0)    # [batch, hidden_size]
            
            attn_weights = self.attn(key, encoder_outputs) # [batch, 1, timestep]
            context = attn_weights.bmm(encoder_outputs) # [batch, 1, hidden]
            context = context.repeat(1, ans_len, 1) # [batch, ans_len, hidden]
            # context = context.transpose(0, 1)    # [1, batch, hidden]

            rnn_input = torch.cat([embedded, context], 2)   # [batch, ans_len, embed+hidden]
            # batch first: [batch, ans_len, vocab size]
            
            
            # output: [batch, ans_len, hidden_size], hidden: [2, batch, hidden_size]
            output, _ = self.gru(rnn_input, last_hidden)
            
            output = self.out(output)     # [batch, ans_len, output_size]
        else:
            output = torch.zeros(inpt.size(0), ans_len, self.output_size, requires_grad=True)
            output = output.cuda()
            #l = []
            current = inpt
            hidden = last_hidden
            for i in range(0, ans_len):
                embedded = self.word_embed(current).unsqueeze(1) # [batch_size, 1, embed_size]
                
                key = hidden.sum(dim=0)    # [batch, hidden_size]
            
                attn_weights = self.attn(key, encoder_outputs) # [batch, 1, timestep]
                context = attn_weights.bmm(encoder_outputs) # [batch, 1, hidden]
                # context = context.repeat(1, ans_len, 1) # [batch, ans_len, hidden]
                # context = context.transpose(0, 1)    # [1, batch, hidden]

                rnn_input = torch.cat([embedded, context], 2)   # [batch, 1, embed+hidden]
                
                current, hidden = self.gru(rnn_input, hidden)
                # [batch, vocab_size]
                current = self.out(current).squeeze(1)
                
                output[:, i, :] = current
                #l.append(current)
                current = F.log_softmax(current, dim=1)
                current = current.max(1)[1]
                
                
                
        output = F.log_softmax(output, dim=2)
        return output
    
    

        