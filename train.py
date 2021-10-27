import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import torch
import argparse
import time

from vdgnn.dataset.dataloader import VisDialDataset
from torch.utils.data import DataLoader
from vdgnn.models.decoder_rnn import Decoder_RNN
from vdgnn.models.decoder_rnn_hdno import DecoderRNN
from vdgnn.models.encoder import GCNNEncoder
from vdgnn.models.decoder import DiscriminativeDecoder
from vdgnn.options.train_options import TrainOptions
from vdgnn.trainer import Trainer

if __name__ == '__main__':
    # For reproducibility
    RANDOM_SEED = 0
    torch.manual_seed(RANDOM_SEED)
    
    opts = TrainOptions().parse()
    if opts.gpuid >= 0 and torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
        torch.cuda.set_device(opts.gpuid)
        opts.use_cuda = True
    
    TrainOptions().print_options(opts)
    
    dataset = VisDialDataset(opts, 'train', isTrain=True)
    dataset_val = VisDialDataset(opts, 'val', isTrain=True)
    dataloader = DataLoader(dataset,
                            batch_size=opts.batch_size,
                            shuffle=True,
                            drop_last=True,
                            collate_fn=dataset.collate_fn)

    dataloader_val = DataLoader(dataset_val,
                                batch_size=opts.batch_size,
                                shuffle=True,
                                drop_last=True,
                                collate_fn=dataset_val.collate_fn)

    # transfer all options to model
    model_args  = opts

    for key in {'num_data_points', 'vocab_size', 'max_ques_count',
            'max_ques_len', 'max_ans_len'}:
        setattr(model_args, key, getattr(dataset, key))

    encoder = GCNNEncoder(model_args)

    # decoder = DiscriminativeDecoder(model_args, encoder)
    # decoder = Decoder_RNN(model_args, encoder)
    decoder = DecoderRNN(input_dropout_p=model_args.decoder_dropout,
                                  rnn_cell='lstm',
                                  input_size=model_args.embed_size,
                                  hidden_size=model_args.rnn_hidden_size,
                                  num_layers=2,
                                  output_dropout_p=model_args.decoder_dropout,
                                  bidirectional=False,
                                  vocab_size=model_args.vocab_size,
                                  use_attn=True,
                                  ctx_cell_size=model_args.rnn_hidden_size,
                                  attn_mode='cat',
                                  sys_id=model_args.vocab_size - 2,
                                  eos_id=model_args.vocab_size - 1,
                                  use_gpu=True,
                                  max_dec_len=20,
                                  embedding=None)

    trainer = Trainer(dataloader, dataloader_val, model_args)

    trainer.train(encoder, decoder)

