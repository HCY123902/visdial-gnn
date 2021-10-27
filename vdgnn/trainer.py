import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import gc
import os
import math
import time
import json
from tqdm import tqdm
from vdgnn.models.decoder_rnn import Decoder_RNN
from vdgnn.utils.eval_utils import *
from vdgnn.utils.metrics import NDCG
# from torch.utils.tensorboard import SummaryWriter
import random

class Trainer(object):
    def __init__(self, dataloader, dataloader_val, model_args):
        self.args = model_args
        self.output_dir = model_args.save_path

        self.num_epochs = model_args.num_epochs
        self.lr = model_args.lr
        self.lr_decay_rate = model_args.lr_decay_rate
        self.min_lr = model_args.min_lr
        self.ckpt = model_args.ckpt
        self.use_cuda = model_args.use_cuda
        self.log_step = model_args.log_step

        self.dataloader = dataloader
        self.dataloader_val = dataloader_val

        self.model_dir = os.path.join(self.output_dir, 'checkpoints')
        self.record_path = model_args.record_path
        self.vocab_path = model_args.vocab_path
#         self.writer = SummaryWriter(log_dir='../log/dailydialog/')
        self.teach_force = model_args.teach_force
    
        vocab = open(self.vocab_path, "r")
        ind2word = json.load(vocab)
        ind2word[0] = "<PAD>"
        self.ind2word = ind2word
        self.sos = len(ind2word) - 2
        vocab.close()

    def train(self, encoder, decoder):

        #criterion = nn.CrossEntropyLoss()
        # Adjusted
#        criterion = nn.NLLLoss(ignore_index=0)
        criterion = nn.NLLLoss()

        running_loss = None

        optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),
                                lr=self.lr)

        scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=self.lr_decay_rate)
        
#         title = "dailydialog"args.record_path
        record_path = self.record_path

        if self.ckpt != None:
            components = torch.load(self.ckpt)
            print('Loaded checkpoint from: ' + self.ckpt)
            encoder.load_state_dict(components['encoder'])
            decoder.load_state_dict(components['decoder'])

        if self.use_cuda:
            encoder = encoder.cuda()
            decoder = decoder.cuda()
            criterion = criterion.cuda()

        for epoch in range(1, self.num_epochs+1):
            epoch_time = time.time()

            encoder.train()
            decoder.train()
            iter_time = time.time()

            for iter, batch in enumerate(self.dataloader):
                print("[Epoch: {:3d}][Iter: {:6d}] Starts".format(epoch, iter+1))
                optimizer.zero_grad()

                for key in batch:
                    if not isinstance(batch[key], list):
                        if self.use_cuda:
                            batch[key] = batch[key].cuda()

                batch_size, max_num_rounds = batch['ques'].size()[:2]

                # enc_output = torch.zedros(batch_size, max_num_rounds, self.args.message_size, requires_grad=True)
                # Adjusted
                ans_len = batch['ans'].size(2)
                dec_output = torch.zeros(batch_size, max_num_rounds, ans_len, decoder.output_size, requires_grad=True)

                if self.use_cuda:
                    #enc_output = enc_output.cuda()
                    dec_output = dec_output.cuda()

                # iterate over dialog rounds
                for rnd in range(max_num_rounds):
                    round_info = {}

                    # round_info['img_feat'] = batch['img_feat']

                    round_info['ques'] = batch['ques'][:, rnd, :]
                    round_info['ques_len'] = batch['ques_len'][:, rnd]

                    round_info['hist'] = batch['hist'][:,:rnd+1, :]
                    round_info['hist_len'] = batch['hist_len'][:, :rnd+1]
                    round_info['round'] = rnd

                    pred_adj_mat, enc_out, context_out = encoder(round_info, self.args)
                    context_out = context_out.transpose(1, 2).contiguous()
                    # print(enc_out)

                    # enc_output[:, rnd, :] = enc_out

                    # [2, batch_size, hidden_size]
                    initial_hidden = torch.stack([enc_out, enc_out], 0)

                    # [batch_size, ans_len]
                    ans_tokens = batch['ans'][:, rnd, :]
                    # ans_len = batch['ans'].size(2)
                    
                    use_teacher = random.random() < self.teach_force
                    if use_teacher:
                        out = decoder(ans_tokens, initial_hidden, context_out)
                    else:
#                         start_tokens = torch.zeros(batch_size, dtype=torch.long).fill_(self.sos)
#                         if self.use_cuda:
#                             start_tokens = start_tokens.cuda()
                        out = decoder(ans_tokens[:, 0], initial_hidden, context_out, ans_len=ans_tokens.size(1), is_train=False)

                    dec_output[:, rnd, :, :] = out

                # dec_out = decoder(enc_output.contiguous().view(-1, self.args.message_size), batch)
                
                # # Added temporarily
                # ans_ind = torch.tensor([[1] * 10] * 32).cuda()
                # print("ans_ind size: {}".format(ans_ind.size()))

                # cur_loss = criterion(dec_out, ans_ind.view(-1))
                cur_loss = criterion(dec_output.view(-1, decoder.output_size), batch['ans'].view(-1))
                
                cur_loss.backward()

                optimizer.step()
                gc.collect()

                if running_loss is not None:
                    running_loss = 0.95 * running_loss + 0.05 * cur_loss.data
                else:
                    running_loss = cur_loss.data

                if optimizer.param_groups[0]['lr'] > self.min_lr:
                    scheduler.step()

                # --------------------------------------------------------------------
                # Logging
                # --------------------------------------------------------------------
                if (iter+1) % self.log_step == 0:
                    print("[Epoch: {:3d}][Iter: {:6d}][Loss: {:6f}][lr: {:6f}][Duration: {:6.2f}s]".format(
                        epoch, iter+1, running_loss, optimizer.param_groups[0]['lr'], time.time() - iter_time))
                    iter_time = time.time()
                    ppl = self.validate(encoder, decoder, record_path.format(epoch), epoch)
#                 evaluate_prediction_result(record_path, self.writer, epoch, ppl)
                    self.evaluate_prediction_result(record_path.format(epoch), epoch, ppl)
                    encoder.train()
                    decoder.train()

            print("[Epoch: {:3d}][Loss: {:6f}][lr: {:6f}][Time: {:6.2f}s]".format(
                        epoch, running_loss, optimizer.param_groups[0]['lr'], time.time() - epoch_time))
            # self.writer.add_scalar('{}-Loss/train'.format(title), running_loss, epoch)
            # self.writer.add_scalar('{}-lr/train'.format(title), optimizer.param_groups[0]['lr'], epoch)

            # --------------------------------------------------------------------
            # Save checkpoints
            # --------------------------------------------------------------------
            if epoch % 1 == 0:
                if not os.path.exists(self.model_dir):
                    os.makedirs(self.model_dir)

#                 torch.save({
#                     'encoder':encoder.state_dict(),
#                     'decoder':decoder.state_dict(),
#                     'optimizer': optimizer.state_dict(),
#                     'model_args': self.args
#                 }, os.path.join(self.model_dir, 'model_epoch_{:06d}.pth'.format(epoch)))

                ppl = self.validate(encoder, decoder, record_path.format(epoch), epoch)
#                 evaluate_prediction_result(record_path, self.writer, epoch, ppl)
                self.evaluate_prediction_result(record_path.format(epoch), epoch, ppl)

        torch.save({
            'encoder':encoder.state_dict(),
            'decoder':decoder.state_dict(),
            'optimizer': optimizer.state_dict(),
            'model_args': self.args
        }, os.path.join(self.model_dir, 'model_epoch_final.pth'))


    def validate(self, encoder, decoder, record_path, epoch):
        print('Evaluating...')
        encoder.eval()
        decoder.eval()
        ndcg = NDCG()

        eval_time = time.time()
        
        total_e = None
        batch_number = 0
        
#         criterion = nn.NLLLoss(ignore_index=0)
        criterion = nn.NLLLoss()
    
        
        
        prediction_records = open(record_path, "w")

        

        for i, batch in enumerate(tqdm(self.dataloader)):

            for key in batch:
                if not isinstance(batch[key], list):
                    if self.use_cuda:
                        batch[key] = batch[key].cuda()

            batch_size, max_num_rounds = batch['ques'].size()[:2]

            # enc_output = torch.zeros(batch_size, max_num_rounds, self.args.message_size, requires_grad=True)
            ans_len = batch['ans'].size(2)
            dec_output = torch.zeros(batch_size, max_num_rounds, ans_len, decoder.output_size, requires_grad=True)

            if self.use_cuda:
                # enc_output = enc_output.cuda()
                dec_output = dec_output.cuda()

            # iterate over dialog rounds
            with torch.no_grad():
                for rnd in range(max_num_rounds):
                    round_info = {}

                    # round_info['img_feat'] = batch['img_feat']

                    round_info['ques'] = batch['ques'][:, rnd, :]
                    round_info['ques_len'] = batch['ques_len'][:, rnd]

                    round_info['hist'] = batch['hist'][:,:rnd+1, :]
                    round_info['hist_len'] = batch['hist_len'][:, :rnd+1]
                    round_info['round'] = rnd

                    pred_adj_mat, enc_out, context_out = encoder(round_info, self.args)
                    context_out = context_out.transpose(1, 2).contiguous()

                    #enc_output[:, rnd, :] = enc_out
                    
                    # [2, batch_size, hidden_size]
                    initial_hidden = torch.stack([enc_out, enc_out], 0)

                    # [batch_size, ans_len]
                    ans_tokens = batch['ans'][:, rnd, :]
                    # ans_len = batch['ans'].size(2)
                    
#                     start_tokens = torch.zeros(batch_size, dtype=torch.long).fill_(self.sos)
#                     if self.use_cuda:
#                             start_tokens = start_tokens.cuda()

                    out = decoder(ans_tokens[:, 0], initial_hidden, context_out, ans_len=ans_tokens.size(1), is_train=False)
                    
                    dec_output[:, rnd, :, :] = out
                    
                    self.translate(round_info['hist'], round_info['ques'], out, ans_tokens, prediction_records)
                    

                    
                cur_loss = criterion(dec_output.view(-1, decoder.output_size), batch['ans'].view(-1))
                batch_number = batch_number + 1
                #total_e = total_e + int(cur_loss.data)
                
                if total_e is not None:
                    total_e = total_e + cur_loss.data
                else:
                    total_e = cur_loss.data
        
        
        prediction_records.close()
        
        l = round(total_e.item() / batch_number, 4)
        ppl = math.exp(l)

        gc.collect()

        return ppl

    def translate(self, source, question, prediction, reference, record):
        
#         ind2word = json.load()
#        ind2word[0] = "<PAD>"
        ind2word = self.ind2word
        
        end_token = len(ind2word)
        
        batch_size, ans_len = prediction.size()[:2]
        _, rounds, hist_len = source.size()
        
        # [batch_size, ans_len]
        tokens = prediction.max(2)[1]
        
        for i in range(batch_size):
            src_list = []
            for r in range(rounds):
                src = list(map(int, source[i, r, :].tolist()))
                end_index = src.index(0) if 0 in src else len(src)
                end_index_ = src.index(end_token) if end_token in src else len(src)
                src = src[:min(end_index, end_index_)]
                src = [ind2word.get(str(token), '<UNK>') for token in src]
                src_list.append(' '.join(src))
            src = ' __eou__ '.join(src_list)
            
            ques = list(map(int, question[i, :].tolist()))
            end_index = ques.index(0) if 0 in ques else len(ques)
            end_index_ = ques.index(end_token) if end_token in ques else len(ques)
            #end_index = min(end_index, end_index_)
            ques = ques[:min(end_index, end_index_)]
            ques = [ind2word.get(str(token), '<UNK>') for token in ques]
            ques = ' '.join(ques)
            ques.replace('<START>', '').strip()
            ques.replace('<END>', '').strip()
            
            pred = list(map(int, tokens[i, :].tolist()))
            end_index = pred.index(0) if 0 in pred else len(pred)
            end_index_ = pred.index(end_token) if end_token in pred else len(pred)
            #end_index = min(end_index, end_index_)
            pred = pred[:min(end_index, end_index_)]
            pred = [ind2word.get(str(token), '<UNK>') for token in pred]
            pred = ' '.join(pred)
            pred.replace('<START>', '').strip()
            pred.replace('<END>', '').strip()
        
            ref = list(map(int, reference[i, :].tolist()))
            # print(ref)
            end_index = ref.index(0) if 0 in ref else len(ref)
            end_index_ = ref.index(end_token) if end_token in ref else len(ref)

            ref = ref[:min(end_index, end_index_)]
            
            ref = [ind2word.get(str(token), '<UNK>') for token in ref]
            ref = ' '.join(ref)
            ref.replace('<START>', '').strip()
            ref.replace('<END>', '').strip()
            
            record.write('- src: {}\n'.format(src))
            record.write('- que: {}\n'.format(ques))
            record.write('- ref: {}\n'.format(ref))
            record.write('- tgt: {}\n\n'.format(pred))
   
    
    def evaluate_prediction_result(self, pred_path, epoch, ppl):
        # obtain the performance
        # print('[!] measure the performance and write into tensorboard')
        with open(pred_path) as f:
            ref, tgt = [], []
            for idx, line in enumerate(f.readlines()):
                line = line.lower()    # lower the case
                if idx % 5 == 2:
                    line = line.replace("user1", "").replace("user0", "").replace("- ref: ", "").replace('<sos>', '').replace('<eos>', '').strip()
                    # print(line)
                    ref.append(line.split())
                elif idx % 5 == 3:
                    line = line.replace("user1", "").replace("user0", "").replace("- tgt: ", "").replace('<sos>', '').replace('<eos>', '').strip()
                    # print(line)
                    tgt.append(line.split())

        assert len(ref) == len(tgt)

        # ROUGE
        rouge_sum, bleu1_sum, bleu2_sum, bleu3_sum, bleu4_sum, counter = 0, 0, 0, 0, 0, 0
        for rr, cc in tqdm(list(zip(ref, tgt))):
            rouge_sum += cal_ROUGE(rr, cc)
            # rouge_sum += 0.01
            counter += 1

        # BlEU
        refs, tgts = [' '.join(i) for i in ref], [' '.join(i) for i in tgt]
        bleu1_sum, bleu2_sum, bleu3_sum, bleu4_sum = cal_aggregate_BLEU_nltk(refs, tgts)

#         # Distinct-1, Distinct-2
#         candidates, references = [], []
#         for line1, line2 in zip(tgt, ref):
#             candidates.extend(line1)
#             references.extend(line2)
#         distinct_1, distinct_2 = cal_Distinct(candidates)
#         rdistinct_1, rdistinct_2 = cal_Distinct(references)

        # Embedding-based metric: Embedding Average (EA), Vector Extrema (VX), Greedy Matching (GM)
        # load the dict
        # with open('./data/glove_embedding.pkl', 'rb') as f:
        #     dic = pickle.load(f)
#         dic = gensim.models.KeyedVectors.load_word2vec_format('./data/GoogleNews-vectors-negative300.bin', binary=True)
#         print('[!] load the GoogleNews 300 word2vector by gensim over')
#         ea_sum, vx_sum, gm_sum, counterp = 0, 0, 0, 0
#         for rr, cc in tqdm(list(zip(ref, tgt))):
#             ea_sum += cal_embedding_average(rr, cc, dic)
#             vx_sum += cal_vector_extrema(rr, cc, dic)
#             gm_sum += cal_greedy_matching_matrix(rr, cc, dic)
#             counterp += 1

        # write into the tensorboard
#         writer.add_scalar('{}-Performance/PPL'.format(title), ppl, epoch)
#         writer.add_scalar('{}-Performance/BLEU-1'.format(title), bleu1_sum, epoch)
#         writer.add_scalar('{}-Performance/BLEU-2'.format(title), bleu2_sum, epoch)
#         writer.add_scalar('{}-Performance/BLEU-3'.format(title), bleu3_sum, epoch)
#         writer.add_scalar('{}-Performance/BLEU-4'.format(title), bleu4_sum, epoch)
#         writer.add_scalar('{}-Performance/ROUGE'.format(title), rouge_sum / counter, epoch)
#         writer.add_scalar(f'{writer_str}-Performance/Distinct-1', distinct_1, epoch)
#         writer.add_scalar(f'{writer_str}-Performance/Distinct-2', distinct_2, epoch)
#         writer.add_scalar(f'{writer_str}-Performance/Ref-Distinct-1', rdistinct_1, epoch)
#         writer.add_scalar(f'{writer_str}-Performance/Ref-Distinct-2', rdistinct_2, epoch)
#         writer.add_scalar(f'{writer_str}-Performance/Embedding-Average', ea_sum / counterp, epoch)
#         writer.add_scalar(f'{writer_str}-Performance/Vector-Extrema', vx_sum / counterp, epoch)
#         writer.add_scalar(f'{writer_str}-Performance/Greedy-Matching', gm_sum / counterp, epoch)

        print("[Epoch: {:3d}][BLEU-1: {:6f}][BLEU-2: {:6f}][BLEU-3: {:6f}][BLEU-4: {:6f}][ROUGE: {:6f}][PPL: {:6f}]".format(
                        epoch, bleu1_sum, bleu2_sum, bleu3_sum, bleu4_sum, rouge_sum / counter, ppl))

        # write now
        # writer.flush()
        
        

