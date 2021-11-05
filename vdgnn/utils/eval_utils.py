import torch
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction
from nltk.collocations import BigramCollocationFinder
from nltk.probability import FreqDist
import argparse
import codecs
import math
from rouge import Rouge
import numpy as np
import os, re

def cal_aggregate_BLEU_nltk(refs, tgts):
    #print(refs)
    #print(tgts)
    smoothie = SmoothingFunction().method7
    weights = [(1, 0, 0, 0), (0.5, 0.5, 0, 0), (0.33, 0.33, 0.33, 0), (0.25, 0.25, 0.25, 0.25)]
    
    refs = [[ref.split(' ')] for ref in refs]
    tgts = [tgt.split(' ') for tgt in tgts]
    result = []
    for i in range(0,4):
        result = result + [corpus_bleu(refs, tgts, weights=weights[i], smoothing_function=smoothie)]
    return result[0], result[1], result[2], result[3]

def cal_ROUGE(refer, candidate):
    if len(candidate) == 0:
        candidate = ['<unk>']
    elif len(candidate) == 1:
        candidate.append('<unk>')
    if len(refer) == 0:
        refer = ['<unk>']
    elif len(refer) == 1:
        refer.append('<unk>')
    rouge = Rouge()
    scores = rouge.get_scores(' '.join(candidate), ' '.join(refer))
    return scores[0]['rouge-2']['f']

def cal_Distinct(corpus):
    """
    Calculates unigram and bigram diversity
    Args:
        corpus: tokenized list of sentences sampled
    Returns:
        uni_diversity: distinct-1 score
        bi_diversity: distinct-2 score
    """
    bigram_finder = BigramCollocationFinder.from_words(corpus)
    if bigram_finder.N == 0:
        bi_diversity = 0.9
    else:
        bi_diversity = float(len(bigram_finder.ngram_fd)) / float(bigram_finder.N)

    dist = FreqDist(corpus)
    if len(corpus) == 0:
        uni_diversity = 0.5
    else:
        uni_diversity = float(len(dist)) / float(len(corpus))

    return uni_diversity, bi_diversity

def cal_intra_Distinct(lines):
    """
    Calculates unigram and bigram diversity
    Args:
        corpus: tokenized list of sentences sampled
    Returns:
        uni_diversity: distinct-1 score
        bi_diversity: distinct-2 score
    """
    unary = []
    binary = []
    
    for line in lines:
      uni_distinct, bi_distinct = cal_Distinct(line)
      unary.append(uni_distinct)
      
      binary.append(bi_distinct)

    return float(np.mean(unary)), float(np.mean(binary))

def cal_vector_extrema(x, y, dic):
    # x and y are the list of the words
    # dic is the gensim model which holds 300 the google news word2ved model
    def vecterize(p):
        vectors = []
        for w in p:
            if w in dic:
                # Adjusted
                if w.lower() in dic:
                    vectors.append(dic[w.lower()])
                else:
                    vectors.append(dic[w])
        if not vectors:
            vectors.append(np.random.randn(300))
        return np.stack(vectors)
    x = vecterize(x)
    y = vecterize(y)
    vec_x = np.max(x, axis=0)
    vec_y = np.max(y, axis=0)
    assert len(vec_x) == len(vec_y), "len(vec_x) != len(vec_y)"
    zero_list = np.zeros(len(vec_x))
    if vec_x.all() == zero_list.all() or vec_y.all() == zero_list.all():
        return float(1) if vec_x.all() == vec_y.all() else float(0)
    res = np.array([[vec_x[i] * vec_y[i], vec_x[i] * vec_x[i], vec_y[i] * vec_y[i]] for i in range(len(vec_x))])
    cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))
    return cos


def cal_embedding_average(x, y, dic):
    # x and y are the list of the words
    def vecterize(p):
        vectors = []
        for w in p:
            if w in dic:
                # Adjusted
                if w.lower() in dic:
                    vectors.append(dic[w.lower()])
                else:
                    vectors.append(dic[w])
        if not vectors:
            vectors.append(np.random.randn(300))
        return np.stack(vectors)
    x = vecterize(x)
    y = vecterize(y)
    
    vec_x = np.array([0 for _ in range(len(x[0]))])
    for x_v in x:
        x_v = np.array(x_v)
        vec_x = np.add(x_v, vec_x)
    vec_x = vec_x / math.sqrt(sum(np.square(vec_x)))
    
    vec_y = np.array([0 for _ in range(len(y[0]))])
    #print(len(vec_y))
    for y_v in y:
        y_v = np.array(y_v)
        vec_y = np.add(y_v, vec_y)
    vec_y = vec_y / math.sqrt(sum(np.square(vec_y)))
    
    assert len(vec_x) == len(vec_y), "len(vec_x) != len(vec_y)"
    
    zero_list = np.array([0 for _ in range(len(vec_x))])
    if vec_x.all() == zero_list.all() or vec_y.all() == zero_list.all():
        return float(1) if vec_x.all() == vec_y.all() else float(0)
    
    vec_x = np.mat(vec_x)
    vec_y = np.mat(vec_y)
    num = float(vec_x * vec_y.T)
    denom = np.linalg.norm(vec_x) * np.linalg.norm(vec_y)
    cos = num / denom
    
    # res = np.array([[vec_x[i] * vec_y[i], vec_x[i] * vec_x[i], vec_y[i] * vec_y[i]] for i in range(len(vec_x))])
    # cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))
    
    return cos


def cal_greedy_matching(x, y, dic):
    # x and y are the list of words
    def vecterize(p):
        vectors = []
        for w in p:
            if w in dic:
                # Adjusted
                if w.lower() in dic:
                    vectors.append(dic[w.lower()])
                else:
                    vectors.append(dic[w])
        if not vectors:
            vectors.append(np.random.randn(300))
        return np.stack(vectors)
    x = vecterize(x)
    y = vecterize(y)
    
    len_x = len(x)
    len_y = len(y)
    
    cosine = []
    sum_x = 0 

    for x_v in x:
        for y_v in y:
            assert len(x_v) == len(y_v), "len(x_v) != len(y_v)"
            zero_list = np.zeros(len(x_v))

            if x_v.all() == zero_list.all() or y_v.all() == zero_list.all():
                if x_v.all() == y_v.all():
                    cos = float(1)
                else:
                    cos = float(0)
            else:
                # method 1
                res = np.array([[x_v[i] * y_v[i], x_v[i] * x_v[i], y_v[i] * y_v[i]] for i in range(len(x_v))])
                cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))

            cosine.append(cos)
        if cosine:
            sum_x += max(cosine)
            cosine = []

    sum_x = sum_x / len_x
    cosine = []

    sum_y = 0

    for y_v in y:

        for x_v in x:
            assert len(x_v) == len(y_v), "len(x_v) != len(y_v)"
            zero_list = np.zeros(len(y_v))

            if x_v.all() == zero_list.all() or y_v.all() == zero_list.all():
                if (x_v == y_v).all():
                    cos = float(1)
                else:
                    cos = float(0)
            else:
                # method 1
                res = np.array([[x_v[i] * y_v[i], x_v[i] * x_v[i], y_v[i] * y_v[i]] for i in range(len(x_v))])
                cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))

            cosine.append(cos)

        if cosine:
            sum_y += max(cosine)
            cosine = []

    sum_y = sum_y / len_y
    score = (sum_x + sum_y) / 2
    return score


def cal_greedy_matching_matrix(x, y, dic):
    # x and y are the list of words
    def vecterize(p):
        vectors = []
        for w in p:
            if w in dic:
                # Adjusted
                if w.lower() in dic:
                    vectors.append(dic[w.lower()])
                else:
                    vectors.append(dic[w])
        if not vectors:
            vectors.append(np.random.randn(300))
        return np.stack(vectors)
    x = vecterize(x)     # [x, 300]
    y = vecterize(y)     # [y, 300]
    
    len_x = len(x)
    len_y = len(y)
    
    matrix = np.dot(x, y.T)    # [x, y]
    matrix = matrix / np.linalg.norm(x, axis=1, keepdims=True)    # [x, 1]
    matrix = matrix / np.linalg.norm(y, axis=1).reshape(1, -1)    # [1, y]
    
    x_matrix_max = np.mean(np.max(matrix, axis=1))    # [x]
    y_matrix_max = np.mean(np.max(matrix, axis=0))    # [y]
    
    return (x_matrix_max + y_matrix_max) / 2

def get_gt_ranks(ranks, ans_ind):
    ans_ind = ans_ind.view(-1)
    gt_ranks = torch.LongTensor(ans_ind.size(0))
    for i in range(ans_ind.size(0)):
        gt_ranks[i] = int(ranks[i, ans_ind[i]])
    return gt_ranks


def process_ranks(ranks):
    num_ques = ranks.size(0)
    num_opts = 100

    # none of the values should be 0, there is gt in options
    if torch.sum(ranks.le(0)) > 0:
        num_zero = torch.sum(ranks.le(0))
        print("Warning: some of ranks are zero: {}".format(num_zero))
        ranks = ranks[ranks.gt(0)]

    # rank should not exceed the number of options
    if torch.sum(ranks.ge(num_opts + 1)) > 0:
        num_ge = torch.sum(ranks.ge(num_opts + 1))
        print("Warning: some of ranks > 100: {}".format(num_ge))
        ranks = ranks[ranks.le(num_opts + 1)]

    ranks = ranks.float()
    num_r1 = float(torch.sum(torch.le(ranks, 1)))
    num_r5 = float(torch.sum(torch.le(ranks, 5)))
    num_r10 = float(torch.sum(torch.le(ranks, 10)))
    return {
        'num_ques': num_ques,
        'r_1': num_r1 / num_ques,
        'r_5': num_r5 / num_ques,
        'r_10': num_r10 / num_ques,
        'mr': torch.mean(ranks),
        'mrr': torch.mean(ranks.reciprocal())
    }

def scores_to_ranks(scores):
    # sort in descending order - largest score gets highest rank
    sorted_ranks, ranked_idx = scores.sort(1, descending=True)

    # convert from ranked_idx to ranks
    ranks = ranked_idx.clone().fill_(0)
    for i in range(ranked_idx.size(0)):
        for j in range(100):
            ranks[i][ranked_idx[i][j]] = j
    ranks += 1
    return ranks


