import json
import os
import nltk



import numpy as np

import argparse
import logging
import sys



if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Convert a conversation graph to a set of connected components (i.e. threads).')
    # parser.add_argument('--dir', help='The directory that contains the dialogue <filename>:...')
    # parser.add_argument('--max_ques_len', default=20, type=int, help='Max length of questions')
    # parser.add_argument('--max_ans_len', default=20, type=int, help='Max length of answers')
    # parser.add_argument('--max_num_round', default=10, type=int)
    # parser.add_argument('--result', help='The file containing the cluster content as <filename>:...')

    # args = parser.parse_args()

    # result = {}



    # train = open(os.path.join(args.dir, "dial.train.txt"), "r")
    # valid = open(os.path.join(args.dir, "dial.valid.txt"), "r")
    # test = open(os.path.join(args.dir, "dial.test.txt"), "r")


    # t = train.readlines()
    # v = valid.readlines()

    # assert len(t[-1]) > 1
    # assert len(v[-1]) > 1

    # num_train_samples = len(t)
    # num_valid_samples = len(v)

    # result['ques_train'] = np.zeros([num_train_samples, args.max_num_round, args.max_ques_length], dtype=np.int)
    # result['ques_length_train'] = np.zeros([num_train_samples, args.max_num_round], dtype=np.int)
    # result['ans_train'] = np.zeros([num_train_samples, args.max_num_round, args.max_ques_length], dtype=np.int)
    # result['ans_length_train'] = np.zeros([num_train_samples, args.max_num_round], dtype=np.int)

    # result['num_rounds_train'] = np.zeros(num_train_samples, dtype=np.int)

    # result['ques_val'] = []
    # result['ques_length_val'] = []
    # result['ans_val'] = []
    # result['ans_length_val'] = []

    # result['num_rounds_val'] = []

    # word

    # for line in train:
    #     # dialogue = {"question": , "answer": }
    #     dialogue = [l.strip() for l in line.split("__eou__")[:-1]]
    #     dialogue = [nltk.tokenize.word_tokenize(l)[:args.max_ques_length] for l in dialogue]
    #     dialogue = dialogue[:min(len(dialogue, args.max_num_round))]
    

    parser = argparse.ArgumentParser(description='Convert a conversation graph to a set of connected components (i.e. threads).')
    parser.add_argument('--dir', help='The directory that contains the dialogue <filename>:...')
    parser.add_argument('--result', help='The file containing the cluster content as <filename>:...')

    args = parser.parse_args()

    result_train = {"version": "1.0", "split": "train2018", "data": {
        "questions": [],
        "answers": [],
        "dialogs": []
    }}
    result_val = {"version": "1.0", "split": "val2018", "data": {
        "questions": [],
        "answers": [],
        "dialogs": []
    }}
    result_test = {"version": "1.0", "split": "test2018", "data": {
        "questions": [],
        "answers": [],
        "dialogs": []
    }}

    train = open(os.path.join(args.dir, "dialogues_train.txt"), "r")
    valid = open(os.path.join(args.dir, "dialogues_validation.txt"), "r")
    test = open(os.path.join(args.dir, "dialogues_test.txt"), "r")

    train_json = open(os.path.join(args.result, "visdial_1.0_train.json"), "w")
    valid_json = open(os.path.join(args.result, "visdial_1.0_val.json"), "w")
    test_json = open(os.path.join(args.result, "visdial_1.0_test.json"), "w")


    # t = train.readlines()
    # v = valid.readlines()
    # te = test.readlines()
    # sentense_mapping = []

    # for line in train:
    #     t = [l.strip() for l in line.split("__eou__")[:-1]]
    #     sentense_mapping.extend(t)
    # for line in valid:
    #     t = [l.strip() for l in line.split("__eou__")[:-1]]
    #     sentense_mapping.extend(t)


    shared_answers = []
    shared_questions = []
    shared_count = 0

    for line in train:
        rounds = [l.strip() for l in line.split("__eou__")[:-1]]
        dialog = []
        for i in range(len(rounds) - 1):
            shared_questions.append(rounds[i])
            shared_answers.append(rounds[i + 1])
            
            dialog.append({
                    "question": shared_count,
                    "answer": shared_count
                })
            shared_count = shared_count + 1

        result_train["data"]["dialogs"].append({
            "dialog": dialog
        })
        assert rounds[-1] == shared_answers[-1]


    for line in valid:
        rounds = [l.strip() for l in line.split("__eou__")[:-1]]
        dialog = []
        for i in range(len(rounds) - 1):
            shared_questions.append(rounds[i])
            shared_answers.append(rounds[i + 1])
            
            dialog.append({
                    "question": shared_count,
                    "answer": shared_count
                })
            shared_count = shared_count + 1
        result_val["data"]["dialogs"].append({
            "dialog": dialog
        })
        assert rounds[-1] == shared_answers[-1]

    for line in test:
        rounds = [l.strip() for l in line.split("__eou__")[:-1]]
        dialog = []
        for i in range(len(rounds) - 1):
            shared_questions.append(rounds[i])
            shared_answers.append(rounds[i + 1])
            
            dialog.append({
                    "question": shared_count,
                    "answer": shared_count
                })
            shared_count = shared_count + 1
        result_test["data"]["dialogs"].append({
            "dialog": dialog
        })
        assert rounds[-1] == shared_answers[-1]

    result_train["data"]["questions"] = shared_questions
    result_train["data"]["anwsers"] = shared_answers
    result_val["data"]["questions"] = shared_questions
    result_val["data"]["anwsers"] = shared_answers
    result_test["data"]["questions"] = shared_questions
    result_test["data"]["anwsers"] = shared_answers

    print("Number of questions are {}".format(len(shared_questions)))
    print("Number of answers are {}".format(len(shared_answers)))

    json.dump(result_train, train_json)
    json.dump(result_val, valid_json)
    json.dump(result_test, test_json)

