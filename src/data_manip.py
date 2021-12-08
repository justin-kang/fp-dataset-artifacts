import json
import pandas as pd
import random
import math
import numpy as np
from nltk.tokenize import word_tokenize

random.seed(10)

def save_df(output_file,df):
    with open(output_file,'w') as file:
        for i in range(len(df)):
            json.dump(df.iloc[i].to_dict(),file,separators=(',',':'))
            file.write('\n')

def adv_to_json(train_file,adv_file,output_file):
    train_data = pd.read_json(train_file,lines=True)
    adv_data = pd.read_json(adv_file,lines=True)
    adv_data = pd.DataFrame(adv_data['data'][0])
    new_df = pd.DataFrame(columns=train_data.columns)
    for i,title in enumerate(adv_data['title']):
        paragraphs = adv_data['paragraphs'][i]
        for paragraph in paragraphs:
            for question in paragraph['qas']:
                answers = {}
                for answer in question['answers']:
                    for key in answer.keys():
                        if key in answers:
                            answers[key].append(answer[key])
                        else:
                            answers[key] = [answer[key]]
                new_row = {'id':question['id'],'title':title,
                        'context':paragraph['context'],
                        'question':question['question'],'answers':answers}
                new_df = new_df.append(new_row,ignore_index=True)
    save_df(output_file,new_df)

def incorrect_preds(input_file,output_file):
    eval_data = pd.read_json(input_file,lines=True)
    new_df = pd.DataFrame(columns=eval_data.columns)
    for i in range(len(eval_data)):
        refs = eval_data['answers'][i]['text']
        pred = eval_data['predicted_answer'][i]
        contains = False
        for ref in refs:
            if ref in pred or pred in ref:
                contains = True
                break
        if contains:
            continue
        '''
        question = eval_data['question'][i]
        context = eval_data['context'][i]
        adv_begin = context[:-1].rindex('.')+2
        adversary = context[adv_begin:]
        '''
        new_df = new_df.append(eval_data.iloc[i],ignore_index=True)
    save_df(output_file,new_df)

def augment_training(adv_file,output_file):
    adv_data = pd.read_json(adv_file,lines=True)
    new_df = pd.DataFrame(columns=adv_data.columns)
    num_rows = len(adv_data)
    # sample 10% of the adversarial data to inoculate our data
    samples = random.sample(list(range(num_rows)),round(num_rows/10))
    for sample in samples:
        data = adv_data.iloc[sample]
        # separate out the adversarial sentence from the original context
        context = data['context']
        if 'turk' not in data['id']:
            new_df = new_df.append(data,ignore_index=True)
            continue
        adv_sent_idx = context.strip().rindex('. ')+1
        adv_sent = context[adv_sent_idx:]
        og_context = context[:adv_sent_idx]
        sent_ends = [i for i in range(len(og_context)) if og_context[i] == '.']
        # move the adversarial sentence to a random sentence position
        new_idx = random.choice(sent_ends)+1
        answer_idx = data['answers']['answer_start']
        old_answers = []
        for i in range(len(answer_idx)):
            old_answers.append(answer_idx[i])
            if answer_idx[i] > new_idx:
                answer_idx[i] += len(adv_sent)
        new_context = og_context[:new_idx] + adv_sent + og_context[new_idx:]
        data['context'] = new_context
        new_df = new_df.append(data,ignore_index=True)
    save_df(output_file,new_df)

if True:
    adv_to_json('../data/train_data.json',
            '../data/sample1k-HCVerifySample.json','../data/add_one_sent.json')
    adv_to_json('../data/train_data.json',
            '../data/sample1k-HCVerifyAll.json','../data/add_sent.json')

if True:
    incorrect_preds('../output/base_when_eval_output/eval_predictions.jsonl',
        '../output/base_when_eval_output/incorrect_preds.json')
    incorrect_preds('../output/addsent_when_eval_output/eval_predictions.jsonl',
        '../output/addsent_when_eval_output/incorrect_preds.json')

if True:
    augment_training('../data/add_sent.json','../data/new_train_data.json')