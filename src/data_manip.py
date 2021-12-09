import json
import pandas as pd
import random
import math
import numpy as np
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

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

def cos_sim(question,adversary):
    q_toks = word_tokenize(question)
    adv_toks = word_tokenize(adversary)
    q_set = set([i for i in q_toks if not i in stopwords.words('english')])
    adv_set = set([i for i in adv_toks if not i in stopwords.words('english')])
    union = q_set.union(adv_set)
    q_list = []
    adv_list = []
    for w in union:
        if w in q_set:
            q_list.append(1)
        else:
            q_list.append(0)
        if w in adv_set:
            adv_list.append(1)
        else:
            adv_list.append(0)
    q_list = np.asarray(q_list)
    adv_list = np.asarray(adv_list)
    c = sum([q_list[i]*adv_list[i] for i in range(len(union))])
    return c/math.sqrt(sum(q_list)*sum(adv_list))

def incorrect_preds(input_file,output_file,hist_file=None,q_file=None):
    eval_data = pd.read_json(input_file,lines=True)
    new_df = pd.DataFrame(columns=eval_data.columns)
    cos_sims = []
    q_len = []
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
        if 'turk' not in eval_data['id'][i]:
            new_df = new_df.append(eval_data.iloc[i],ignore_index=True)
            continue
        question = eval_data['question'][i]
        context = eval_data['context'][i]
        adv_begin = context.rindex('. ')+2
        adversary = context[adv_begin:]
        q_len.append(len(question.strip().split()))
        cos_sims.append(cos_sim(question,adversary))
        new_df = new_df.append(eval_data.iloc[i],ignore_index=True)
    if hist_file is not None:
        plt.figure()
        plt.hist(cos_sims,np.arange(0,1,0.05),range=(0,1))
        plt.xlim(left=0)
        plt.ylim(bottom=0)
        plt.xlabel('Cosine similarity')
        plt.ylabel('Number of examples')
        plt.tight_layout()
        plt.savefig(hist_file)
    if q_file is not None:
        plt.figure()
        plt.hist(q_len)
        plt.xlabel('Length of question [words]')
        plt.ylabel('Number of examples')
        plt.tight_layout()
        plt.savefig(q_file)
    save_df(output_file,new_df)

def augment_training(adv_file,output_file,num_ex):
    adv_data = pd.read_json(adv_file,lines=True)
    new_df = pd.DataFrame(columns=adv_data.columns)
    num_rows = len(adv_data)
    # sample 10% of the adversarial data to inoculate our data
    samples = random.sample(list(range(num_rows)),num_ex)
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

def plot_performance():
    x_labels = [0,10,50,100,400,500,750,1000]
    x = list(range(len(x_labels)))
    base_perf = np.asarray([86.1424059782915,85.82956072688161,
            84.13013823552198,84.39225375650783,84.1681715817831,
            83.82294538021519,83.45308918464718,82.7942590310177])
    adv_perf = np.asarray([61.25396482560945,64.43348907576933,
            66.8575992578582,71.05605105167675,80.76673732203967,
            85.12420704750517,88.10408832121742,91.67969030017157])
    plt.figure()
    plt.plot(x,base_perf,'--',label='Original')
    plt.plot(x,adv_perf,'-.',label='Challenge')
    plt.xticks(ticks=x,labels=x_labels)
    plt.xlabel('Number of fine-tuning examples')
    plt.ylabel('F1 [%]')
    plt.legend()
    plt.tight_layout()
    plt.savefig('../img/performance.png')

if True:
    adv_to_json('../data/train_data.json',
            '../data/sample1k-HCVerifySample.json','../data/add_one_sent.json')
    adv_to_json('../data/train_data.json',
            '../data/sample1k-HCVerifyAll.json','../data/add_sent.json')

if True:
    incorrect_preds('../output/base_when_eval_output/eval_predictions.jsonl',
        '../output/base_when_eval_output/incorrect_preds.json')
    incorrect_preds('../output/addsent_eval_output/eval_predictions.jsonl',
        '../output/addsent_eval_output/incorrect_preds.json',
        '../img/addsent_hist.png','../img/addsent_qlen.png')

if True:
    num_exs = [10,50,100,400,500,750,1000]
    for num_ex in num_exs:
        augment_training('../data/add_sent.json',
                '../data/'+str(num_ex)+'_train_data.json',num_ex)

if True:
    plot_performance()