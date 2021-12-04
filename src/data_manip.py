import json
import pandas as pd

test_data = pd.read_json('../data/test_data.json',lines=True)

def adv_to_json(input_file,output_file):
    adv_data = pd.read_json(input_file,lines=True)
    adv_data = pd.DataFrame(adv_data['data'][0])
    new_df = pd.DataFrame(columns=test_data.columns)
    for i,title in enumerate(adv_data['title']):
        paragraphs = adv_data['paragraphs'][i]
        for paragraph in paragraphs:
            for question in paragraph['qas']:
                qid = question['id']
                answers = {}
                for answer in question['answers']:
                    for key in answer.keys():
                        if key in answers:
                            answers[key].append(answer[key])
                        else:
                            answers[key] = [answer[key]]
                new_row = {'id':qid,'title':title,
                        'context':paragraph['context'],
                        'question':question['question'],'answers':answers}
                new_df = new_df.append(new_row,ignore_index=True)
    '''
    TODO: df
    '''
    with open(output_file,'w+') as file:
        for i in range(len(new_df)):
            json.dump(new_df.iloc[i].to_dict(),file,separators=(',',':'))
            file.write('\n')

adv_to_json('../data/sample1k-HCVerifyAll.json','../data/add_sent.json')
adv_to_json('../data/sample1k-HCVerifySample.json','../data/add_one_sent.json')
