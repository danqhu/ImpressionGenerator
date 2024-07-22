from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.spice.spice import Spice
import json

import pandas as pd
import numpy as np
import evaluate




def bleu():
    scorer = Bleu(n=4)
    # scorer += (hypo[0], ref1)   # hypo[0] = 'word1 word2 word3 ...'
    #                                 # ref = ['word1 word2 word3 ...', 'word1 word2 word3 ...']
    score, scores = scorer.compute_score(gts, res)

    print('belu = %s' % score)

def cider():
    scorer = Cider()
    # scorer += (hypo[0], ref1)
    (score, scores) = scorer.compute_score(gts, res)
    print('cider = %s' % score)

def meteor():
    scorer = Meteor()
    score, scores = scorer.compute_score(gts, res)
    print('meter = %s' % score)

def meteor_hugg():
    meteor =  evaluate.load('meteor')

    imgIds = list(gts.keys())

    scores = []
    for id in imgIds:
        hypo = res[id]
        ref  = gts[id]

        scores.append(meteor.compute(predictions=hypo, references=ref)['meteor'])

        # Sanity check.
        assert(type(hypo) is list)
        assert(len(hypo) == 1)
        assert(type(ref) is list)
        assert(len(ref) > 0)

    score = np.mean(np.array(scores))
    scores = np.array(scores)

    print('meter = %s' % score)

def rouge():
    scorer = Rouge()
    score, scores = scorer.compute_score(gts, res)
    print('rouge = %s' % score)

def spice():
    scorer = Spice()
    score, scores = scorer.compute_score(gts, res)
    print('spice = %s' % score)






def main():
    meteor_hugg()
    bleu()
    rouge()




extracted = True

EXAM_TYPE = 'CT'
MODEL_TYPE  = 'chatgpt 20231225'
reports = pd.read_excel('generated_impressions/R_肺癌{}报告 {} v3.0.xlsx'.format(EXAM_TYPE, MODEL_TYPE))

if extracted:
    reports = reports[['PATIENTIDENTIFIER', 'CONCLUSION_label', 'Response1', 'Response2', 'Response3']]
else:
    reports = reports[['PATIENTIDENTIFIER', 'CONCLUSION_label', 'Response1_old', 'Response2_old', 'Response3_old']]
    reports.columns = ['PATIENTIDENTIFIER', 'CONCLUSION_label', 'Response1', 'Response2', 'Response3']

reports = reports.dropna()


gts = {}
res = {}






reports['CONCLUSION_label'] = reports['CONCLUSION_label'].apply(lambda x: x.replace('_x000D_',''))

reports['Response1'] = reports['Response1'].str.rstrip('\n')
reports['Response1'] = reports['Response1'].str.lstrip('\n')
reports['Response1'] = reports['Response1'].apply(lambda x: x.replace(' ',''))
reports['Response1'] = reports['Response1'].replace(r'\n+','\n', regex=True)
reports['Response2'] = reports['Response2'].str.rstrip('\n')
reports['Response2'] = reports['Response2'].str.lstrip('\n')
reports['Response2'] = reports['Response2'].apply(lambda x: x.replace(' ',''))
reports['Response2'] = reports['Response2'].replace(r'\n+','\n', regex=True)
reports['Response3'] = reports['Response3'].str.rstrip('\n')
reports['Response3'] = reports['Response3'].str.lstrip('\n')
reports['Response3'] = reports['Response3'].apply(lambda x: x.replace(' ',''))
reports['Response3'] = reports['Response3'].replace(r'\n+','\n', regex=True)








for index, row in reports.iterrows():

    id = row['PATIENTIDENTIFIER']
    label = row['CONCLUSION_label']
    pred = row['Response1']

    gts[id] = [' '.join(pred)]
    res[id] = [' '.join(label)]
    temp = 0

main()






# gts = {}
# res = {}

# for index, row in reports.iterrows():

#     id = row['PATIENTIDENTIFIER']
#     label = row['CONCLUSION_label']
#     pred = row['Response2']

#     gts[id] = [' '.join(pred)]
#     res[id] = [' '.join(label)]
#     temp = 0

# main()







# gts = {}
# res = {}

# for index, row in reports.iterrows():

#     id = row['PATIENTIDENTIFIER']
#     label = row['CONCLUSION_label']
#     pred = row['Response3']

#     gts[id] = [' '.join(pred)]
#     res[id] = [' '.join(label)]
#     temp = 0



# main()
