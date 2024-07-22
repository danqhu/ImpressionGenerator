
import pandas as pd

file = pd.read_excel('US_qing.xlsx')


model_name = 'bard'

eval_itme = 'replaceability'

shot_list = [['zero', 'one'],['one', 'three'],['zero','three']]

for pair in shot_list:



    a_shot_type = pair[0]

    b_shot_type = pair[1]



    a = []

    b = []

    for index, row in file.iterrows():

        a_col_name = model_name + '_' + eval_itme + '_' + a_shot_type + '_'
        b_col_name = model_name + '_' + eval_itme + '_' + b_shot_type + '_'

        if row[a_col_name+'1'] =='是': a.append(1)
        if row[a_col_name+'2'] =='是': a.append(2)
        if row[a_col_name+'3'] =='是': a.append(3)
        if row[a_col_name+'4'] =='是': a.append(4)
        if row[a_col_name+'5'] =='是': a.append(5)

        if row[b_col_name+'1'] =='是': b.append(1)
        if row[b_col_name+'2'] =='是': b.append(2)
        if row[b_col_name+'3'] =='是': b.append(3)
        if row[b_col_name+'4'] =='是': b.append(4)
        if row[b_col_name+'5'] =='是': b.append(5)



    from scipy.stats import mannwhitneyu
    import numpy as np

    print(np.bincount(np.array(a)))
    print(np.bincount(np.array(b)))


    U1, p = mannwhitneyu(a, b)

    print('{} vs {} :'.format(a_shot_type, b_shot_type))

    print(U1)
    print(p)

