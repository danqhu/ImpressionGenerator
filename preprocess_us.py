import pandas as pd
import numpy as np


US_TYPE = 'abdomen'
US_TYPE = 'heart'

reports = pd.read_excel('肺癌US {}报告 template.xlsx'.format(US_TYPE))



reports['CONCLUSION_label'] = reports['CONCLUSION'].str.rstrip('\n')
reports['CONCLUSION_label'] = reports['CONCLUSION_label'].astype('string')
reports['CONCLUSION_label'].fillna('', inplace=True)
reports['CONCLUSION_label'] = reports['CONCLUSION_label'].apply(lambda x: x.replace(' ',''))
reports['CONCLUSION_label'] = reports['CONCLUSION_label'].apply(lambda x: x.replace('_x000D_',''))

reports['FINDINGS_label'] = reports['FINDINGS'].str.rstrip('\n')
reports['FINDINGS_label'] = reports['FINDINGS_label'].astype('string')
reports['FINDINGS_label'].fillna('', inplace=True)
reports['FINDINGS_label'] = reports['FINDINGS_label'].apply(lambda x: x.replace(' ',''))
reports['FINDINGS_label'] = reports['FINDINGS_label'].apply(lambda x: x.replace('_x000D_',''))
 
temp = 1
reports.to_excel('data/肺癌US {}报告 template v3.0.xlsx'.format(US_TYPE), index=False)