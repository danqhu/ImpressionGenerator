import pandas as pd
import numpy as np




reports = pd.read_excel('肺癌CT报告 template.xlsx')
# reports= pd.read_excel('肺癌PETCT报告 v3.0.xlsx')
# reports = reports.dropna()
# reports = reports.head(100)


reports['CONCLUSION_label'] = reports['印象'].str.rstrip('\n')
reports['CONCLUSION_label'] = reports['CONCLUSION_label'].astype('string')
reports['CONCLUSION_label'].fillna('', inplace=True)
reports['CONCLUSION_label'] = reports['CONCLUSION_label'].apply(lambda x: x.replace(' ',''))

reports['FINDINGS_label'] = reports['报告发现'].str.rstrip('\n')
reports['FINDINGS_label'] = reports['FINDINGS_label'].astype('string')
reports['FINDINGS_label'].fillna('', inplace=True)
reports['FINDINGS_label'] = reports['FINDINGS_label'].apply(lambda x: x.replace(' ',''))
 
temp = 1
reports.to_excel('data/肺癌CT报告 template v3.0.xlsx', index=False)