import pandas as pd
import numpy as np




reports = pd.read_excel('肺癌PETCT报告 v2.1.xlsx')
# reports= pd.read_excel('肺癌PETCT报告 v3.0.xlsx')
# reports = reports.dropna()
# reports = reports.head(100)


reports['CONCLUSION_label'] = reports['CONCLUSION.1'].str.rstrip('_x000D_\n')
reports['CONCLUSION_label'] = reports['CONCLUSION_label'].apply(lambda x: x.replace('_x000D_',''))

reports['FINDINGS2'] = reports['FINDINGS.1'].str.rstrip('_x000D_\n')
reports['FINDINGS2'] = reports['FINDINGS2'].apply(lambda x: x.replace('_x000D_',''))
 

reports.to_excel('data/肺癌PETCT报告 v3.0.xlsx', index=False)