import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from openpyxl import load_workbook


def get_drug_ids_from_colors(path, column):
    sheet = load_workbook(path, data_only = True)['Ark1']

    color_to_drug = {8:3, # blue is drug 3
                     9:2, # green is drug 2
                     'FFFFC000':1, # orange is drug 1
                     '00000000':np.nan} # empty cell

    drug_ids = [color_to_drug[cell[0].fill.start_color.index] for cell in sheet['{}3:{}52'.format(column, column)]]

    return drug_ids


path = os.path.join("data", "raw", "pupillometry_colored.xlsx")

temp = pd.read_excel(path).values
temp_values = np.stack(temp[1:])
df = pd.DataFrame(data = temp_values, columns = temp[0]).rename(columns={"Patient ID": "id"}).set_index('id')

times_dict = {'A':0, 'B':1, 'C':2}
columns_dict = {1:"B", 2:"K", 3:"T"} # which column to read the drug colors from in the xlsx file, based on session number
all_dfs = []

for session in range(1, 4):
        
    drug_ids = get_drug_ids_from_colors(path, columns_dict[session]) # only one drug used by session

    for recording in ['A', 'B', 'C']:

        columns = ["V{}{}_{}".format(session, recording, type) for type in ['auditory', 'moderate', 'hard']] # TODO : hyperparameter (what event types are used)
        scores = df[columns].sum(axis=1, skipna=False).values

        # Now we have all the data for all patients for one session number and one recording

        ids = ["{}".format(i) for i in df.index.values] # needed for joining on other DF in next script
        times = np.array([times_dict[recording]]*len(ids))
        temp_data = np.column_stack([ids, drug_ids, times, scores])
        temp_df = pd.DataFrame(data = temp_data,
                               columns = ['id', 'drug', 'time', 'pupillometry_score'])
        
        all_dfs.append(temp_df)

df = pd.concat(all_dfs, ignore_index=True)
df.to_csv(os.path.join("data", "processed", "pupillometry_features.csv"), index = False)

nan_count = df['pupillometry_score'].isna().sum()
print("{} values are missing".format(nan_count))
plt.hist(df['pupillometry_score'].values, bins = 50)
plt.title("Pupillometry score distribution")
plt.xlabel("Score (/13)")
plt.ylabel("Count")
plt.show()