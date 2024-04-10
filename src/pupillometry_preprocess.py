import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

temp = pd.read_excel(os.path.join("data", "raw", "pupillometry.xlsx")).values
temp_values = np.stack(temp[1:])
df = pd.DataFrame(data = temp_values, columns = temp[0]).rename(columns={"Patient ID": "id"}).set_index('id')

times_dict = {'A':0, 'B':1, 'C':2}
all_dfs = []

for session in range(1, 4):

    for recording in ['A', 'B', 'C']:

        columns = ["V{}{}_{}".format(session, recording, type) for type in ['moderate', 'hard']] # 'auditory',  # TODO : hyperparameter (what event type are used)
        scores = df[columns].sum(axis=1, skipna=False).values

        # Now we have all the data for all patients for one session number and one recording

        ids = df.index.values
        sessions = np.array([session]*len(ids))
        times = np.array([times_dict[recording]]*len(ids))
        temp_data = np.column_stack([ids, sessions, times, scores])
        temp_df = pd.DataFrame(data = temp_data,
                               columns = ['id', 'session', 'time', 'pupillometry_score']
                               )
        
        all_dfs.append(temp_df)

df = pd.concat(all_dfs, ignore_index=True)

nan_count = df['pupillometry_score'].isna().sum()

print(nan_count)

plt.hist(df['pupillometry_score'].values, bins = 50)
plt.title("Pupillometry score distribution")
plt.xlabel("Score (/10)")
plt.ylabel("Count")
plt.show()

pass