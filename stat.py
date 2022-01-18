import numpy as np
import pandas as pd

df = pd.read_csv('./base_de_données.csv')

q_tache = df['q_tache'].fillna(0)
Y = q_tache.to_numpy(np.float32)
Y = (Y > 3)
Y = np.int16(Y)

# print(Y)
# print(np.count_nonzero(Y))

q_coulure = df['q_coulure'].fillna(0)
Y = q_coulure.to_numpy(np.float32)
Y = (Y > 3)
Y = np.int16(Y)

# print(Y)
# print(np.count_nonzero(Y))

q_givrage = df['q_givrage'].fillna(0)
Y = q_givrage.to_numpy(np.float32)
Y = (Y > 3)
Y = np.int16(Y)

print(Y)
print(np.count_nonzero(Y))