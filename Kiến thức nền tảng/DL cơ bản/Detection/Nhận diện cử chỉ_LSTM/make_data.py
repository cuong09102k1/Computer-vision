import pandas as pd
import numpy as np
Body_df = pd.read_csv("Body.txt")                                        #0
Dumbbell_Bicep_Curl_df = pd.read_csv("Dumbbell Bicep Curl.txt")          #1
Dumbbell_Shoulder_Press_df = pd.read_csv("Dumbbell Shoulder Press.txt")  #2


for i in range(68, 76):
    a = str(i)
    Body_df.insert(i+1, a, np.random.rand(len(Body_df)))
    Body_df = Body_df.rename(columns={'Unnamed: 0': ''})
# Lưu dataframe vào file CSV mới
Body_df.to_csv('Body_new.txt', index=False)

for i in range(68, 76):
    a = str(i)
    Dumbbell_Bicep_Curl_df.insert(i+1, a, np.random.rand(len(Dumbbell_Bicep_Curl_df)))
    Dumbbell_Bicep_Curl_df = Dumbbell_Bicep_Curl_df.rename(columns={'Unnamed: 0': ''})
# Lưu dataframe vào file CSV mới
Dumbbell_Bicep_Curl_df.to_csv('Dumbbell_Bicep_Curl_new.txt', index=False)

for i in range(68, 76):
    a = str(i)
    Dumbbell_Shoulder_Press_df.insert(i+1, a, np.random.rand(len(Dumbbell_Shoulder_Press_df)))
    Dumbbell_Shoulder_Press_df = Dumbbell_Shoulder_Press_df.rename(columns={'Unnamed: 0': ''})
# Lưu dataframe vào file CSV mới
Dumbbell_Shoulder_Press_df.to_csv('Dumbbell_Shoulder_Press_new.txt', index=False)

