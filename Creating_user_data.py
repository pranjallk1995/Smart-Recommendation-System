# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 12:37:18 2019

@author: Pranjall
"""

import numpy as np
import pandas as pd

dataset = pd.read_csv('Amazon_Fine_Food_data.csv') 
dataset = dataset.drop(columns = ['Summary', 'Text', 'Time', 'Id', 'ProfileName'])

dataset_by_user = dataset.groupby('UserId')

list_of_users = list(dataset_by_user.groups.keys())
user_dataset = pd.DataFrame(list_of_users, columns = ['UserId'])

Resp_to_question1 = np.random.randint(4, size = (len(list_of_users), 1))
Resp_to_question2 = np.random.randint(4, size = (len(list_of_users), 1))
Resp_to_question3 = np.random.randint(4, size = (len(list_of_users), 1))
Resp_to_question4 = np.random.randint(4, size = (len(list_of_users), 1))
Resp_to_question5 = np.random.randint(4, size = (len(list_of_users), 1))

Responses = np.concatenate((Resp_to_question1, Resp_to_question2, Resp_to_question3, Resp_to_question4, Resp_to_question5), axis = 1)
Total_resp_score = Responses.sum(axis = 1).reshape(len(Responses), 1) + np.random.uniform(low = -1, high = 1, size=(len(Responses), 1))          #adding some error to total score.
Total_resp_score = np.around(Total_resp_score)
Total_resp_score[Total_resp_score < 0] = 0
Total_resp_score = np.around(Total_resp_score / (16 / 5))
Responses = np.concatenate((Responses, Total_resp_score), axis = 1)

user_dataset['ResponseQ1'] = Responses[:, 0]
user_dataset['ResponseQ2'] = Responses[:, 1]
user_dataset['ResponseQ3'] = Responses[:, 2]
user_dataset['ResponseQ4'] = Responses[:, 3]
user_dataset['ResponseQ5'] = Responses[:, 4]
user_dataset['PersonalityType'] = Responses[:, 5]

user_dataset.to_csv('Amazon_user_data.csv', sep=',', index = False)