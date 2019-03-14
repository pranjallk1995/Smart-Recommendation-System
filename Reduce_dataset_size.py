# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 12:01:00 2019

@author: Pranjall
"""

import pandas as pd

dataset = pd.read_csv('Amazon_Fine_Food_data.csv') 
dataset = dataset.drop(columns = ['Summary', 'Text', 'Time', 'Id', 'ProfileName'])

dataset.to_csv('Amazon_Fine_Food_data_2.csv', sep=',', index = False)