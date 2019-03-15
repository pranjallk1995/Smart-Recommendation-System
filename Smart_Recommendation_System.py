# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 08:23:52 2019

@author: Pranjall
"""

import pandas as pd

if __name__ == '__main__':
    
    #Importing dataset.
    user_data = pd.read_csv('Amazon_user_data.csv')
    product_data = pd.read_csv('Amazon_Fine_Food_data_2.csv')
    
    #Joining datasets.
    dataset = pd.merge(product_data, user_data, on = 'UserId')
    
    #Sorting data based on score.
    dataset = dataset.sort_values(by = ['Score'], ascending = False)
    
    #Grouping data by personality type.
    dataset_by_personality = dataset.groupby('PersonalityType')
    
    #Extracting useful features.
    products_recommended_by_personality = []
    types_of_personalities = dataset['PersonalityType'].unique()
    for i in range(0, len(types_of_personalities)):
        products_recommended_by_personality.append(dataset_by_personality.get_group(i))    
    for i in range(0, len(types_of_personalities)):
        products_recommended_by_personality[i] = products_recommended_by_personality[i].iloc[:, [0, 4, 10]]
        
    #Writing results in a file.
    products_recommended_by_personality[0].to_csv('Products_for_type_0_people.csv', sep=',', index = False)
    products_recommended_by_personality[1].to_csv('Products_for_type_1_people.csv', sep=',', index = False)
    products_recommended_by_personality[2].to_csv('Products_for_type_2_people.csv', sep=',', index = False)
    products_recommended_by_personality[3].to_csv('Products_for_type_3_people.csv', sep=',', index = False)
    products_recommended_by_personality[4].to_csv('Products_for_type_4_people.csv', sep=',', index = False)
    products_recommended_by_personality[5].to_csv('Products_for_type_5_people.csv', sep=',', index = False)