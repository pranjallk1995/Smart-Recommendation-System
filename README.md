# Smart-Recommendation-System
A recommendation system that recommends products to customers by learning their psychology.

Important: Original dataset 'Amazon_Fine_Food_data.csv' was too large to upload, hence not present here.

1.) Get the dataset and make sure the file name is 'Amazon_Fine_Food_data_2.csv'.

2.) Put the dataset and all the code files in the same directory.

3.) First run 'Creating_user_data.py' to create a sample dataset of users and their presonality types, 'Amazon_user_data.csv'.
    This data represents already studied users and forms the training/testing data for the following presonality predition model.
    
4.) Now run 'User_psychology_prediction_system.py' to train an Artificial Neural Network that predicts users personality type.
    An accuracy of appox. 85% is achieved.
    
5.) Finally, run the 'Smart_recommendation_system.py' file to get the recommended items for specific personality types, sorted based on product rating.
