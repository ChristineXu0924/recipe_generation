import pandas as pd 
import numpy as np 
import ast

nlg_path = 'data/RecipeNLG_dataset.csv.zip'
foodcom_path = 'data/RAW_recipes.csv.zip'

# def dataset(nlg_path, foodcom_path):
nlg_recipe = pd.read_csv(nlg_path, compression='zip',  header=0, sep=',', quotechar='"')
nlg_recipe = nlg_recipe.drop(columns='Unnamed: 0')

url_features = nlg_recipe['link'].str.split('/',expand=True)
nlg_recipe['website'] = url_features[0]

# Split out the food.com subset, 
nlg_foodcom = nlg_recipe[nlg_recipe['website'] == 'www.food.com']

id_lst = []
for link in nlg_foodcom['link']:
    cur_id = int(link.split('-')[-1])
    id_lst.append(cur_id)

nlg_foodcom['id'] = id_lst
nlg_foodcom['title'] = nlg_foodcom['title'].str.lower()

foodcom = pd.read_csv(foodcom_path, compression='zip', header=0, sep=',', quotechar='"');
combined_foodcom = foodcom.merge(nlg_foodcom, left_on = 'id', right_on = "id", how='inner')
combined_foodcom['ingredients_x'] = combined_foodcom['ingredients_x'].apply(lambda x:ast.literal_eval(x))
    
    # return combined_foodcom 

