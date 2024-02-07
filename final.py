import os
import sys
import pandas as pd
import numpy as np 
import json

# dependencies
from data import data 
from search import search_engine
from input_image import read_image 



## set recipe dataset from data, search_engine from search,
## preprocessing from input_image
## add parts for filter, should process before search.
## Anytime user change the filter, rerun the whole search component.

## search component is hidden from user interface
## connecting streamlit with the image input component, interact with the user
## then connect to search, with final input and filters 

## (optional) include a "final" button, so that start the interacting session 


# placeholder for future search and filters 
def filter_recipe(recipe, filter):
    return None 


## read in dataset once, and do not re-read it. filter on it instead 
def read_data():
    return None


def main():
    print("Please input the photo of your available ingredeints!")
#     photo = raw_input()
    photo = 'image1.jpg'
    out = read_image.ingredient_identifier(photo)
    query = json.loads(out['choices'][0]['message']['content'].strip('` \n').strip('json\n'))['ingredients']
    
    recipe = data.combined_foodcom
    
    input_text = []
    for i in recipe['ingredients_x']:
        input_text.append(', '.join(i))
        
    corpus = input_text
    
    similarity = search_engine.search(query, corpus)
    recipe['score'] = similarity 
    recipe = recipe.sort_values(by='score', ascending=False)[:8]

    print(recipe) 



if __name__ == "__main__":
    main()


