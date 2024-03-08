# Code refactored from https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps
# pip install -r requirements.txt
# <div class="primary-image svelte-wgcq7z"> image source

import streamlit as st
import openai
from openai import OpenAI

from operator import itemgetter
from langchain_community.vectorstores import faiss
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

import numpy as np
import base64
import requests
from PIL import Image
from io import BytesIO
from input_image import read_image
from combined import Recommender
import json
import sqlite3
import pickle
import dill
import config
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
import os
import re
import chat
import ast

# st.write(os.getcwd())
# load in data and models
@st.cache_resource
def load_data():
    model = Word2Vec.load(config.W2V_PATH)
    # load in tfdif model and encodings
    with open(config.PICKLE_FULL_PATH, 'rb') as f:
        full_recipes = pickle.load(f)
    with open(config.TFIDF_MODEL_PATH, 'rb') as f:
        tfidf = dill.load(f)
    with open(config.TFIDF_ENCODING_PATH, 'rb') as f:
        tfidf_encodings = dill.load(f)

    rec = Recommender(model, tfidf, tfidf_encodings, full_recipes)
    return model, full_recipes, tfidf, tfidf_encodings, rec

model, full_recipes, tfidf, tfidf_encodings, rec = load_data()

diets = [None, 'vegan', 'lactose free']
cuisines = [None, 'african', 'american', 'asian', 'creole and cajun', 
            'english','french','greek','indian','italian',
            'latin american','mediterranean','middle eastern',
            'spanish', 'thai']
types = [None, 'breakfast and brunch', 'main dish', 'side dish', 'salad', 'baked goods']
condiments = ['salt', 'pepper', 'oil']


# create page header
st.header(':cook: PicToPlate', divider='rainbow')
st.subheader("Welcome to PicToPlate! Cooking at home has never been easier.")

st.markdown('PicToPlate is an innovation application that allows home chefs to input an image of ingredients \
            they have, and PicToPlate will return a list of recipes they can cook using what they already have. \
            To get started, follow the instructions below.')

#create page caption
st.caption(":shallow_pan_of_food: A cooking assistant powered by OpenAI LLM")

#default on screen message
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Please upload an image so we can get started!"}]

#website sidebar, with api_key input and filters/sort
with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"

    



############# Set up chatbot ################
client = OpenAI(api_key=openai_api_key)

def generate_response(input_text):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=st.session_state.messages
    )
    return response.choices[0].message.content

def create_prompt(recipes_df, num_recipes=50):
    """Generate a prompt for the LLM with top recipes from the DataFrame."""
    recipes_text = ""
    for i, row in recipes_df.iterrows():
        recipes_text += f"{i + 1}. {row['name']} - Ingredients: {row['ingredients_x']}; Description: {row['tags']}...\n"
        if i >= num_recipes - 1:
            break  # Only use top num_recipes entries

    prompt = (f"I have found {num_recipes} recipes based on the ingredients provided. "
              "Please rank the best three recipes based on taste, healthiness, and simplicity:\n\n"
              f"{recipes_text}")
    return prompt

def get_top_three_recipes(recipes_df, openai_api_key, category=False):
    prompt = create_prompt(recipes_df)

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # Or another suitable engine
        messages=prompt,
        max_tokens=300,  # Adjust as needed
        temperature=0.7
    )

    return response.choices[0].text.strip()  # This should contain the ranking or selection of top recipes

def extract_recipe_indices(response_text):
    # Example assuming the response is "The best recipes are 1, 4, and 5."
    import re
    indices = [int(num) - 1 for num in re.findall(r'\d+', response_text)]  # Converts string numbers to list of indices
    return indices


if openai_api_key:
    llm = ChatOpenAI(
            temperature=0, model="gpt-3.5-turbo-0613", openai_api_key=openai_api_key, streaming=True
        )
if 'results' not in st.session_state:
    st.session_state['results'] = None

if 'pandas_df_agent' not in st.session_state:
    st.session_state['pandas_df_agent'] = None


################## SideBar and Inputs ###################################

with st.sidebar:
    st.subheader("Filters")
    with st.form('category_form', clear_on_submit=True):
        st.selectbox('Select Dietary Restriction', diets, key='diet')
        st.selectbox('Select Cuisines', cuisines,key='cuisine')
        st.selectbox('Select Meal Type', types, key='type')

        "---"
        submitted = st.form_submit_button('Save inputs')

if submitted:
    # TODO: merging the data based on submitted fields 
    categorical = str(st.session_state['diet']) + ' '+ str(st.session_state['cuisine']) +' ' + str(st.session_state['type'])
    st.write(f'Current categorical : {categorical}')
    
    response = generate_response(f'Use the provided ingredients to generate a {categorical} recipe. Do not have to use all of them')
    # Display response
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)



######################### Use input_image/image_reader instead ##########
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

def image_to_base64(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)

    # Convert the image to base64
    image_base64 = image_to_base64(image)

    if uploaded_file and not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")

    # User clicks the button to start processing
    if st.button('Identify Ingredients') and openai_api_key:
        # Clear existing chat history
        st.session_state["messages"] = []

        headers = {
            "Authorization": f"Bearer {openai_api_key}",
            "Content-Type": "application/json"
        }

        response = read_image.ingredient_identifier(image_base64, 'streamlit', openai_api_key)
        ing = json.loads(response['choices'][0]['message']['content'].strip('` \n').strip('json\n'))['items']
        ing_message = 'Looks like we have ' + ', '.join(ing) + ' available.'
        st.chat_message("assistant").write(ing_message)

        # st.session_state["messages"] = [{"role": "assistant", "content": ing_message}]
        initial_recs = rec.get_recommend(ing, 30)
        
        # st.session_state['results'] = output
        chatbot = chat.query_llm(initial_recs, openai_api_key)
        result = chatbot.invoke(f'what is 3 recipe that can be suits best with the available ingredients: {ing}? return as a list of their index')
        list1 = ast.literal_eval(result)
        result_lst = [i - 1 for i in list1]
        top_recipes_df = initial_recs[initial_recs.index.isin(result_lst)]

        st.dataframe(top_recipes_df)

        ##### 



################### Load in search ##################

if st.session_state.messages:
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

# User input
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    if st.session_state['pandas_df_agent'] is not None:
        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response = st.session_state['pandas_df_agent'].run(st.session_state.messages, callbacks=[st_cb])
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write(response)




