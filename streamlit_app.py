# Code refactored from https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps
# pip install -r requirements.txt
# <div class="primary-image svelte-wgcq7z"> image source

import streamlit as st
import openai
from openai import OpenAI

from langchain.agents import AgentType
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI

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


# connect to dataset
model = Word2Vec.load("models/model_cbow_2.bin")
# load in tfdif model and encodings
with open(config.PICKLE_FULL_PATH, 'rb') as f:
    full_recipes = pickle.load(f)
with open(config.TFIDF_MODEL_PATH, 'rb') as f:
    tfidf = pickle.load(f)
with open(config.TFIDF_ENCODING_PATH, 'rb') as f:
    tfidf_encodings = pickle.load(f)

rec = Recommender(model, tfidf, tfidf_encodings, full_recipes)


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



############# Set up OpenAI chatbot ################
client = OpenAI(api_key=openai_api_key)

def generate_response(input_text):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=st.session_state.messages
    )
    return response.choices[0].message.content

llm = ChatOpenAI(
        temperature=0, model="gpt-3.5-turbo-0613", openai_api_key=openai_api_key, streaming=True
    )
if 'results' not in st.session_state:
    st.session_state['results'] = None

if 'pandas_df_agent' not in st.session_state:
    st.session_state['pandas_df_agent'] = None


################## SideBar and Inputs ###################################

with st.sidebar:
# Using a select form?
    diets = [None, 'vegan', 'lactose free']
    cuisines = [None, 'african', 'american', 'asian', 'creole and cajun', 
                'english','french','greek','indian','italian',
                'latin american','mediterranean','middle eastern',
                'spanish', 'thai']
    types = [None, 'breakfast and brunch', 'main dish', 'side dish', 'salad', 'baked goods']


    with st.form('category_form', clear_on_submit=True):
        st.selectbox('Select Dietary Restriction', diets, key='diet')
        st.selectbox('Select Cuisines', cuisines,key='cuisine')
        st.selectbox('Select Meal Type', types, key='type')

        "---"
        submitted = st.form_submit_button('Save inputs')
if submitted:
    # TODO: merging the data based on submitted fields 
    categorical = str(st.session_state['diet']) + ' '+ str(st.session_state['cuisine']) +' ' + str(st.session_state['type'])
    # TODO: save into database for interaction
    st.write(f'Current categorical : {categorical}')
    # st.write(f'Current additional comment: {words}')
    
    response = generate_response(f'Use the provided ingredients to generate a {categorical} recipe. Do not have to use all of them')
    # Display response
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)



uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])


######################### Use input_image/image_reader instead ##########
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
        # response = read_image.ingredient_identifier(image_base64, openai_api_key)
        ing = json.loads(response['choices'][0]['message']['content'].strip('` \n').strip('json\n'))['items']
        ing_message = 'Looks like we have ' + ', '.join(ing) + ' available.'

        st.session_state["messages"] = [{"role": "assistant", "content": ing_message}]

        st.session_state['results'] = rec.get_recommend(ing, 5)

        # Create the DataFrame agent and store it in st.session_state
        st.session_state['pandas_df_agent'] = create_pandas_dataframe_agent(
            llm,
            st.session_state['results'],  # Use the results stored in session state
            verbose=True,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            handle_parsing_errors=True,
        )
        if st.session_state['pandas_df_agent'] is not None:
            # Construct an introductory message for the recipes
            recipe_intro = "Based on the ingredients identified, here are some recipe suggestions:"
            st.session_state.messages.append({"role": "assistant", "content": recipe_intro})
            st.chat_message("assistant").write(recipe_intro)
            
            # Generate a response with the recipes using the DataFrame agent
            # You may need to format your results DataFrame in a way that's easy to display as a message
            recipe_details = ""  # Initialize a string to store details of suggested recipes
            
            # Assuming 'results' contains columns 'Recipe Name', 'Ingredients', 'Instructions'
            # Modify this according to your actual DataFrame structure
            for index, row in st.session_state['results'].iterrows():
                recipe_details += f"**Recipe {index + 1}: {row['name']}**\n"
                recipe_details += f"Ingredients: {row['ingredients_x']}\n"
                recipe_details += f"Instructions: {row['steps'][:150]}... (more)\n\n"  # Truncate for brevity
            
            # Add the constructed recipe details to the chat
            st.session_state.messages.append({"role": "assistant", "content": recipe_details})
            st.chat_message("assistant").write("checking")
            st.chat_message("assistant").write(recipe_details)



# for now suppose there is no change
################### Load in search ##################

# TODO: write the RAG in separated file and import here. How to deal with filters?

# TODO: connect to database, retrieve the data and display the result. scrap picture too.



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




