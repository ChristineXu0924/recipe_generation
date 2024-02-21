# Code refactored from https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps
# pip install -r requirements.txt

#TODO: move filters to side bar

import streamlit as st
import openai
from openai import OpenAI
import numpy as np
import base64
import requests
from PIL import Image
from io import BytesIO
from input_image import read_image
import json

# create website sidebar
with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"


# create page header
st.header(':cook: PicToPlate', divider='rainbow')

#create page subheader
st.subheader("Welcome to PicToPlate! Cooking at home has never been easier.")

#introduction to product
st.markdown('PicToPlate is an innovation application that allows home chefs to input an image of ingredients \
            they have, and PicToPlate will return a list of recipes they can cook using what they already have. \
            To get started, follow the instructions below.')

#create page caption
st.caption(":shallow_pan_of_food: A cooking assistant powered by OpenAI LLM")

#default on screen message
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Please identify all the food ingredients in the image, response in json format with the list of items."}]


# TODO: do we need streamlit_option_menu?


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
        headers = {
            "Authorization": f"Bearer {openai_api_key}",
            "Content-Type": "application/json"
        }

        # response = requests.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers)
        response = read_image.ingredient_identifier(image_base64, 'streamlit', openai_api_key)
        ing = json.loads(response['choices'][0]['message']['content'].strip('` \n').strip('json\n'))['ingredients']
        st.write(ing)
        
        # if response.status_code == 200:
        #     st.json(response.json())
        # else:
        #     st.error(f"Error in API response: {response.status_code}")

##----------------------------------------------------

################## Ask for ingredient editing ################
# Using a select form?
diets = [None, 'vegan', 'lactose free']
cuisines = [None, 'african', 'american', 'asian', 'creole and cajun', 
            'english','french','greek','indian','italian',
            'latin american','mediterranean','middle eastern',
            'russian and ukranian','spanish',
            'thai','turkish']
types = [None, 'breakfast and brunch', 'main dish', 'side dish', 'salad', 'baked goods']

with st.form('caegory_form', clear_on_submit=True):
    col1, col2, col3 = st.columns(3)
    col1.selectbox('Select Dietary Restriction', diets, key='diet')
    col2.selectbox('Select Cuisines', cuisines,key='cuisine')
    col3.selectbox('Select Meal Type', types, key='type')

    "---"
    with st.expander('Extra requirements'):
        requests = st.text_area("", placeholder='Enter extra requirements here...')

    submitted = st.form_submit_button('Save inputs')
    if submitted:

        # TODO: merging the data based on submitted fields 
        categorical = str(st.session_state['diet']) + str(st.session_state['cuisine']) + str(st.session_state['type'])
        words = str(st.session_state['Extra requirements'])

        # TODO: save into database for interaction
        st.write(f'Current categorical : {categorical}')
        st.write(f'Current additional comment: {words}')

                
##----------------------------------------------------

# for now suppose there is no change
################### Load in search ##################

# TODO: write the RAG in separated file and import here. How to deal with filters?
        

##----------------------------------------------------

#################### 

# # TODO: create separated menu? for the interactive guiding page

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    client = OpenAI(api_key=openai_api_key)

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    response = client.chat.completions.create(model="gpt-4-vision-preview", messages=st.session_state.messages)
    msg = response.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)
