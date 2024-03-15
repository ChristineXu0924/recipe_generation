# UCSD DSC Capstone Project: PicToPlate
Link to github website for details on our project: https://christinexu0924.github.io/recipe_retrieval/

Artifact repository: https://github.com/meganhuynh02/artifact-directory-template

### Authors
Megan Huynh (mlhuynh@ucsd.edu)
Jiarui(Christine) Xu (jix028@ucsd.edu)

## Introduction
The goal of this project is to provide users with a list of recipes and they can cook based and an AI cooking assistant on a user input image of their cooking ingredients. 

## How this Repo is Constructed

### Folders
```
├── docs
├── streamlit
│   ├── data
│   │   ├── data.py
│   │   └── full_recipes.pkl
│   ├── input_image 
│   │   ├── read_image.py
│   └── models
│   │   ├── model_cbow_2.bin
│   │   └── tfidf.pkl
│   │   └── tfidf_encodings.pkl
│   ├── chat.py
│   ├── combined.py
│   ├── config.py
│   ├── ingredient_parser.py
│   ├── requirements.txt
│   ├── streamlit_app.py
├── build.py
└── README.md
```
- docs: Source html and style.css code for our static website
- streamlit: This repository contains the implementations of our Pic2Plate Streamlit app.
  - data:
    - `full_recipes.pkl`: full recipe dataset from food.com.
  - input_image:
    - `read_image.py`: Code for recognizing ingredients from user input image. 
  - models: contains the model for our recipe search component
    - `model_cbow_2.bin`: Word2Vec model
    - `tfidf.pkl`: Embedded vectors of the original full dataset
    - `tfidf_encodings.pkl`: TF-IDF encoding model
  - `chat.py`: LangChain chatbot setup
  - `combined.py`: create Recommender object
  - `ingredient_parser.py`: code for data preprocessing (named-entity recognition)
  - `streamlit_app.py`: Our Pic2Plate recipe retrieval app

### Setup
After cloning the github repo to local repository and installing anaconda, cd into the directory and initiat a conda enviornment with command
```
conda create -n your_enviornment_name
conda activate your_enviornment_name
conda env create -f streamlit/requirements.txt
```
### Running
In terminal, cd into the corresponding directory and run the following command to launch the Streamlit app in your web browser.
```
streamlit run streamlit/streamlit_app.py
```
### Running the Application
The link to our website is following: https://pic2plate.streamlit.app/
For now, user can insert their own openai API key to play around with the image input and the following recipe retrieval. We are still working on improving search results, personalization, and user interface. 
### Code Credit
We utilized preexisting code and models from many sources, including:
- GPT4 with Vision (https://platform.openai.com/docs/guides/vision)
- Streamlit app template (https://github.com/dataprofessor/openai-chatbot)
- LangChain: Framework for LLMs applications (https://python.langchain.com/docs/get_started/introduction)


