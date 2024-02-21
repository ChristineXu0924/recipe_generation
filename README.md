# UCSD DSC Capstone Project: PicToPlate
link to static github website: https://christinexu0924.github.io/recipe_retrieval/
### Authors
Megan Huynh (mlhuynh@ucsd.edu)
Christine Xu (jix028@ucsd.edu)

# Introduction
The goal of this project is to provide users with a list of recipes and they can cook based and an AI cooking assistant on a user input image of their cooking ingredients. 

# How this Repo is Constructed

### Folders
- Data: the code for data preprocessing. For now, please download the original data from: https://drive.google.com/drive/folders/1xpnS0K3ecIbtySb0AxIhGMr7YiuI8_JP?usp=drive_link, and paste them into the data folder after cloning this repository. 
- input_image: contains the code for input image preprocessing. Currenly out of privacy concern, we masked our openai API key.  
- search: contains the code for our recipe search component

### Dependency and Enviornment Setup
After cloning the github repo to local repository and installing anaconda, cd into the directory and initiat a conda enviornment with command
```
conda create -n your_enviornment_name
conda activate your_enviornment_name
conda env create -f enviornment.yml
```
### Reproduce (terminal results)
In terminal, cd into the corresponding directory and run the following command to get the matching recipe result.
```
python final.py
```
### Running the Application
The link to our website is following: [https://reciperetrieval-mdhclv7xytdzvwane4r4sf.streamlit.app/](https://reciperetrieval-p8wpbszwyxjcy5n2quqyhr.streamlit.app/)
For now, user can insert their own openai API key to play around with the image input and the following recipe retrieval. We are still working on improving search results, personalization, and user interface. 
### Code Credit
We utilized preexisting code and models from many sources, including:
- GPT4 with Vision (https://platform.openai.com/docs/guides/vision)
- Streamlit app template (https://github.com/dataprofessor/openai-chatbot)


