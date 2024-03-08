from operator import itemgetter

from langchain_community.vectorstores import faiss
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

import re
import ast 


def create_prompt(recipes_df, num_recipes=30):
    """Generate a prompt for the LLM with top recipes from the DataFrame."""
    recipes_text = ""
    for i, row in recipes_df.iterrows():
        recipes_text += f"{i + 1}. {row['name']} - Ingredients: {row['ingredients_x']}; Description: {row['tags']}...\n"
        
    # prompt = (f"I have found {num_recipes} recipes based on the ingredients provided. "
    #           "Please rank the best three recipes based on available ingredients:\n\n"
    #           f"{recipes_text}")
    return recipes_text

def query_llm(recs, openai_api_key, num_recipes=30):
    # Generate the prompt using the provided recipes
    context = create_prompt(recs, num_recipes)
    vectorstore = faiss.FAISS.from_texts(
    [context], embedding=OpenAIEmbeddings(openai_api_key = openai_api_key)
    )
    retriever = vectorstore.as_retriever()
    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    model = ChatOpenAI(openai_api_key = openai_api_key)
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    return chain

def extract_recipe_indices(response_text):
    return None
