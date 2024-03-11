from operator import itemgetter
import streamlit as st

from langchain_community.vectorstores import faiss
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts.prompt import PromptTemplate

from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
# from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_core.messages import AIMessage, HumanMessage, get_buffer_string
from langchain_core.prompts import format_document
# from langchain.vectorstores import DocArrayInMemorySearch
# from langchain.text_splitter import RecursiveCharacterTextSplitter

import re
import ast 
import os

class CustomDataChatbot:

    def __init__(self):
        # utils.configure_openai_api_key()
        self.openai_model = "gpt-3.5-turbo"

    # def save_file(self, file):
    #     folder = 'tmp'
    #     if not os.path.exists(folder):
    #         os.makedirs(folder)
        
    #     file_path = f'./{folder}/{file.name}'
    #     with open(file_path, 'wb') as f:
    #         f.write(file.getvalue())
    #     return file_path

    def create_prompt(self, recipes_df, num_recipes=30):
        """Generate a prompt for the LLM with top recipes from the DataFrame."""
        recipes_text = ""
        for i, row in recipes_df.iterrows():
            recipes_text += f"{i + 1}. {row['name']} - Ingredients: {row['ingredients_x']}; Description: {row['tags']}...\n"
            
        # prompt = (f"I have found {num_recipes} recipes based on the ingredients provided. "
        #           "Please rank the best three recipes based on available ingredients:\n\n"
        #           f"{recipes_text}")
        return recipes_text
    

    @st.spinner('Analyzing documents..')
    def query_llm(self, recs, openai_api_key, num_recipes=30):
        # Generate the prompt using the provided recipes
        context = self.create_prompt(recs, num_recipes)
        vectorstore = faiss.FAISS.from_texts(
        [context], embedding=OpenAIEmbeddings(openai_api_key = openai_api_key)
        )
        retriever = vectorstore.as_retriever()

        # template = """Answer the question based only on the following context:
        # {context}

        # Question: {question}
        # """
        # prompt = ChatPromptTemplate.from_template(template)

        # memory = ConversationBufferMemory(
        #     memory_key='chat_history',
        #     return_messages=True
        # )

        _template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

        Chat History:
        {chat_history}
        Follow Up Input: {question}
        Standalone question:"""
        CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

        template = """Answer the question based only on the following context:
        {context}

        Question: {question}
        """
        ANSWER_PROMPT = ChatPromptTemplate.from_template(template)
        DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")


        def _combine_documents(
            docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
        ):
            doc_strings = [format_document(doc, document_prompt) for doc in docs]
            return document_separator.join(doc_strings)



        memory = ConversationBufferMemory(
            return_messages=True, output_key="answer", input_key="question"
        )
        loaded_memory = RunnablePassthrough.assign(
            chat_history=RunnableLambda(memory.load_memory_variables) | itemgetter("history"),
        )

        standalone_question = {
            "standalone_question": {
                "question": lambda x: x["question"],
                "chat_history": lambda x: get_buffer_string(x["chat_history"]),
            }
            | CONDENSE_QUESTION_PROMPT
            | ChatOpenAI(temperature=0, openai_api_key = openai_api_key)
            | StrOutputParser(),
        }

        # Now we retrieve the documents
        retrieved_documents = {
            "docs": itemgetter("standalone_question") | retriever,
            "question": lambda x: x["standalone_question"],
        }
        # Now we construct the inputs for the final prompt
        final_inputs = {
            "context": lambda x: _combine_documents(x["docs"]),
            "question": itemgetter("question"),
        }
        # And finally, we do the part that returns the answers
        answer = {
            "answer": final_inputs | ANSWER_PROMPT | ChatOpenAI(openai_api_key = openai_api_key),
            "docs": itemgetter("docs"),
        }
        # And now we put it all together!
        final_chain = loaded_memory | standalone_question | retrieved_documents | answer



        # llm = ChatOpenAI(model_name=self.openai_model, temperature=0, streaming=True, openai_api_key = openai_api_key)
        # chain = (
        #     {"context": retriever, "question": RunnablePassthrough()}
        #     | prompt
        #     | llm
        #     | StrOutputParser()
        # )

        # qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory, verbose=True)
        return final_chain

    