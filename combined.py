import pandas as pd
import numpy as np
import pickle
import dill
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec

import config
from ingredient_parser import ingredients_parser

# model = Word2Vec.load("models/model_cbow_2.bin")
# # load in tfdif model and encodings
# with open(config.PICKLE_FULL_PATH, 'rb') as f:
#     full_recipes = pickle.load(f)
# with open(config.TFIDF_MODEL_PATH, 'rb') as f:
#     tfidf = pickle.load(f)
# with open(config.TFIDF_ENCODING_PATH, 'rb') as f:
#     tfidf_encodings = dill.load(f)

class Recommender:
    def __init__(self, model, tfidf, tfidf_encodings, data):
        self.model = model
        self.doc_vec = tfidf
        self.tfidf_vec_tr = tfidf_encodings
        self.recipe = data

    def get_and_sort_corpus(self, data):
        corpus_sorted = []
        for doc in data.parsed.values:
            doc.sort()
            corpus_sorted.append(doc)
        return corpus_sorted

    def recommendation_results(self, scores, N=1000):
        # df_recipes = pd.read_csv(config.RECIPES_PATH)
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:N]
        # recommendation = pd.DataFrame(columns = ['recipe', 'ingredients', 'score', 'url'])
        # count = 0
        # for i in top:
        #     recommendation.at[count, "recipe"] = df_recipes["name"][i]
        #     recommendation.at[count, "ingredients"] = df_recipes['ingredients_x'][i]
        #     recommendation.at[count, "url"] = df_recipes["link"][i]
        #     recommendation.at[count, "score"] = f"{scores[i]}"
        #     count += 1
        return self.recipe[self.recipe.index.isin(top)]
    
    def get_recommend(self, in_ing, N=1000):
        # in_ing = in_ing.split(", ")
        in_ing = [x.lower() for x in in_ing]
        # corpus = self.get_and_sort_corpus(self.data['ingredients_parsed'])
        in_ing = ingredients_parser(in_ing)
        # get embeddings for ingredient doc
        input_embedding = self.tfidf_vec_tr.transform([in_ing])[0].reshape(1, -1)

        # get cosine similarity between input embedding and all the document embeddings
        cos_sim = map(lambda x: cosine_similarity(input_embedding, x)[0][0], self.doc_vec)
        scores = list(cos_sim)
        return self.recommendation_results(scores, N)

