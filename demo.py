from sentence_transformers import SentenceTransformer
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from geopy.geocoders import Nominatim
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import hamming
import time


def get_coordinates(loc=Nominatim(user_agent='GetLoc'), address=None):
    return loc.geocode(address).latitude, loc.geocode(address).longitude,

def flatten_list(irregular_list):
    return [element for item in irregular_list for element in flatten_list(item)] if type(irregular_list) is list else [irregular_list]

def mean_similarity(user_embedding: np.ndarray, reviews: list) -> np.float32:
    return np.mean([cosine_similarity(user_embedding, r)[0][0] for r in reviews])


# ===== Load dataset =====

df = pd.read_pickle('medium.pickle.compress', compression='gzip')

max_latitude = 60.1674881
max_longitude = 24.9427473

cuisine_styles = list(np.unique(flatten_list(list(df.cuisine_style))))
vectorizer = CountVectorizer()
vectorizer.fit(cuisine_styles)

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
loc = Nominatim(user_agent='GetLoc')

df['cuisine_vector'] = df.cuisine_style.map(lambda x: list(np.sum(vectorizer.transform(x).toarray(), axis=0)))
df.latitude = df.latitude.map(lambda x: x / max_latitude)
df.longitude = df.longitude.map(lambda x: x / max_longitude)


# ===== Streamlit app =====

st.title('Fuzzy Restaurants')

user_city = st.text_input('Where would you like to eat?')
user_price_range = st.select_slider('How much money do you want to spend?', options=['$', '$$', '$$$'])
user_cuisine_style = st.multiselect('What type of cuisine interests you?', cuisine_styles)
user_utterance = st.text_input('What are your preferences for the location?')

if st.button('Recommend'):
    # records start time
    start = time.perf_counter()

    user_latitude, user_longitude = get_coordinates(loc, user_city)
    user_cuisine_vector = vectorizer.transform(user_cuisine_style).toarray()[0]
    user_embedding = model.encode([user_utterance])
    user_price = 0.0 if user_price_range == '$' else 0.5 if user_price_range == '$$' else 1.0

    user_vector = [user_latitude / max_latitude, user_longitude / max_longitude, 1.0, 0.0, user_price, 1.0]  # cuisine_style=0.0, rating=0.0, similarity=1.0

    recommended = df

    # recommended.cuisine_vector = recommended.cuisine_vector.map(lambda x: hamming(user_cuisine_vector, x))
    recommended.cuisine_vector = recommended.cuisine_vector.map(lambda x: cosine_similarity([user_cuisine_vector], [x]))
    recommended.cuisine_vector = recommended.cuisine_vector.map(lambda x: float(x[0]))

    recommended.embeddings = recommended.embeddings.map(lambda x: mean_similarity(user_embedding, x))

    # recommended['score'] = recommended.apply(
    #     lambda x: hamming(
    #         user_vector, 
    #         x[['latitude', 'longitude', 'cuisine_vector', 'rating', 'price_range', 'embeddings']]
    #     ), axis=1
    # )

    recommended['score'] = recommended.apply(
        lambda x: cosine_similarity(
            [user_vector], 
            [x[['latitude', 'longitude', 'cuisine_vector', 'rating', 'price_range', 'embeddings']]]
        ), axis=1
    )
    recommended.score = recommended.score.map(lambda x: float(x[0]))

    # top10 = list(recommended.sort_values(by=['score'], ascending=True)[:10]['id_ta'])
    top10 = list(recommended.sort_values(by=['score'], ascending=False)[:10]['id_ta'])


    recommended.id_ta = recommended.id_ta.apply(lambda x: x if x in top10 else float('nan'))
    recommended = recommended.dropna().sort_values(by=['score'], ascending=False)

    i = 1
    for _, row in recommended.iterrows():
        st.markdown(f'''
            ## {i}. [{row['name']}](https://tripadvisor.com{row['url_ta']})
            ### {row['city']}

            Cusine style: {', '.join(row['cuisine_style'])}

            Price range: {'$' if row['price_range'] == 0.0 else '$$' if row['price_range'] == 0.5 else '$$$'}

            Reviews: {', '.join(row['reviews'])}
        ''')
        i += 1


    # st.dataframe(recommended)

    # record end time
    end = time.perf_counter()

    # find elapsed time in seconds
    ms = (end-start)
    st.write(f"Elapsed {ms:.03f} s.")