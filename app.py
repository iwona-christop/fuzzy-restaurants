import streamlit as st
from sentence_transformers import SentenceTransformer
from geopy.geocoders import Nominatim
from pandas import read_pickle
from recommender import Recommender


recommender = Recommender(
    model=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'),
    loc=Nominatim(user_agent='GetLoc'),
    df=read_pickle('large.pickle.compress', compression='gzip'),
)


st.title('Fuzzy Restaurants')

with st.form('user_preferences'):
    user_city = st.text_input('Where would you like to eat?')
    user_price_range = st.select_slider('How much money do you want to spend?', options=['$', '$$', '$$$'])
    user_cuisine_style = st.multiselect('What type of cuisine interests you?', recommender.cuisine_styles)
    user_utterance = st.text_input('What are your preferences for the location?')

    if st.form_submit_button('Recommend'):
        result = recommender.recommend(
            user={
                'city' : user_city,
                'cuisine_style' : user_cuisine_style,
                'utterance' : user_utterance,
                'vector' : [
                    0.0, # max rating
                    1/3 if user_price_range == '$' else 2/3 if user_price_range == '$$' else 1.0, 
                    1.0 # max sentence similarity
                ]
            }
        )

        i = 1
        for _, row in result.iterrows():
            st.markdown(f'''
                ## {i}. [{row['name']}](https://tripadvisor.com{row['url_ta']})
                ### {row['city']}

                Cusine style: {', '.join(row['cuisine_style'])}

                Price range: {'$' if row['price_range'] == 1/3 else '$$' if row['price_range'] == 2/3 else '$$$'}

                Reviews: {', '.join(row['reviews'])}
            ''')
            i += 1