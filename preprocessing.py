import pandas as pd
from geopy.geocoders import Nominatim
import re
from sentence_transformers import SentenceTransformer


def clean_reviews(reviews):
    reviews = reviews.split(',')
    reviews = [re.sub(r'[^\w\s]', '', r) for r in reviews]
    reviews = [re.sub(r'^\s', '', r) for r in reviews]
    reviews = [re.sub(r'\d+', '', r) for r in reviews]
    reviews = [r for r in reviews if r != '']
    return reviews

def text_to_embeddings(model, reviews):
    return [model.encode([r]) for r in reviews]


if __name__ == '__main__':
    # Read original dataset and drop NaN
    df = pd.read_csv('TA_restaurants_curated.csv')
    df = df.dropna()

    # Change all names to lowcase and whitespaces to '_'
    df.columns = df.columns.str.lower().str.replace(' ', '_')

    # Add latitude and longitude of the cities
    cities = list(df.city.unique())
    coordinates = {}

    loc = Nominatim(user_agent='GetLoc')

    for c in cities:
        coordinates[c] = {'latitude' : loc.geocode(c).latitude, 'longitude': loc.geocode(c).longitude}
    
    df['latitude'] = df.city.map(lambda x: coordinates[x]['latitude'])
    df['longitude'] = df.city.map(lambda x: coordinates[x]['longitude'])

    # Turn cuisine styles into lists of strings
    df.cuisine_style = df.cuisine_style.map(lambda x: re.sub(r'\W+', ' ', x).split(' ')[1:-1])

    # Count rating as rating/number_of_reviews
    df.rating = df.rating / df.number_of_reviews

    # Delete ranking, number of reviews and unnamed: 0
    df = df.drop(['unnamed:_0', 'ranking', 'number_of_reviews'], axis=1)

    # Translate price range
    df.price_range = df.price_range.map(lambda x: 0.0 if x == '$' else 0.5 if x == '$$ - $$$' else 1.0)

    # Clean reviews and add embeddings
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    df.reviews = df.reviews.map(lambda x: float('nan') if x == '[[], []]' else x)
    df = df.dropna()
    df.reviews = df.reviews.map(lambda x: clean_reviews(x))

    df['embeddings'] = df.reviews.map(lambda x: text_to_embeddings(model, x))

    # Save as a pickle Rick
    df.to_pickle('data.pickle.compressed', compression='gzip')