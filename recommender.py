from geopy.geocoders import Nominatim
from geopy import distance
import pandas as pd
import numpy as np
from operator import itemgetter
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine


class Recommender():
    def __init__(
        self,
        model: SentenceTransformer,
        loc: Nominatim,
        df: pd.DataFrame,
    ) -> None:
        self.model = model
        self.loc = loc
        self.df = df
        self.cuisine_styles = self._get_styles()
        self.cities = self._get_cities()
        self.weights = [0.35, 0.18, 0.12]     # rating, price_range, embeddings

    def _flatten_list(
        self,
        irregular_list: list,
    ) -> list:
        return [element for item in irregular_list for element in self._flatten_list(item)] if type(irregular_list) is list else [irregular_list]

    def _get_styles(self) -> list:
        return list(np.unique(self._flatten_list(list(self.df.cuisine_style))))

    def _get_cities(self) -> list:
        cities = {}
        for c in list(np.unique(self._flatten_list(list(self.df.city)))):
            cities[c] = (self.loc.geocode(c).latitude, self.loc.geocode(c).longitude)
        return cities

    def _find_nearest_cities(
        self,
        loc: Nominatim,
        cities_coords: dict,
        user_city: str,
    ) -> list:
        user_coords = (loc.geocode(user_city).latitude, loc.geocode(user_city).longitude)
        distance_dict = {}
        for c in cities_coords:
            distance_dict[c] = distance.geodesic(user_coords, cities_coords[c]).km
        return list(dict(sorted(distance_dict.items(), key = itemgetter(1))[:5]).keys())

    def _mean_similarity(
        self,
        user_embedding: np.ndarray, 
        reviews: list
    ) -> np.float32:
        return np.mean([cosine_similarity(user_embedding, r)[0][0] for r in reviews])
        
    def recommend(
        self,
        user: dict,
    ) -> pd.DataFrame:
        res = self.df

        nearest_cities = self._find_nearest_cities(self.loc, self.cities, user['city'])
        res.city = res.city.apply(lambda x: x if x in nearest_cities else float('nan'))
        res = res.dropna()

        res['cuisine_style_is_relevant'] = res.cuisine_style
        for style in user['cuisine_style']:
            res.cuisine_style_is_relevant = res.cuisine_style_is_relevant.apply(
                lambda x: True if x != True and style in x else x
            )
        res.cuisine_style_is_relevant = res.cuisine_style_is_relevant.apply(
            lambda x: float('nan') if x != True else x
        )
        res = res.dropna()

        user['embedding'] = self.model.encode([user['utterance']])
        res.embeddings = res.embeddings.map(
            lambda x: self._mean_similarity(user['embedding'], x)
        )

        res['score'] = res.apply(
            lambda x: cosine(
                user['vector'], 
                x[['rating', 'price_range', 'embeddings']],
                w=self.weights,
            ), axis=1
        )

        top10 = list(res.sort_values(by=['score'], ascending=False)[:10]['id_ta'])

        res.id_ta = res.id_ta.apply(lambda x: x if x in top10 else float('nan'))
        res = res.dropna().sort_values(by=['score'], ascending=False)

        return res