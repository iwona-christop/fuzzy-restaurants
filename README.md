# Fuzzy Restaurants

This project is an attempt to create a system recommending restaurants based on user preferences.

## Prerequisites

To use application, it is recommended to use a virtual environment. Go to the project directory and run the following line in terminal:

```bash
virtualenv -p python3.10 venv && source venv/bin/activate && python3.10 -m pip install -r requirements.txt && streamlit run app.py
```

The application should automatically open in a new tab in your browser.

## Dataset

The dataset comes from [Kaggle](https://www.kaggle.com/datasets/damienbeneschi/krakow-ta-restaurans-data-raw) and consists of information on restaurants in 31 European cities. The original file (name) consists of 125,433 entries and is structured as follow:

* **Name** - name of the restaurant
* **City** - city location of the restaurant
* **Cuisine Style**: cuisine style(s) of the restaurant, in a Python list object (94 046 non-null)
* **Ranking** - rank of the restaurant among the total number of restaurants in the city as a float object (115 645 non-null)
* **Rating** - rate of the restaurant on a scale from 1 to 5, as a float object (115 658 non-null)
* **Price Range** -  price range of the restaurant among 3 categories , as a categorical type (77 555 non-null)
* **Number of Reviews** - number of reviews that customers have let to the restaurant, as a float object (108 020 non-null)
* **Reviews** - reviews that are displayed on the restaurants scrolling page of the city, as a list of list object where the first list contains the 2 reviews, and the second le dates when these reviews were written (115 673 non-null)
* **URL_TA** - part of the URL of the detailed restaurant page that comes after `www.tripadvisor.com` as a string object (124 995 non-null)
* **ID_TA** - identification of the restaurant in the TA database constructed a one letter and a number (124 995 non-null)

Missing information for restaurants (for example unrated or unreviewed restaurants) were labeled as `NaN` (`numpy.nan`).

### Preprocessing

The dataset was processed as `pandas.DataFrame`. First, all missing values were removed, resulting in 74 225 items left in the collection. All column names have been changed to lower case, and spaces replaced with `_`.

Based on the `city` column, two more columns were added - `latitude` and `longitude`, which contain information about the city's geographic coordinates.

The values from the `cuisine_style` column were converted from `String` type to a list of cuisine style names.

The restaurant rating was recalculated as the quotient of `rating` and `number_of_reviews`.

The price range, represented by the characters $$$$, $$$-$$$$ and $$$$$$, was transformed into numeric values (successively) $0.33$, $0.67$, $1.0$. In the initial approach, values in the range of $0$ to $1$ were chosen, but they negatively affected the result of the final vector similarity.

All reviews were cleaned of unnecessary characters and then encoded using the `sentence-transformers/all-MiniLM-L6-v2` model to obtain sentence embeddings. This made it possible to obtain the data necessary for semantic comparison of sentences.

Columns `ranking` and `number_of_reviews` were removed and processed dataset was saved as `pickle`.


## Solution

A demo version of the system was created using the Streamlit package. Using a graphical interface, the user can specify:
* city or address (text input),
* price range (slider),
* cuisine style (multiselect),
* personal preferences (text input).

![demo](demo.png)

For a city (or address) provided by the user, its geographical coordinates and distance from each of the 31 cities available in the dataset are calculated. The five locations for which the distance from the selected location is the smallest are then selected.

The initial approach was to add the longitude and latitude values of the location to the user's preference vector. Due to the large size of the dataset, there was a need to abandon advanced calculations and the approach mentioned above was tried. As a result, recommendations were more likely to include restaurants in the user's immediate vicinity, and less likely to include those located far away but meeting the other criteria.

Thus, with the first step of the solution, the number of data to be processed in subsequent steps is significantly reduced.

The next step is to analyze the preferred types of cuisine, selected through a multiple-choice field, and leave in the database only those restaurants that offer at least one of the selected types of cuisine. This again reduces the number of records to be analyzed in subsequent steps.

The price range is selected using a slider with three options: $$$$, $$$$$ and $$$$$$, which are then converted to the following numerical values: $0.33$, $0.67$, $1.0$. As mentioned earlier, the initial approach selected values in the range of $0$ to $1$, but they negatively affected the result of the final vector similarity. The selected price range then appears in the user's preference vector.

The user's personal preferences are, like restaurant reviews, encoded using the `sentence-transformers/all-MiniLM-L6-v2` model to produce sentence embeddings. These are then compared to the embeddings of all restaurant reviews, resulting in cosine similarity - the restaurants with the highest value of this parameter are those whose reviews were most similar to the user's preferences.

As mentioned, a vector of user preferences is then created:

```python
user_vector = [0.0, price_range, 1.0]
```

 where $0.0$ is the highest possible restaurant rating, `price_range` is the price range converted to a floating point number, and $1.0$ is the highest possible cosine similarity value representing the semantic similarity between user preferences and restaurant reviews.
 
In a similar manner, its own feature vector was created for each restaurant:

```python
restaurant_vector = [rating, price_range, embedding_similarity]
```

In order to find the best method for comparing user and restaurant vectors, four approaches were tested. To select the best method, the value of the coefficient of mean square error ($MSE$), determined by the following formula, was used:

$$ MSE = \frac{1}{n} \sum_{i=1}^n \left( \hat{y_i} - y_i \right)^2 $$

where $\hat{y_i}$ - the value in the column `score` for each restaurant, and $y_i$ - the best possible score, i.e. the highest possible similarity or the smallest possible distance. The test results can be seen in the table below.

| Method             | Trial 1 | Trial 2 | Trial 3 |
| ------------------ | ------- | ------- | ------- |
| Hamming distance   | $0.778$ | $0.445$ | $0.445$ |
| Cosine similarity  | $0.368$ | $0.051$ | $0.421$ |
| Manhattan distance | $3.776$ | $4.002$ | $1.732$ |
| Euclidean distance | $1.396$ | $1.705$ | $0.968$ |

As can be seen in the table, the best result in each trial was achieved using cosine similarity, so it was decided not to undertake further testing and to create a system based on just this method.

For each attribute in the testing process, the weights shown in the table below were calculated.

| `rating` | `price_range` | `embeddings` | $MSE$   |
| -------- | ------------- | ------------ | ------- |
| $1.0$    | $1.0$         | $1.0$        | $0.234$ |
| $0.53$   | $0.5$         | $0.12$       | $0.295$ |
| $0.35$   | $0.18$        | $0.12$       | $0.164$ |
| $2.8$    | $5.7$         | $8.2$        | $0.237$ |

It was decided to adopt the weights $\left[0.35, 0.18, 0.12\right]$ because of the lowest value of the coefficient of mean square error. In addition, the selection of the given values solved the problem of the excessive influence of the value of the parameter `price_range` on the final result -- the clear difference between the values of the attribute resulted in a significant increase in the rank of the record.

Comparing the user and restaurant vectors triggered the calculation of the `score` value for each item, which contained values between $-1$ and $1$ and represented the cosine similarity of the vectors expressed by the formula:

$$ \cos\theta = \frac{\texttt{userVector}\cdot \texttt{restaurantVector}}{|\texttt{userVector}|\cdot |\texttt{restaurantVector}|} $$

The last step performed by the system is to sort the restaurants in descending order of the final score obtained and display to the user the 10 locations with the highest similarity value.

## Evaluation

An important issue in the design of a recommendation system is measuring its success, especially important from a business perspective. In order to determine a measure of classification accuracy, a study was conducted with the participation of a dozen participants. This helped determine the algorithm's ability to make effective decisions.

Each user was asked to use the system and determine which of the proposed recommendations were useful and met his expectations. The results of the study can be seen in the table below.

| Ranking | Marked as useful | Marked as useless |
| ------- | ---------------- | ----------------- |
| 1       | 10               | 2                 |
| 2       | 8                | 4                 |
| 3       | 9                | 3                 |
| 4       | 4                | 8                 |
| 5       | 7                | 5                 |
| 6       | 7                | 5                 |
| 7       | 7                | 5                 |
| 8       | 7                | 5                 |
| 9       | 8                | 4                 |
| 10      | 8                | 4                 |

$83.33\%$ users were most satisfied with the recommendation ranked highest. The overall usability of the recommendations equal to $62.5\%$.

The two main reasons for the low ratings of some recommendations were:
* large distance between the city given by the user and the location of the restaurant,
* the type of cuisine offered was not quite in line with the user's preferences.

## Conclusions

The created restaurant recommendation system fully meets the expectations and goals set at the beginning of the project. Tests with independent users have shown that the overall usability of recommendations is $62.5\%$. It can be assumed that this result would be significantly improved with a larger dataset.

One of the main problems experienced by users was the distance between the location given by the user and the location of the restaurant. This is a result of the limited dataset, which contained entries about restaurants from only 31 European cities, so that the system often directed the user to another country -- that's where the closest restaurants meeting the user's expectations were located. Limiting the number of searchable cities to the five closest significantly reduced this problem, but without adding more data to the collection, it is impossible to meet the expectations of all users.

The second problem was the incompatibility of the type of cuisine offered with the user's preferences, resulting from the fact that the system takes into account all restaurants where at least one of the types chosen by the user is available. With this solution, it is almost impossible to return to the user a number of recommendations less than 3. In the initial assumptions of the project, the types of cuisine were to be represented by vectors, but this method was abandoned due to the very low similarity values between them and the large distances.
