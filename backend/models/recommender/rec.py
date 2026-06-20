import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

df2 = pd.read_csv('./models/recommender/final.csv')
makeup = pd.read_csv('./models/recommender/makeup_final.csv')
entries = len(df2)
LABELS = list(df2.label.unique())

features = [
    'normal', 'dry', 'oily', 'combination', 'acne', 'sensitive', 'fine lines', 'wrinkles',
    'redness', 'dull', 'pore', 'pigmentation', 'blackheads', 'whiteheads', 'blemishes',
    'dark circles', 'eye bags', 'dark spots',
]

# Skin-type dimensions are weighted higher than optional concern tags
FEATURE_WEIGHTS = np.array([
    2.0, 2.0, 2.0, 2.0,   # skin types
    2.5,                   # acne
    1.5, 1.5, 1.5, 1.5, 1.2, 1.2, 1.5, 1.5, 1.5, 1.2, 1.2, 1.2, 1.5,
])


def wrap(info_arr):
    return {
        'brand': info_arr[0],
        'name': info_arr[1],
        'price': info_arr[2],
        'url': info_arr[3],
        'img': info_arr[4],
        'skin type': info_arr[5],
        'concern': str(info_arr[6]).split(','),
    }


def wrap_makeup(info_arr):
    return {
        'brand': info_arr[0],
        'name': info_arr[1],
        'price': info_arr[2],
        'url': info_arr[3],
        'img': info_arr[4],
        'skin type': info_arr[5],
        'skin tone': info_arr[6],
    }


one_hot_encodings = np.zeros([entries, len(features)])

for i in range(entries):
    sk_type = str(df2.iloc[i]['skin type']).lower()
    if sk_type == 'all':
        one_hot_encodings[i][0:4] = 1
    elif sk_type in features:
        one_hot_encodings[i][features.index(sk_type)] = 1

for i in range(entries):
    concern_text = str(df2.iloc[i]['concern']).lower()
    for j in range(5, len(features)):
        if features[j] in concern_text:
            one_hot_encodings[i][j] = 1

weighted_encodings = one_hot_encodings * FEATURE_WEIGHTS


def recs_cs(vector=None, name=None, label=None, count=5, skin_type=None):
    products = []
    if name:
        idx = df2[df2["name"] == name].index.tolist()[0]
        fv = one_hot_encodings[idx]
    elif vector is not None:
        fv = np.array(vector, dtype=float)
    else:
        return products

    weighted_user = (fv * FEATURE_WEIGHTS).reshape(1, -1)
    cs_values = cosine_similarity(weighted_user, weighted_encodings)[0]

    dff = df2.copy()
    dff['cs'] = cs_values

    if label:
        dff = dff[dff['label'] == label]

    if skin_type and skin_type != 'all':
        dff = dff[dff['skin type'].isin([skin_type, 'all'])]

    if name:
        dff = dff[dff['name'] != name]

    recommendations = dff.sort_values('cs', ascending=False).head(count)
    data = recommendations[['brand', 'name', 'price', 'url', 'img', 'skin type', 'concern']].to_dict('split')['data']
    for element in data:
        products.append(wrap(element))
    return products


def recs_essentials(vector=None, name=None, skin_type=None):
    response = {}
    for label in LABELS:
        if name:
            r = recs_cs(name=name, label=label, skin_type=skin_type)
        elif vector is not None:
            r = recs_cs(vector=vector, label=label, skin_type=skin_type)
        else:
            r = []
        response[label] = r
    return response


def makeup_recommendation(skin_tone, skin_type):
    result = []
    skin_type = str(skin_type).lower()

    def pick(category, limit=2):
        exact = makeup[
            (makeup['skin tone'] == skin_tone)
            & (makeup['skin type'] == skin_type)
            & (makeup['label'] == category)
        ]
        if len(exact) >= limit:
            return exact.head(limit)
        relaxed = makeup[
            (makeup['skin type'] == skin_type) & (makeup['label'] == category)
        ]
        if len(relaxed) >= limit:
            return relaxed.head(limit)
        return makeup[makeup['label'] == category].head(limit)

    parts = [pick('foundation'), pick('concealer'), pick('primer')]
    dff = pd.concat(parts, ignore_index=True).drop_duplicates(subset=['name'])
    if dff.empty:
        return result

    dff = dff.sample(frac=1, random_state=42)
    data = dff[['brand', 'name', 'price', 'url', 'img', 'skin type', 'skin tone']].to_dict('split')['data']
    for element in data:
        result.append(wrap_makeup(element))
    return result
