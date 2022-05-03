import numpy as np
import pickle
import path
import process_functions as pf
path = pf.path

model = pickle.load(open(f"{path}/model/model.pkl", "rb"))
transformer = pickle.load(open(f"{path}/model/transformer.obj", "rb"))
user_input = input("Enter a review: ").strip()
review = pf.clean([user_input])
review = np.array(transformer.transform(review).toarray())


pf.predict(review, model)