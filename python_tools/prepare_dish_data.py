import json
import numpy as np
from time import time
from sklearn.feature_extraction.text import CountVectorizer

restaurant_path = "../data/chinese_business.json"
review_path = "../data/chinese_review.json"
dish_path = "student_dn_annotations.txt"
output = "../data/dish_info.json"


def confirmedDish():
    dishes = []
    with open(dish_path, "r") as dish_file:
        for line in dish_file:
            dishes.append(line.lower().replace("'", "_").strip())
    return dishes


def aboutRestaurants():
    names = {}
    stars = {}
    with open(restaurant_path, "r") as restaurant_file:
        for line in restaurant_file.readlines():
            business_json = json.loads(line)
            business_id = business_json['business_id']
            business_name = business_json['name']
            business_stars = business_json['stars']

            names[business_id] = business_name
            stars[business_id] = business_stars

    return {"names": names, "stars": stars}


def main():
    # get restaurant info (name and stars)
    rest_info = aboutRestaurants()
    rest_names = rest_info['names']
    rest_stars = rest_info['stars']

    # get dish names
    dishes = confirmedDish()
    print("total " + str(len(dishes)) + " dishes")

    n_grams_max = 0
    for name in dishes:
        tmp = name.split(" ")
        if len(tmp) > n_grams_max:
            n_grams_max = len(tmp)
    print("n_grams_max = " + str(n_grams_max))

    review_text = {}  # review_id to text
    restaurant_text = {}  # business_id to text
    review_stars = []  # order of review to stars
    with open(review_path, "r") as review_file:
        for line in review_file:
            rev_content = json.loads(line.strip())
            content_text = rev_content["text"].lower().replace("'", "_").replace("\n", " ").strip()
            review_text[rev_content["review_id"]] = content_text

            business_id = rev_content["business_id"]
            if business_id in restaurant_text:
                restaurant_text[business_id] += " " + content_text
            else:
                restaurant_text[business_id] = content_text

            review_stars.append(rev_content["stars"])

    print("total " + str(len(restaurant_text)) + " restaurants")
    print("total " + str(len(review_text)) + " reviews")

    vectorizer = CountVectorizer(vocabulary=dishes, binary=True, lowercase=True, ngram_range=(1, n_grams_max))

    # fit review_text
    print("vectorze on review")
    t0 = time()
    X = vectorizer.fit_transform(review_text.values())
    print("done in %fs" % (time() - t0))
    print("n_samples: %d, n_features: %d" % X.shape)

    rev_dish_stars = np.zeros(len(dishes))
    rev_dish_count = np.zeros(len(dishes))

    for l, dish_vectortor in enumerate(X.toarray()):
        stars_of_current = review_stars[l]
        # here are dish vector of a review
        rev_dish_count = rev_dish_count + dish_vectortor
        # accumulate review stars by dishes
        rev_dish_stars = rev_dish_stars + dish_vectortor * stars_of_current  # from review_stars

    dish_voc = vectorizer.vocabulary_
    with open("rev_log.txt", "w") as log_file:
        for name in dish_voc:
            index = dish_voc[name]
            if rev_dish_count[index] != 0:
                log_file.write(name + " exist in " + str(rev_dish_count[index]) + " reviews, accumulate stars: " + str(
                        rev_dish_stars[index]) + "\n")

    # fit restaurant_text
    print("vectorze on restaurant")
    t0 = time()
    Y = vectorizer.fit_transform(restaurant_text.values())
    print("done in %fs" % (time() - t0))
    print("n_samples: %d, n_features: %d" % Y.shape)

    rest_dish_stars = np.zeros(len(dishes))
    rest_dish_count = np.zeros(len(dishes))

    rest_ids = restaurant_text.keys()
    for l, dish_vector in enumerate(Y.toarray()):
        stars_of_current = rest_stars[rest_ids[l]]
        # here are dish vector of a restaurant
        rest_dish_count = rest_dish_count + dish_vector
        # accumulate restaurant stars by dishes
        rest_dish_stars = rest_dish_stars + dish_vector * stars_of_current  # from rest_stars

    dish_voc = vectorizer.vocabulary_
    with open("rest_log.txt", "w") as log_file:
        for name in dish_voc:
            index = dish_voc[name]
            if rev_dish_count[index] != 0:
                log_file.write(
                    name + " exist in " + str(rest_dish_count[index]) + " restaurants, accumulate stars: " + str(
                            rest_dish_stars[index]) + "\n")

    # make the count json
    with open(output, "w") as output_file:
        for name in dish_voc:
            index = dish_voc[name]
            if rev_dish_count[index] == 0:
                continue
            info = {
                "name": name.replace("_", "'"),
                "review_count": rev_dish_count[index],
                "review_stars": rev_dish_stars[index],
                "restaurant_count": rest_dish_count[index],
                "restaurant_stars": rest_dish_stars[index]
            }
            output_file.write(json.dumps(info) + "\n")


if __name__ == "__main__":
    main()
