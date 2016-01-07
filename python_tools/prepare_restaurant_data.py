import json
import numpy as np
from time import time
from sklearn.feature_extraction.text import CountVectorizer

restaurant_path = "../data/chinese_business.json"
review_path = "../data/chinese_review.json"
dish_path = "student_dn_annotations.txt"
output = "../data/dish_restaurant_info.json"


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
    review_restaurant = []  # order of review to business_id
    review_stars = []  # order of review to stars
    with open(review_path, "r") as review_file:
        for line in review_file:
            rev_content = json.loads(line.strip())
            content_text = rev_content["text"].lower().replace("'", "_").replace("\n", " ").strip()
            review_text[rev_content["review_id"]] = content_text

            review_restaurant.append(rev_content["business_id"])
            review_stars.append(rev_content["stars"])

    print("total " + str(len(review_text)) + " reviews")

    vectorizer = CountVectorizer(vocabulary=dishes, binary=True, lowercase=True, ngram_range=(1, n_grams_max))

    # fit review_text
    print("vectorze on review")
    t0 = time()
    X = vectorizer.fit_transform(review_text.values())
    print("done in %fs" % (time() - t0))
    print("n_samples: %d, n_features: %d" % X.shape)

    dish_restaurant_stars = {}  # dish order to map( business_id to stars )
    rest_in_review = {}  # business_id to restaurant name , for those restaurant mentioned in reviews
    for l, dish_vectortor in enumerate(X.toarray()):
        stars_of_current = review_stars[l]
        business_of_current = review_restaurant[l]
        name_of_current = rest_names[business_of_current]

        # add to map
        rest_in_review[business_of_current] = name_of_current

        for m, count in enumerate(dish_vectortor):
            if (count == 0):
                continue

            if m not in dish_restaurant_stars:
                dish_restaurant_stars[m] = {}

            if business_of_current not in dish_restaurant_stars[m]:
                dish_restaurant_stars[m][business_of_current] = 0
                # dish_restaurant_stars[m][business_of_current] = {"count":0,"stars":0}

            dish_restaurant_stars[m][business_of_current] += stars_of_current
            # dish_restaurant_stars[m][business_of_current]["count"] += 1
            # dish_restaurant_stars[m][business_of_current]["stars"] += stars_of_current

    with open("dish_log.txt", "w") as log_file:
        for l, dish_index in enumerate(dish_restaurant_stars):
            dish_name = dishes[dish_index]
            log_file.write(dish_name + " : " + str(dish_restaurant_stars[dish_index]))
    dish_data = {}  # dish name to map (business_id to stars)
    for l, dish_index in enumerate(dish_restaurant_stars):
        dish_name = dishes[dish_index]
        before_sort = dish_restaurant_stars[dish_index]
        sorted_business = sorted(before_sort, key=before_sort.__getitem__, reverse=True)  # list of business_id
        after_sort = {}  # business_id to accumulated stars
        for business_id in sorted_business[:100]:
            after_sort[business_id] = before_sort[business_id]
        dish_data[dish_name] = after_sort  # keep only top 100 restaurants about one dish

    result_data = {"restaurant_names": rest_in_review,
                   "dish_data": dish_data
                   }

    with open(output, "w") as output_file:
        output_file.write(json.dumps(result_data))


if __name__ == "__main__":
    main()
