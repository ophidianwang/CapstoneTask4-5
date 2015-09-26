import json
import numpy as np
from time import time
from sklearn.feature_extraction.text import CountVectorizer

restaurant_path = "../data/chinese_business.json"
review_path = "../data/chinese_review.json"
dish_path = "student_dn_annotations.txt"
output = "../data/restaurant_info.json"

target_dish = "fried rice"
target_dish = target_dish.lower().replace("'","_").strip()

def confirmedDish():
    dishes = []
    with open(dish_path, "r") as dish_file:
        for line in dish_file:
            dishes.append( line.lower().replace("'","_").strip() )
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
            
    return {"names":names, "stars":stars}

def main():

    # get restaurant info (name and stars)
    rest_info = aboutRestaurants()
    rest_names = rest_info['names']
    rest_stars = rest_info['stars']

    review_text = {}    #review_id to text
    restaurant_text = {}    #business_id to text
    review_stars = [] #order of review to stars
    with open(review_path,"r") as review_file:
        for line in review_file:
            rev_content = json.loads( line.strip() )
            content_text = rev_content["text"].lower().replace("'","_").replace("\n", " ").strip()

            #we only want reviews about our taget
            if(content_text.find(target_dish) == -1):
                continue

            review_text[ rev_content["review_id"] ] = content_text

            business_id = rev_content["business_id"]
            if business_id in restaurant_text:
                restaurant_text[business_id] += " " + content_text
            else:
                restaurant_text[business_id] = content_text

            review_stars.append( rev_content["stars"] )

    print("total " + str(len(restaurant_text)) + " restaurants about our target")
    print("total " + str(len(review_text)) + " reviews about our taget")

if __name__=="__main__":
    main()


