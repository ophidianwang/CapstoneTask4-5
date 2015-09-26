import json
import pickle

path2files="../../yelp_dataset_challenge_academic_dataset/"
path2buisness=path2files+"yelp_academic_dataset_business.json"
path2reviews=path2files+"yelp_academic_dataset_review.json"
cat2rid_path = "data_cat2rid.pickle"

chinese_business = "../data/chinese_business.json"
chinese_review = "../data/chinese_review.json"

def getRestaurant(target_business_ids):
    cursor = 0
    with open(chinese_business, "w") as dest_file, open(path2buisness,"r") as source_file:
        for line in source_file.readlines():
            business_json = json.loads(line)

            if(business_json['business_id'] in target_business_ids):
                dest_file.write(line)
                cursor +=1
    print(str(cursor) + " business are cloned." )

def getReview(target_business_ids):
    cursor = 0
    with open(chinese_review, "w") as dest_file, open(path2reviews,"r") as source_file:
        for line in source_file.readlines():
            review_json = json.loads(line)

            if(review_json['business_id'] in target_business_ids):
                dest_file.write(line)
                cursor +=1
    print(str(cursor) + " reviews are cloned." )
            
def getRids(category):
    pkl_file = open(cat2rid_path, 'rb')
    data = pickle.load(pkl_file)

    if( category in data ):
        return data[category]
    else:
        return []

if __name__=="__main__":

    target_business_ids = getRids("Chinese")
    print("There are " + str( len(target_business_ids) ) + " business targets")

    getRestaurant(target_business_ids)
    getReview(target_business_ids)
