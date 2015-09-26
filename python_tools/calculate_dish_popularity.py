import json
import math
import numpy as np

dish_info_path = "../data/dish_info.json"
f_map_path = "../data/f_popularity_map.json"
v_map_path = "../data/v_popularity_map.json"

def getDishInfo():
    all_info = []
    with open(dish_info_path, "r") as info_file:
        for line in info_file.readlines():
            dish_info = json.loads(line)
            all_info.append(dish_info)
            
    return all_info

def fMeasure(val1,val2):
    """
        special case of F-measure
    """
    return 2*val1*val2/(val1+val2)

def vectorLength(val1, val2):
    """
        treat input as a vector and measure it's length
    """
    return  math.sqrt(val1*val1 + val2*val2)

if __name__=="__main__":
    all_info = getDishInfo()

    f_popularity_map = {}   #map of dish name to f_popularity
    v_popularity_map = {}   #map of dish name to v_popularity

    with open("dish_pop_log.txt","w") as log_file:
        for i,dish_info in enumerate(all_info):
            name = dish_info['name']
            f_popularity = fMeasure( dish_info['review_stars'], dish_info['restaurant_stars'] )
            v_popularity = vectorLength( dish_info['review_stars'], dish_info['restaurant_stars'] )

            f_popularity_map[name] = f_popularity
            v_popularity_map[name] = v_popularity

            #print(name + " : f_popularity=" + str(f_popularity) + " , v_popularity=" + str(v_popularity) )
            log_file.write(name + " : f_popularity=" + str(f_popularity) + " , v_popularity=" + str(v_popularity) + "\n" )

    name_sorted_f = sorted( f_popularity_map, key=f_popularity_map.__getitem__ ,reverse=True )
    name_sorted_v = sorted( v_popularity_map, key=v_popularity_map.__getitem__ ,reverse=True )

    sorted_f_hash = []
    sorted_v_hash = []

    for name in name_sorted_f[:200]:
        sorted_f_hash.append({"name":name,"pop":f_popularity_map[name]})

    for name in name_sorted_v[:200]:
        sorted_v_hash.append({"name":name,"pop":v_popularity_map[name]})

    with open(f_map_path, "w") as f_map_file:
        f_map_file.write( json.dumps(sorted_f_hash) )

    with open(v_map_path, "w") as v_map_file:
        v_map_file.write( json.dumps(sorted_v_hash) )