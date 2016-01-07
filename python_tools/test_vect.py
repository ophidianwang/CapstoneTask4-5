from time import time
from sklearn.feature_extraction.text import CountVectorizer

ori_dishes = ["General Tso's Chicken",
              "chicken",
              "Tso's Chicken",
              "choco",
              "choco cake"
              ]

dishes = {}
for i, name in enumerate(ori_dishes):
    trimed_name = name.lower().replace("\n", " ").replace("'", "_").strip()
    dishes[trimed_name] = i
print(dishes)

ori_text = ["I love choco",
            "I ate choco cake.",
            "China King, you are my new man. To those fools who have given them anything but 5 stars, I say you must have ordered off of the gringo menu.   I lived in Chandler, where Olive Garden, Outback and Chili's are considered \"good eatin\". Sacrilegious is what I call it. My dining mate was wary of Chinese, the way most people are, because you are never really sure of what you're getting. I am so happy I tried this place. I went back three times in one week for more. I LOVE IT!!  The first time I went in for dinner, I was taken aback by all the Chinese people eating there. Think back to every Chinese place you've been. Have you ever seen any kind of Asian person in there (other than the ones working)? Chances are, the answer is no. This place was PACKED. We were handed menus with about 8 items on it. It's your typical General Tso's Chicken, Sesame Chicken, etc...I looked around at what everyone else was eating and it did not look like \"typical\" Chinese food. So I said to the waiter- \"what are they eating?\" He said something back that I didn't understand, then he disappeared. My mate looked nervous and asked what I just got us into. Who cares! It can't be that bad!!  When the waiter returned, he had a menu that was about 5 pages long with pictures of delicious looking treats. We told the waiter that we liked duck, fish, noodles (and a bunch of other random things) and to just bring whatever was best. We got a half duck as an appetizer that was melt-in-your-mouth good. Crispy skin and tender meat, exactly how it should be. For entree, we got a whole fish and some crispy noodles. We even got some Chinese beer. Dessert was sketchy, but then again, it was free so who can complain??   The other times I went, I had the BBQ pork, the sizzling chicken plate, and some other delicious items I can't recall or pronounce. I also went back for lunch and had the Dim Sum. I almost think I should write a separate review for the lunch because it was that good. I mean, really people just coming around with the BEST food ever asking if I want to try it. You're damn right I do. And don't stray too far, because I'll want seconds.  All in all, the most amazing Chinese food in a casual setting with very reasonable prices. The staff is super friendly and efficient. Also, there was some Chinese type \"Idol\" on TV and I felt very authentic being there, like I would step outside and be on a busy street with open air markets and people serving fried grasshopper. Instead, I walked outside and saw the glaring neon lights of the adjacent Boston Market. Oh Chandler. Does Boston Market even know that it is in the presence of greatness? Bow to the King.  On a last note...what is up with these people who write glowing reviews and only give 4 stars?? What do you want them to do, wipe your ass, too? Give China King the comeuppance it deserves!!!!"
            ]

text = []
for single in ori_text:
    trimed_single = single.lower().replace("\n", " ").replace("'", "_").strip()
    text.append(trimed_single)
print(text)

vectorizer = CountVectorizer(vocabulary=dishes, ngram_range=(1, 5), lowercase=True)

t0 = time()
X = vectorizer.fit_transform(text)
print("done in %fs" % (time() - t0))
print("n_samples: %d, n_features: %d" % X.shape)

voc = vectorizer.vocabulary_
print(voc)

for l, text_vec in enumerate(X.toarray()):
    print(text_vec)
