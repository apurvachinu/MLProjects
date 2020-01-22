#Assignment 1 and 2 combined. Assignment 2 part on line no 18
#Since there was no data provided, i built the model on IMDB dataset. For your dataset, just change the feature list at line 14.

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

#Step 1: Read CSV File
df = pd.read_csv("movie_dataset.csv")

#Step 2: Select Features

features = ['keywords','cast','genres']


#Step 3: Create a column in DF which combines all selected features
#ASSIGNMENT 2: Fill blank for poorly labeled video dataset, and use a try except block
for feature in features:
	df[feature] = df[feature].fillna('')

def combine_features(row):
	try:
		return row['keywords'] +" "+row['cast']+" "+row["genres"]
	except:
		print ("Error:", row)	

df["combined_features"] = df.apply(combine_features,axis=1)

#Step 4: Create word count matrix from this new combined column
cv = CountVectorizer()

count_matrix = cv.fit_transform(df["combined_features"])

#Step 5: Compute the Cosine Similarity
cosine_sim = cosine_similarity(count_matrix) 

#Let a user watches a video called "Avatar"
video_user_watched = "Avatar"



# Step 6: Get index of this movie from its title
def get_title_from_index(index):
	return df[df.index == index]["title"].values[0]

def get_index_from_title(title):
	return df[df.title == title]["index"].values[0]

video_index = get_index_from_title(video_user_watched)

similar_videos =  list(enumerate(cosine_sim[video_index]))

# Step 7: Get a list of similar videos in descending order of similarity score
sorted_similar_videos = sorted(similar_videos,key=lambda x:x[1],reverse=True)

# Step 8: recommending first 10 videos based on similarity score
i=0
for vid in sorted_similar_videos:
		print (get_title_from_index(vid[0]))
		i=i+1
		if i>10:
			break