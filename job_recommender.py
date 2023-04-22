import streamlit as st
import scipy
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import bz2

# Add title to the app
st.title("Job Recommender")

# Define user inputs
job_title = st.text_input("Enter your desired job title", "Data Scientist")
resume = st.text_input("Enter your summary", "Experienced data scientist with a PhD in statistics")
user_location = st.selectbox("Select your location", options=["New York", "San Francisco", "Seattle"])

# Load the vectorizer objects from the pickle files
with open("Data/job_desc_tfidf_vectorizer.pkl", "rb") as f:
    job_desc_tfidf = pickle.load(f)

with open("Data/job_title_tfidf_vectorizer.pkl", "rb") as f:
    job_title_tfidf = pickle.load(f)

# Transform the user input using the vectorizer models
job_title_tfidf_user = job_title_tfidf.transform([job_title])
job_desc_tfidf_user = job_desc_tfidf.transform([resume])


# open the compressed bz2 file for reading
with bz2.BZ2File('Data/one_per_new_df.bz2', 'r') as f:
    # read the DataFrame from the compressed file
    new_df = pd.read_csv(f, compression='bz2')
# # Load the new_df dataframe from the pickle file
# with open("Data/new_df.pkl", "rb") as f:
#     new_df = pickle.load(f)

    

new_df=new_df.rename(columns = {'json.schemaOrg.title':'Title','text':'Job Description','json.schemaOrg.jobLocation.address.addressLocality':'Location'})
job_titles = new_df["Title"].values
job_descriptions = new_df["Job Description"].values

job_desc_tfidf_features = job_desc_tfidf.transform(job_descriptions)
job_title_tfidf_features = job_title_tfidf.transform(job_titles)

# Compute the cosine similarity between the user input and the job features
job_desc_cosine_similarities = cosine_similarity(job_desc_tfidf_user, job_desc_tfidf_features)
job_title_cosine_similarities = cosine_similarity(job_title_tfidf_user, job_title_tfidf_features)

# Compute the final similarity scores
similarity_scores = 0.4 * job_desc_cosine_similarities + 0.4 * job_title_cosine_similarities

# Add location as a high weight feature ordered by close distance, location weight is 0.2
locations = new_df['Location'].values

location_weight = np.zeros(len(locations))
location_weight[locations == user_location] = 0.2

similarity_scores += location_weight.reshape(1, -1)

# Sort the jobs by similarity score and return the top 5 recommendations
top_recommendations = new_df[['Title', 'Location', 'Job Description']].copy()
top_recommendations['Similarity Score'] = similarity_scores.reshape(-1)
top_recommendations = top_recommendations.sort_values('Similarity Score', ascending=False).head(5)

# Display top 5 job recommendations
st.write("Top 5 job recommendations:")
st.write(top_recommendations[['Title', 'Location','Job Description','Similarity Score']])

