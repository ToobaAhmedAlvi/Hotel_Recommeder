title =("<center>"
        "<p>""Welcome to Hotel Recommendation System!""</p>"
        "</center>")

head = (
  "<center>"
  "<img src='https://img.freepik.com/free-vector/hotel-tower-concept-illustration_114360-12962.jpg?w=740&t=st=1710571774~exp=1710572374~hmac=6daf26dbfb918ba737df6d2f091351ab0348437afeff121f973efd2d55bfe092' width=400>"
  "The robot was trained to search for relevant hotels from the dataset provided."
  "</center>"
)

#importing libraries
import requests
import os
import gradio as gr
import pandas as pd
import pprint
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from openai.embeddings_utils import get_embedding, cosine_similarity
df = pd.read_pickle('data.pkl')
embedder = SentenceTransformer('all-mpnet-base-v2')
def search(query,pprint=True):
    n = 15
    query_embedding = embedder.encode(query,show_progress_bar=True) #encode the query
    df["rev_sim_score"] = df.embed_1.apply(lambda x: cosine_similarity(x, query_embedding.reshape(768,-1))) #similarity against each doc
    review_results = (
        df.sort_values("rev_sim_score", ascending=False) # re-rank
        .head(n))
    resultlist = []
    hlist = []
    for r in review_results.index:
      if review_results.hotel_name[r] not in hlist:
          smalldf = review_results.loc[review_results.hotel_name == review_results.hotel_name[r]]
          smallarr = smalldf.rev_sim_score[r].max()
          sm =smalldf.rate[r].mean()
          if smalldf.shape[1] > 3:
            smalldf = smalldf[:3]
          resultlist.append(
          {
            "hotel_name":review_results.hotel_name[r],
            "description":review_results.hotel_description[r],
            "relevance score": smallarr.tolist(),
            "rating": sm.tolist(),
            "relevant_reviews": [ smalldf.hotel_info[s] for s in smalldf.index]
          })
          hlist.append(review_results.hotel_name[r])
          return resultlist

def hotel_info(query, pprint=True):
      query_embedding = embedder.encode(query,show_progress_bar=True) #encode the query
      df["hotel_sim_score"] = df.embed_2.apply(lambda x: cosine_similarity(x, query_embedding.reshape(768,-1)))
      #similarity against each doc
      n=3
      hotel_results = (
        df.sort_values("hotel_sim_score", ascending=False) # re-rank
        .head(n))
      resultlist = []
      hlist = []
      for r in hotel_results.index:
        if hotel_results.hotel_name[r] not in hlist:
          smalldf = hotel_results.loc[hotel_results.hotel_name == hotel_results.hotel_name[r]]
          smallarr = smalldf.hotel_sim_score[r].max()
          sm =smalldf.rate[r].mean()
          if smalldf.shape[1] > 3:
            smalldf = smalldf[:3]
            resultlist.append(
          {
            "name":hotel_results.hotel_name[r],
            "description":hotel_results.hotel_description[r],
            "hotel_picture":hotel_results.hotel_image[r],
            "relevance score": smallarr.tolist(),

          })

      return resultlist
def search_ares(query):
   x_api_key=os.getenv("x_api_key")
   url = "https://api-ares.traversaal.ai/live/predict"
   payload = {"query": [query]}
   headers = {
  "x-api-key": x_api_key,
  "content-type": "application/json"}  
   response = requests.post(url, json=payload, headers=headers)
   content = response.json()
   return content


def greet(name):
    print("Hi! I am your AI assistant.Please let me know your name please.. ")
    return "Hi  " + name + "!"

    #hotel_details = hotel_info(query)
    #hotel_reviews = search(query)
    #return hotel_details,hotel_reviews

blocks = gr.Blocks()
with blocks as demo:
   
  greet = gr.Interface(fn=greet, inputs="textbox",title=title, description=head, outputs="textbox")
  hotel_info= gr.Interface(fn=hotel_info, inputs="text",outputs=[gr.components.Textbox(lines=3, label="Write query to search about hotel info")])
  search = gr.Interface(fn=search, inputs="text", outputs=[gr.components.Textbox(lines=3, label="Write query to search about hotel reviews")])
  search_ares= gr.Interface(fn=search_ares, inputs="textbox", outputs=[gr.components.Textbox(lines=3, label="Write query to search using Ares API")])
  


demo.launch(share=True,debug=True)