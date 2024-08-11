# Databricks notebook source
# MAGIC %pip install -U --quiet databricks-sdk langchain-core langchain-community==0.2.6 youtube_search Wikipedia grandalf
# MAGIC
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from langchain_core.output_parsers import StrOutputParser


# COMMAND ----------

from langchain.prompts import PromptTemplate

prompt_template = PromptTemplate.from_template('''Tell me the name of latest {genre} movie which {actor} is one of the actors. Output the movie name in the form of a json e.g. {{"movie_name": "movie name"}}. Output only the json without any additional information or side notes.''')
prompt_template.format(genre="drama", actor="Brad Pitt")

# COMMAND ----------

from langchain_community.chat_models import ChatDatabricks
import json

def invoke_llm(prompt):
  llm = ChatDatabricks(endpoint="databricks-meta-llama-3-1-405b-instruct", max_tokens = 500)
  ai_message = llm.invoke(prompt)
  return str(ai_message.content)

def extract_movie (text:str) -> str:
    return json.loads(text)['movie_name']

# test LLM
invoke_llm(prompt_template.format(genre="Action", actor="Al Pacino"))


# COMMAND ----------

movie = extract_movie(invoke_llm(prompt_template.format(genre="Action", actor="Al Pacino")))
movie

# COMMAND ----------

# DBTITLE 1,Retriever
from langchain_community.retrievers import WikipediaRetriever

def retrieve_movie_info(movie):
  retriever = WikipediaRetriever()
  return retriever.invoke(input=f"hollywood english movie: {movie}")

# test
docs = retrieve_movie_info("House of Gucci")
print(docs[0].page_content[:100])

# COMMAND ----------

from langchain_community.tools import YouTubeSearchTool

def get_youtube_trailer(movie):
  tool = YouTubeSearchTool()
  return tool.run(f"{movie} movie trailer").split("'")[1]


#test 
get_youtube_trailer("The Shawshank Redemption")

# COMMAND ----------

recommend_movie_chain =  prompt_template | invoke_llm | StrOutputParser() | extract_movie 

# test chain 1
recommend_movie_chain.invoke(input={"genre":"action", "actor":"al pacino"})

# COMMAND ----------

wiki_template = PromptTemplate.from_template(
    """ 
    Summarize the below movie description in a few sentences, only output the summary nothing else:: 
    
    {wiki_dump}
    
    """
)


# COMMAND ----------

final_response = PromptTemplate.from_template(
  """ 
    Combine the results from movie name, movie plot description and youtube trailer link below into  a well written movie recommendation::
    {movie}
    {movie_plot}
    {youtube_trailer}
    """
)

# COMMAND ----------

movie_plot_chain = retrieve_movie_info | wiki_template | invoke_llm | StrOutputParser()

# COMMAND ----------

youtube_trailer_chain = get_youtube_trailer

# COMMAND ----------

chain = (
          recommend_movie_chain | StrOutputParser()
          | {"movie_plot": movie_plot_chain , "youtube_trailer": youtube_trailer_chain} 
          | final_response 
          | invoke_llm       
        )


# COMMAND ----------

print(chain.invoke(input={"genre":"action", "actor":"al pacino"}))

# COMMAND ----------

chain.get_graph().print_ascii()

# COMMAND ----------
