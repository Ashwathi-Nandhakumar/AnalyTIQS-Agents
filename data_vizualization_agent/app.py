#import the necessay libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_groq import ChatGroq
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from dotenv import load_dotenv
import os
import glob

#load the environment variables
load_dotenv()

# a function that finds and displays all .csv files in the cwd
def available_csv_files():
    """list all the .csv files in the local/current working directory"""
    files = glob.glob(os.path.join(os.getcwd(),"*.csv"))
    if not files:
        return None
    return [os.path.basename(file) for file in files]

# asks the user to choose a file from the list
print(available_csv_files())
chosen_df = input("Enter the filename of the dataset you wish to analyze:")

#initialize llm
llm = ChatGroq(api_key=os.getenv("groq_api_key"),model="llama-3.3-70b-versatile",temperature=0)

# load dataset
df = pd.read_csv(chosen_df)

#create agent using the llm and df
agent = create_pandas_dataframe_agent(llm=llm,df=df, verbose=False,return_intermediate_steps=True)

#conversation loop
print("Ask questions about your dataset (type 'exit' to quit):")

while True:
    user_input = input("You: ")
    if user_input.strip().lower() in ["exit", "quit"]:
        print("See ya later!")
        break

    response = agent.invoke(user_input)
    print(response['intermediate_steps'][-1][0].tool_input.replace('; ', '\n'))
