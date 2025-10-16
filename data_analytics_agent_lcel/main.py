#import all the libraries 
import os
import glob
import pandas as pd
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain.agents import Tool,initialize_agent,AgentType
from langchain_core.prompts import ChatPromptTemplate

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error

#tool to list all the csv files that are present in the folder
@tool
def available_csv_files()->Optional[List[str]]:
    """list all the .csv files in the local/current working directory"""
    files = glob.glob(os.path.join(os.getcwd(),"*.csv"))
    if not files:
        return None
    return [os.path.basename(file) for file in files]

#tool to load/cache datasets
df_cache={}
@tool
def dataset_loader(paths:Optional[List[str]] | str)->str:
    """preload/cache the datasets"""
    if isinstance(paths, str):
        paths = [paths]
    loaded,cached =[],[]
    for path in paths:
        path = path.strip("'\"")
        if path not in df_cache:
            df_cache[path] = pd.read_csv(path)
            loaded.append(path)
        else:
            cached.append(path)
    return f"Loaded: {loaded} \n Cached: {cached}"

#to get descriptions and info on the dataset 
@tool
def dataset_summary_info(dataset_paths:Optional[List[str]] | str)->List[Dict[str,Any]]:
    """Returns the datasets basic information and descriptions"""
    if isinstance(dataset_paths, str):
        dataset_paths = [dataset_paths]
    summaries=[]
    for d_path in dataset_paths:
        d_path = d_path.strip("'\"")
        if d_path not in df_cache:
            df_cache[d_path] = pd.read_csv(d_path)
        df = df_cache[d_path]
        summaries.append({
            "filename":d_path,
            "col_names":list(df.columns),
            "data_types":dict(df.dtypes.astype(str)),
            "df_shape": list(df.shape),
            "value_counts":{col: df[col].value_counts().to_dict() for col in df.columns}
        })

    return summaries

# tool to run any datafream methods mentioned by the user
@tool
def use_df_methods(file_name:str,method:str)->str:
    """Ask to execute classic dataframe methods on a chosen dataset"""
    if file_name not in df_cache:
        try:
            file_name = file_name.strip("\"'")
            df_cache[file_name]= pd.read_csv(file_name)
        except FileNotFoundError:
            return "File not Found"
    
    df = df_cache[file_name]
    func = getattr(df,method,None)
    if not callable(func):
        return f"'{method}' is not a valid DataFrame method."
    try:
        result = func()
        return str(result)
    except Exception as e:
        return f"Error calling {method}: {e}"

# tool to create and evaluate a classification model
@tool
def classification_dataset_tool(file_name:str,target:str)->Dict[str,float]:
    """A tool that splits data and trains and evaluates a model for classification datasets/purposes"""
    if file_name not in df_cache:
        try: 
            file_name = file_name.strip("\"'")
            df_cache[file_name] = pd.read_csv(file_name)
        except FileNotFoundError:
            return "File not Found"
    df = df_cache[file_name]

    X = df.drop(target,axis=1)
    y = df[target]
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    return {"accuracy":accuracy_score(y_test,y_pred)}

# tool to create and evaluate a regression model
@tool
def regression_dataset_tool(file_name:str,target:str)->Dict[str,float]:
    """ Splits data and trains/evaluates regression models for regression datasets"""
    if file_name not in df_cache:
        try:
            file_name = file_name.strip("\"'")
            df_cache[file_name] = pd.read_csv(file_name)
        except FileNotFoundError:
            return "File not Found"
    df = df_cache[file_name]
    X = df.drop(target,axis=1)
    y = df[target]
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    model = RandomForestRegressor()
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    return {"R2_score":r2_score(y_pred,y_test),"MSE":mean_squared_error(y_pred,y_test)}

#tools list to make invoking easier
tools = [available_csv_files,dataset_loader,dataset_summary_info,use_df_methods,regression_dataset_tool,classification_dataset_tool]


prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "You are DataWizard — an AI data scientist. "
     "Use the tools provided to analyze CSV files, determine data types, "
     "and decide whether each dataset suits classification or regression tasks."
     "Always pass the exact filename of the CSV when using a tool, like 'IRIS.csv'."
     "When performing classification, always ask the user to specify the target column."
     "Never assume it."
     "Dont call unecessary tools, just use the one you need."
     "Only use the model tools if the user asks you to create a model"),
    ("user", "{input}"),

])

# wrapping tools in LC objects
lc_tools = [
    Tool(name="Available CSV Files", func=available_csv_files, description="List all CSV files in the current directory."),
    Tool(name="Dataset Loader", func=dataset_loader, description="Load and cache CSV datasets."),
    Tool(name="Dataset Summary", func=dataset_summary_info, description="Get dataset columns, types, shape, and value counts."),
    Tool(name="DataFrame Methods", func=use_df_methods, description="Execute basic pandas DataFrame methods like head(), describe(), etc."),
    Tool(name="Classification Model", func=classification_dataset_tool, description="Train and evaluate a RandomForest classifier on a dataset."),
    Tool(name="Regression Model", func=regression_dataset_tool, description="Train and evaluate a RandomForest regressor on a dataset."),
]

#load environment variables from .env
load_dotenv()

#initializw the llm
llm = ChatGroq(api_key = os.getenv("groq_api_key"),model="llama-3.3-70b-versatile", temperature=0)

# Create the agent
agent_executor = initialize_agent(
    tools=lc_tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)
#conversation loop
print("Ask questions about your dataset (type 'exit' to quit):")

while True:
    user_input = input("You: ")
    if user_input.strip().lower() in ["exit", "quit"]:
        print("See ya later!")
        break

    result = agent_executor.invoke(user_input)
    print(f"DataWiz: {result}")












