
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from IPython.display import Markdown
from dotenv.ipython import load_dotenv
import os
from langchain.messages import HumanMessage
from langchain_core.tools import create_retriever_tool
from langchain_community.vectorstores import Chroma

load_dotenv()  # charge les variables d'environnement

embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
llm = ChatOpenAI(model="gpt-4o", temperature=0)

texts = [
         "Amine Salmi, 26 ans, diplômé d’un Master en Intelligence Artificielle et Data Science, "
        "est passionné par l’analyse de données, le développement de modèles prédictifs et "
        "l’optimisation de workflows intelligents. Curieux et rigoureux, il aime relever des "
        "défis techniques et collaborer sur des projets innovants. Sa maîtrise de Python, "
        "Pandas et des environnements Linux/Ubuntu lui permet de concevoir des solutions "
        "robustes et reproductibles. En dehors du cadre académique et professionnel, Amine "
        "consacre du temps à la lecture, explore de nouvelles cultures à travers ses voyages, "
        "écoute de la musique pour nourrir sa créativité et pratique la randonnée pour garder "
        "un équilibre entre bien-être et performance."
]
embedding_model =   OpenAIEmbeddings(model="text-embedding-ada-002")

vectorstore= Chroma.from_texts(texts,embedding_model, collection_name="cv_collection")
retrieval= vectorstore.as_retriever(kwargs={"k":5})
retrieval_tool = create_retriever_tool(
    retriever=retrieval,name="kb_search",description="Search information aboit me"
)
@tool
def send_email(email: str , subject:str ,content:str):
    """Send email to the givel email with the provided subject and content """
    print("=="*50)
    print("send_mail tool invoked")
    print("=="*50)
    return f"This email has been sent : destination :{email}, subject:{subject},content:{content}   "

@tool
def get_employee_info (name:str):
    """Get Information aboit employee(name,salary,seniority)"""
    print("=="*50)
    print("get_employee_info tool invoked")
    print("=="*50)
    return {"name:":name ,"salary": 12000,"seniority": 5}

graph = create_agent(
    model=llm,
    tools=[get_employee_info, retrieval_tool, send_email],
    system_prompt="answer the user question using prived tools",
)

resp = graph.invoke(
    input={"messages": [HumanMessage("Je veux connaitre le salaire de Yassine")]}
)
print(resp["messages"][-1].content)