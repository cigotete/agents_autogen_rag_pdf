import os
import random
import autogen
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory
from langchain_openai import OpenAI
from langchain.chains import ConversationalRetrievalChain

config_list_gpt = autogen.config_list_from_json(
  "OAI_CONFIG_LIST.json",
  filter_dict={
    "model": ["gpt-3.5-turbo-1106"]
  }
)

# Configuration that defines a function to ask pdf file.
llm_config_gpt = {
  "temperature": 0,
  "timeout": 300,
  "seed": random.randint(100, 100000),
  "config_list": config_list_gpt,
  "functions": [
        {
            "name": "answer_question",
            "description": "Answer any pdf related questions",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question to ask in relation to pdf content.",
                    }
                },
                "required": ["question"],
            },
        }
    ],
}

gpt_api_key = config_list_gpt[0]["api_key"]
os.environ['OPENAI_API_KEY'] = gpt_api_key


#Loading PDF file.
loaders = [ PyPDFLoader('./pdf_files/content.pdf') ]
docs = []
for l in loaders:
    docs.extend(l.load())
text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000)
docs = text_splitter.split_documents(docs)

vectorstore = Chroma(
    collection_name="full_documents",
    embedding_function=OpenAIEmbeddings()
)
vectorstore.add_documents(docs)


# Function that reads pdf file.
qa = ConversationalRetrievalChain.from_llm(
    OpenAI(temperature=0),
    vectorstore.as_retriever(),
    memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True)
)


# Function invoked to read files
def answer_question(question):
  response = qa({"question": question})
  return response["answer"]


def is_termination_msg(content) -> bool:
    have_content = content.get("content", None) is not None
    if have_content and "TERMINATE" in content["content"]:
        return True
    return False


#Agents
assistant = autogen.AssistantAgent(
    name="assistant",
    llm_config=llm_config_gpt,
)

user_proxy = autogen.UserProxyAgent(
    name="user_proxy",  
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    llm_config=llm_config_gpt,
    is_termination_msg=is_termination_msg,
    system_message="""Reply TERMINATE if the task has been solved at full satisfaction.
Otherwise, reply CONTINUE, or the reason why the task is not solved yet.""",
    function_map={"answer_question": answer_question}
)


# Initiate chat
user_proxy.initiate_chat(
    assistant,
    message="""
What are the subject of the content?
"""
)