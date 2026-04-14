
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain.messages import HumanMessage, AIMessage
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
import qdrant_client
from qdrant_client.http.models import Distance, VectorParams, Filter, FieldCondition, MatchValue
from qdrant_client.http.exceptions import UnexpectedResponse
from uuid import uuid4
from langchain_community.embeddings import FastEmbedEmbeddings
from datetime import datetime
from loguru import logger

"""
If you have error with proxy 

#!/bin/bash
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY

export no_proxy="localhost,127.0.0.1,::1"

"""


TEST_USER_ID = "12-12-12"

def create_model_chat(base_url:str , api_key:str, model_name:str, temperature:float, max_tokens:int = 16384) -> ChatOpenAI:
    """Create obj ChatOpenAI from langchain"""
    return ChatOpenAI(
                base_url= base_url,
                api_key=api_key, 
                model=model_name,     
                temperature=temperature,
                max_tokens=max_tokens
            )

class Qdrant_client:
    def __init__(self,url:str = "http://localhost:6333", 
                 api_key:str = "TESTKEY", # your key here! 
                 embeddings_name = "BAAI/bge-small-en-v1.5", 
                 vector_size:int = 384):
        
        self.vector_size = vector_size
        self.qdrant_client = self.__create_client(url, api_key)
        self.embeddings = FastEmbedEmbeddings(model_name=embeddings_name)
        self.create_collection()
        
        if self.qdrant_client:
            self.vector_store_qdrant = QdrantVectorStore(
                client=self.qdrant_client,
                collection_name="rag_database",  
                embedding=self.embeddings,
                distance=Distance.EUCLID
                )
        else: exit(0)
        
    def __create_client(self,url:str = "http://localhost:6333", api_key:str = "TESTKEY") -> QdrantClient:
        try:
            qdrant_client = QdrantClient(
                    url     = url,
                    api_key = api_key,
                )
            return qdrant_client 
        except Exception as e:
            logger.error(f"Error while create qdrant_client: {e}")
            return None
        
    def get_client(self) -> QdrantClient | None:
        """
        Get object qdrant db
        """
        if self.qdrant_client is None:
            logger.warning(f"qdrant client is None! Checkup connection")
        return self.qdrant_client
    
    def create_collection(self, name:str = 'rag_database') -> bool:
        """
        Create collection with name .
        """
        try:
            if self.qdrant_client:
                if not self.qdrant_client.collection_exists(name):
                    self.qdrant_client.create_collection(
                    collection_name=name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.EUCLID,      
                    ),
                    )
                    logger.info(f"Коллекция создана {name}")

                    return True
            else:
                logger.warning(f"Can't create rag collection, qdrant client is None! Checkup connection")
                return False
        except UnexpectedResponse as e:
            logger.error(f"Ошибка при создании коллекции: {e}")
            return False
            
    def add_database(self, 
                     message:str,
                     user_id:str, 
                     role:str) -> bool:
        """ add message or answer in db"""
        doc_id = str(uuid4())
        now = datetime.now()

        self.vector_store_qdrant.add_texts(
            texts=[message],
            metadatas=[{
                "user_id": user_id,         
                "source": role,
                "role": role,
                "timestamp": now.isoformat(),
            }],
            ids=[doc_id]           
        )
    

    def get_from_database(self,user_id:str, num:int = 5) -> str | None:
        """
        Get from database

        args:
        -----
        num : int - get num of similary message
        """
        message = ''
        try:
            if num >= 1:
                qdrant_filter = Filter(
                    must=[
                        FieldCondition(
                            key="metadata.user_id",                  
                            match=MatchValue(value=user_id)
                        )
                    ]
                )
                relevant_docs = self.vector_store_qdrant.similarity_search(
                    query=message,
                    k=num,
                    filter=qdrant_filter
                )
                message = "\n\n".join(doc.page_content for doc in relevant_docs)

                return message
            else:
                logger.warning('in function get_from_database num must be num >= 1')

                return message
        except Exception as e:
            logger.error(f"Error while getting chunks from vector database: {e}")
            return message



model = create_model_chat(f"http://localhost:8282/v1",'',"model_name",0.7)


qdrant_client_ = Qdrant_client()

model_qdrant_client = qdrant_client_.get_client()



logger.success("Start testing ...")




test_question_1 = HumanMessage(content="Hello, my name Jesus Versachi, i working in IT")
test_question_2 = HumanMessage(content="Let's talk about live , how are you doning?")
test_question_3 = HumanMessage(content="Did tou know what i play soccer like pro ? ")


agent = create_agent(model=model)


# ___it 1___

qdrant_client_.add_database(test_question_1.content, TEST_USER_ID, "human message")

response1 = agent.invoke(
    {"messages": [test_question_1]}
)


qdrant_client_.add_database(response1['messages'][1].content, TEST_USER_ID, "AI message")

# ___it 2___

qdrant_client_.add_database(test_question_2.content, TEST_USER_ID, "human message")

response2 = agent.invoke(
    {"messages": [test_question_2]}
)

qdrant_client_.add_database(response2['messages'][1].content, TEST_USER_ID, "AI message")

# ___it 3___


qdrant_client_.add_database(test_question_3.content, TEST_USER_ID, "human message")

response3 = agent.invoke(
    {"messages": [test_question_3]}
)

qdrant_client_.add_database(response3['messages'][1].content, TEST_USER_ID, "AI message")

# ___it 4___



history = qdrant_client_.get_from_database(TEST_USER_ID)

test_question_context = "What my name ? And Where i working?" + "\n\n" + history

test_question_4 = HumanMessage(content=test_question_context)

qdrant_client_.add_database(test_question_4.content, TEST_USER_ID, "human message")

response4 = agent.invoke(
    {"messages": [test_question_4]}
)

qdrant_client_.add_database(response4['messages'][1].content, TEST_USER_ID, "AI message")

logger.debug(response4['messages'][1].content)