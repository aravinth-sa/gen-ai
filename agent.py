import logging
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, AgentType

logging.basicConfig(level=logging.INFO)

custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
        You are a helpful e-commerce assistant for a building materials and home furnishing store.
        Use the product information below to answer the customer's question.

        Product Info:
        {context}

        Customer Question:
        {question}

        Helpful Answer (based only on the product info above):
        """
)

def get_qa_chain(retriever):
    logging.info("Initializing QA chain with retriever: %s", type(retriever).__name__)
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": custom_prompt}
    )
    logging.info("QA chain initialized.")
    return chain

def get_agent(tools, memory):
    logging.info("Initializing agent with tools: %s", [tool.name for tool in tools])
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True
    )
    logging.info("Agent initialized.")
    return agent
