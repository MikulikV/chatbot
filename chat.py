from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.chains import RetrievalQA,  ConversationalRetrievalChain, RetrievalQAWithSourcesChain
from langchain.docstore.document import Document
from langchain.schema import HumanMessage, AIMessage
from langchain.memory import ConversationSummaryBufferMemory
import tiktoken
import re
import os
import glob

from dotenv import load_dotenv 


def fix_newlines(text):
    return re.sub(r"(?<!\n)\n(?!\n)", " ", text)


def fix_tabs(text):
    return re.sub(r"(?<!\t)\t(?!\t)", " ", re.sub(r"\t{2,}", " \t ", text))


def remove_multiple_newlines(text):
    return re.sub(r"\n{2,}", " \n ", text)


def remove_multiple_spaces(text):
    return re.sub(r" +", " ", text)


def remove_time_codes(text):
    return re.sub(r"\d{2}:\d{2}:\d{2}", "", text)


def remove_stars_from_text(text):
    return text.replace("*", "")


def clean_text(data, cleaning_functions):
    prepared_data = []
    for document in data:
        for cleaning_function in cleaning_functions:
            document.page_content = cleaning_function(document.page_content)
        doc = Document(
            page_content=document.page_content,
            metadata=document.metadata
        )
        prepared_data.append(doc)
    
    return prepared_data



# Define data loader
def load_data(source_directory, k):
    # load documents
    loader = WebBaseLoader(source_directory)
    documents = loader.load()

    cleaning_functions = [
        fix_tabs,
        remove_time_codes,
        remove_stars_from_text,
        fix_newlines,
        remove_multiple_newlines,
        remove_multiple_spaces,
    ]

    documents = clean_text(documents, cleaning_functions)

    # Define length_function for text_splitter
    tokenizer = tiktoken.get_encoding("cl100k_base")

    def tiktoken_len(text):
        tokens = tokenizer.encode(
            text,
            disallowed_special=()
        )
        return len(tokens)

    # split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000/k, 
        chunk_overlap=200/k,
        length_function=tiktoken_len,
        separators=["\n\n", "\n", "(?<=\.)", "(?<=\!)", "(?<=\?)", "(?<=\,)", " ", ""],
        add_start_index = True,
    )
    docs = text_splitter.split_documents(documents)

    return docs


# Define vector store
def create_vector_store(docs):
    # define embedding
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    # create vector database from data
    vector_store = DocArrayInMemorySearch.from_documents(docs, embeddings)
    
    return vector_store


# Define retriever
def create_retriever(vector_store, search_type, k):
    retriever = vector_store.as_retriever(search_type=search_type, search_kwargs={"k": k})

    return retriever


# Define chain
def create_chain(retriever, temperature, chain_type):    
    # create a chatbot chain. Memory is managed externally.
    chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=temperature), 
        chain_type=chain_type, 
        retriever=retriever, 
        return_source_documents=True,
        memory=ConversationSummaryBufferMemory(llm=ChatOpenAI(model_name="gpt-3.5-turbo"), memory_key="chat_history", input_key='query', output_key="result", return_messages=True, max_token_limit=650)
    )
    return chain


if __name__ == "__main__":
    # Step 1 - Load OPENAI_API_KEY
    load_dotenv()

    source_urls = [
        "https://www2.cbn.com/lp/faith-homepage", 
        "https://www2.cbn.com/devotions/god-will-help-you-triumph-over-despair",
        "https://www2.cbn.com/faith/who-is-jesus",
        "https://www2.cbn.com/faith/new-christians",
        "https://www2.cbn.com/lp/faith-coming-back-your-faith",
        "https://www2.cbn.com/lp/faith-grow-deeper-your-faith",
        "https://www2.cbn.com/lp/faith-share-your-faith"
    ]

    chunks = load_data(source_urls, 4)
    vector_store = create_vector_store(chunks)
    retriever = create_retriever(vector_store, "similarity", 4)
    chain = create_chain(retriever, 0, "stuff")
    chat_history = []

    while True:
        print()
        question = input("Question: ")

        # Generate answer
        response = chain({"query": question})
        # chat_history.append(HumanMessage(content=question))
        # chat_history.append(AIMessage(content=response["result"]))

        # Retrieve answer
        # answer = response["answer"]
        # source = response["source_documents"]

        # # Display answer
        # print("\n\nSources:\n")
        # for document in source:
        #     print(f"Text chunk: {document.page_content}...\n")
        print(f"Answer: {response}")



