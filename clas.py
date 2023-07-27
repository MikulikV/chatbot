from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyMuPDFLoader, DirectoryLoader, WebBaseLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.docstore.document import Document
import tiktoken
import re

import panel as pn
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import param

from dotenv import load_dotenv 

# OPENAI_API_KEY
load_dotenv()

# WIDGETS

# Sidebar
select_temperature = pn.widgets.FloatSlider(
    name="Temperature", 
    start=0.0, 
    end=1.0, 
    step=0.1, 
    value=0.0,
    styles={"font-size": "16px", "margin-bottom": "10px"}
)
select_chain_type = pn.widgets.RadioButtonGroup(
    name="Chain type", 
    options=["stuff", "map_reduce", "refine", "map_rerank"],
    button_type="primary",
    button_style="outline",
    styles={"margin-bottom": "20px"}
)
select_search_type = pn.widgets.RadioButtonGroup(
    name="Search type", 
    options=["similarity", "mmr", "score_threshold"],
    button_type="primary",
    button_style="outline",
    styles={"margin-bottom": "20px"}
)
select_top_k = pn.widgets.FloatSlider(
    name="Number of relevant chunks",
    start=1, 
    end=10, 
    step=1, 
    value=4, 
    styles={"font-size": "16px", "margin-bottom": "30px"}
)
save_button = pn.widgets.Button(
    name="Save and start", 
    button_type="success", 
    width=150,
    height=35, 
    styles={"margin": "0 auto"}
)

def switch_top_k(event):
    if event.new == "score_threshold":
        select_top_k.name = "Min score threshold"
        select_top_k.start = 0.1
        select_top_k.end = 0.99
        select_top_k.step = 0.1
        select_top_k.value = 0.7
    else:
        select_top_k.name="Number of relevant chunks"
        select_top_k.start=1
        select_top_k.end=10 
        select_top_k.step=1 
        select_top_k.value=4 

select_search_type.param.watch(switch_top_k, "value")

# Main layout
menu = pn.widgets.RadioButtonGroup(
    name="Menu", 
    options=["Conversation", "Database", "Splitter", "Memory"],
    button_type="primary",
    button_style="outline",
    disabled=True,
    styles={"margin-bottom": "10px"}
)
ui = pn.Column(
    pn.Column(
        pn.pane.HTML("<h1>To start conversation set up your settings</h1>", width=820, styles={"text-align": "center"}),
        pn.pane.Image("assets/gizmo.png", width=200, height=200, styles={"margin": "0 auto"}),
    )
)
# Conversation window
question = pn.widgets.TextInput(value="", placeholder="Send a message", width=720, height=40, disabled=True)
send_button = pn.widgets.Button(name="Send", width=80, height=40, disabled=True)
chat = pn.Column(pn.pane.HTML(), pn.Row(question, send_button))
# Database window
database = pn.Row(pn.pane.HTML())
# Splitter window
splitter = pn.Row(pn.pane.HTML())
# Chat history window
chat_history = pn.Row(pn.pane.HTML())

# CLASS CHATBOT

# Cleaning functions

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


cleaning_functions = [
    fix_tabs,
    remove_time_codes,
    remove_stars_from_text,
    fix_newlines,
    remove_multiple_newlines,
    remove_multiple_spaces,
]

# Define LLM
def llm(temperature):
    return ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=temperature,
    )


# Define data loader
def load_data():
    documents = []
    # load SuperBook documents
    txt_loader = DirectoryLoader("docs", glob="**/*.txt", loader_cls=TextLoader)
    documents.extend(txt_loader.load())
    docx_loader = DirectoryLoader("docs", glob="**/*.docx", loader_cls=Docx2txtLoader)
    documents.extend(docx_loader.load())
    # load CBN Faith section
    webloader = WebBaseLoader([
        "https://www2.cbn.com/lp/faith-homepage", 
        "https://www2.cbn.com/faith/devotionals",
        "https://www2.cbn.com/devotions/god-will-help-you-triumph-over-despair",
        "https://www2.cbn.com/faith/who-is-jesus",
        "https://www2.cbn.com/faith/new-christians",
        "https://www2.cbn.com/lp/faith-coming-back-your-faith",
        "https://www2.cbn.com/lp/faith-grow-deeper-your-faith",
        "https://www2.cbn.com/lp/faith-share-your-faith",
        "https://www2.cbn.com/devotions/trust-god",
        "https://www2.cbn.com/article/bible-says/bible-verses-about-prayer-praying",
        "https://www2.cbn.com/resources/ebook/perfect-timing-discover-key-answered-prayer",
        "https://www2.cbn.com/article/purpose/seven-keys-hearing-gods-voice",
        # FAQ SuperBook
        "https://cbn.com/superbook/faq-episodes.aspx", 
        "https://us-en.superbook.cbn.com/faq"
        "https://us-en.superbook.cbn.com/congratulations",
        "https://appscdn.superbook.cbn.com/api/bible/app_qanda.json/?lang=en&f=all&id=1&vid=13653741",
        "https://appscdn.superbook.cbn.com/api/bible/app_profiles.json/?lang=en&f=all&id=0&sort=null&r=100000&vid=13653741"
        "https://appscdn.superbook.cbn.com/api/bible/app_games.json/?lang=en&f=trivia&id=0&sort=null&r=100000&vid=13653741&result_version=2",
        "https://appscdn.superbook.cbn.com/api/bible/app_gospel/?lang=en&vid=13653741",
        "https://appscdn.superbook.cbn.com/api/bible/app_multimedia.json/?lang=en&f=all&id=0&sort=null&r=100000&vid=13653741"
    ])
    documents.extend(webloader.load())
    docs = clean_text(documents, cleaning_functions)

    return docs


# Define length_function for text_splitter
tokenizer = tiktoken.get_encoding("cl100k_base")

def tiktoken_len(text):
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)


# Define documents splitter
def split_documents(documents, top_k, search_type):
    k = 4 if search_type == "score_threshold" else top_k
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000/k, 
        chunk_overlap=100/k,
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
    if search_type == "score_threshold":
        retriever = vector_store.as_retriever(search_type="similarityatscore_threshold", search_kwargs={"score_threshold": k})
    else:
        retriever = vector_store.as_retriever(search_type=search_type, search_kwargs={"k": k})

    return retriever


# Define propmpt
prompt = PromptTemplate(
    input_variables=["history", "context", "question"],
    template="""
You are personal assistant named Gizmo like a character from SuperBook who is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics.
You are able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions.
Always answer as helpfully as possible in the manner of a deep believer only, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
Use the following context (delimited by <ctx></ctx>) and the chat history (delimited by <hs></hs>) to answer the question:
------
<ctx>
{context}
</ctx>
------
<hs>
{history}
</hs>
------
{question}
If it's not enough to answer the question use your own memory. Answer in question's language.
Answer:
""",
)

# Define memory
memory = ConversationBufferWindowMemory( 
    memory_key="history", 
    input_key='question',  
    return_messages=True,
    k=6
)


# Define chain
def create_chain(llm, retriever, chain_type):    
    # create a chatbot chain. Memory is managed externally.
    chain = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type=chain_type, 
        retriever=retriever, 
        return_source_documents=True,
        chain_type_kwargs={"memory": memory, "prompt": prompt},
    )
    return chain


# Define CBN class
class Chatbot(param.Parameterized):
    answer = param.String("")
    panels = param.List([])
    db_query = param.String("")
    db_response = param.List([])
    chat_history = param.List([])
    
    def __init__(self, t, c, s, k, **params):
        super(Chatbot, self).__init__(**params)
        self.temperature = t
        self.chain_type = c
        self.search_type = s
        self.top_k = k
        self.llm = llm(self.temperature)
        self.data = load_data()
        self.chunks = split_documents(self.data, self.top_k, self.search_type)
        self.vector_store = create_vector_store(self.chunks)
        self.retriever = create_retriever(self.vector_store, self.search_type, self.top_k)
        self.qa = create_chain(self.llm, self.retriever, self.chain_type)

    def conversation(self, _):
        query = question.value
        if query:
            response = self.qa({"query": query})
            self.chat_history = self.qa.combine_documents_chain.memory.chat_memory.messages
            self.db_query = response["query"]
            self.db_response = response["source_documents"]
            self.answer = response['result'] 
            self.panels.extend([
                {"You": query},
                {"Gizmo": self.answer},
            ])
            question.value = ""

        return pn.widgets.ChatBox(
            value=self.panels,
            message_icons={
                "You": "assets/user.png",
                "Gizmo": "assets/gizmo.png",
            },
            show_names=False,
            allow_input=False,
            ascending=True,
            height = 280
        )
    
    @param.depends('db_query')
    def get_last_question(self):
        if not self.db_query:
            return pn.Column(
                pn.pane.HTML("<h2>There is no information retrieved from your database</h2>", width=820, styles={"text-align": "center"}),
                pn.pane.HTML("<h2>Please start conversation</h2>", width=820, styles={"text-align": "center"}),
                pn.pane.Image("assets/thinking.png", width=100, height=100, styles={"margin": "0 auto"}),
            )
        return pn.Row(
            pn.pane.HTML("<b>DB query:</b>", styles={"font_size": "16px", "margin": "5px 10px"}),
            pn.pane.HTML(f"{self.db_query}", styles={"font_size": "16px"}),
        )

    @param.depends('db_response')
    def get_sources(self):
        if not self.db_response:
            return 
        rlist=[pn.Row(pn.pane.HTML("<b>Relevant chunks from DB for query:</b>", styles={"font_size": "16px", "margin": "5px 10px"}))]
        for doc in self.db_response:
            rlist.append(pn.pane.HTML(
                f"<b>Page content=</b>{doc.page_content}<br><b>Metadata=</b>{doc.metadata}", 
                styles={
                    "margin": "10px 10px", 
                    "padding": "5px",
                    "background-color": "#fff", 
                    "border-radius": "5px", 
                    "border": "1px gray solid"
                }
            ))
        return pn.Column(pn.layout.Divider(), *rlist)
    
    def count_tokens(self):
        chunks = [tiktoken_len(doc.page_content) for doc in self.chunks]
        df = pd.DataFrame({'Token Count': chunks})
        # Create a histogram of the token count distribution
        fig = plt.figure(figsize=(3.5, 2.5))
        ax = fig.add_subplot()
        df.hist(column='Token Count', bins=40, ax=ax)
        plt.close(fig)  # Close the figure to prevent it from being displayed immediately
        histogram_widget = pn.pane.Matplotlib(fig)
        k = 4 if self.search_type == "score_threshold" else self.top_k
        return pn.Column(
            pn.Row(
                pn.Column(
                    pn.pane.HTML("Documents were splitted into chunks.", styles={"margin-bottom": "20px", "font-size": "16px"}),
                    pn.pane.HTML(f"<b>Selected chunk size</b> = 2000 / {k} = {2000 / k}", styles={"margin-bottom": "20px", "font-size": "16px"}),
                    pn.pane.HTML(f"<b>Min</b> = {min(chunks)}", styles={"margin-bottom": "20px", "font-size": "16px"}),
                    pn.pane.HTML(f"<b>Avg</b> = {int(sum(chunks) / len(chunks))}", styles={"margin-bottom": "20px", "font-size": "16px"}),
                    pn.pane.HTML(f"<b>Max</b> = {max(chunks)}", styles={"margin-bottom": "20px", "font-size": "16px"}),
                ),
                histogram_widget,
            ),
            pn.pane.HTML(f"Array of chunks: {chunks}")
        )
    
    
    @param.depends('chat_history')
    def get_history(self):
        if not self.chat_history:
            return pn.Column(
                pn.pane.HTML("<h2>There is no chat history</h2>", width=820, styles={"text-align": "center"}),
                pn.pane.HTML("<h2>Please start conversation</h2>", width=820, styles={"text-align": "center"}),
                pn.pane.Image("assets/thinking.png", width=100, height=100, styles={"margin": "0 auto"}),
            )
        rlist=[]
        for message in self.chat_history:
            rlist.append(pn.Row(
                pn.pane.HTML(f"{message.type.upper()}:", styles={"padding": "5px"}),
                pn.pane.HTML(
                    f"{message.content}", 
                    styles={
                        "padding": "5px",
                        "background-color": "#fff", 
                        "border-radius": "5px", 
                        "border": "1px gray solid"
                    }
                ),
            ))
        rlist.append(pn.pane.HTML("<b>Current chat history variable:</b>", styles={"font_size": "16px", "margin": "5px 10px"}))
        return pn.Column(*rlist[::-1])


# Callback to create a CBN object
def start(event):
    for widget in [menu, question, send_button, save_button, select_temperature, select_chain_type, select_search_type, select_top_k]:
        widget.disabled = not widget.disabled

    cbn = Chatbot(select_temperature.value, select_chain_type.value, select_search_type.value, select_top_k.value)
    chat_box = pn.bind(cbn.conversation, send_button)
    chat[0] = pn.panel(chat_box, loading_indicator=True, height=335)
    database[0] = pn.Column(
        pn.panel(cbn.get_last_question),
        pn.panel(cbn.get_sources),
    )
    splitter[0] = pn.Column(
        pn.panel(cbn.count_tokens)
    )
    chat_history[0] = pn.Column(
        pn.panel(cbn.get_history),
    )
    menu.value = menu.value
 
save_button.on_click(start)

# Callback to switch between chat, database and history
def switch_ui(event):
    ui[0] = chat if event.new == 'Conversation' else (database if event.new == 'Database' else (splitter if event.new == 'Splitter' else chat_history))

# Watcher to look after menu.value
menu.param.watch(switch_ui, "value", onlychanged=False)

# Define app template
app = pn.template.FastGridTemplate(
    title="CBN Chat",
    favicon="assets/gizmo.png",
    sidebar=[
        pn.Column(
            pn.pane.HTML("<h2>Settings</h2>", styles={"margin": "0 auto"}),
            select_temperature,
            pn.pane.HTML("Chain type:", styles={"font-size": "16px", "margin-bottom": "0"}),
            select_chain_type,
            pn.pane.HTML("Search type:", styles={"font-size": "16px", "margin-bottom": "0"}),
            select_search_type,
            select_top_k,
            save_button,
            margin=10
        )
    ],
    main_max_width="900px",
)
app.main[:3, :6] = pn.Column(menu, ui)
app.servable()
# panel serve clas.py --show --autoreload