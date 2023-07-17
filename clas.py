from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.chains import RetrievalQA,  ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyMuPDFLoader
from langchain.document_loaders import DirectoryLoader
from langchain.schema import HumanMessage, AIMessage
import tiktoken

import panel as pn
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
    options=["similarity", "mmr"],
    button_type="primary",
    button_style="outline",
    styles={"margin-bottom": "20px"}
)
select_top_k = pn.widgets.IntSlider(
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

# Main layout
menu = pn.widgets.RadioButtonGroup(
    name="Menu", 
    options=['Conversation', 'Database', "Splitter", "Chat history"],
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
chat = pn.Column(pn.Row(question, send_button), pn.pane.HTML())
# Database window
database = pn.Row(pn.pane.HTML())
# Splitter window
splitter = pn.Row(pn.pane.HTML())
# Chat history window
chat_history = pn.Row(pn.pane.HTML())

# CLASS CHATBOT

# Define length_function for text_splitter
tokenizer = tiktoken.get_encoding("cl100k_base")

def tiktoken_len(text):
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)

# Define data loader
def load_data(source_directory, k):
    # load documents
    loader = DirectoryLoader(source_directory, glob="**/*.pdf", loader_cls=PyMuPDFLoader)
    documents = loader.load()
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
    chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=temperature), 
        chain_type=chain_type, 
        retriever=retriever, 
        return_source_documents=True,
        return_generated_question=True,
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
        self.source_directory = "docs"
        self.data = load_data(self.source_directory, self.top_k)
        self.vector_store = create_vector_store(self.data)
        self.retriever = create_retriever(self.vector_store, self.search_type, self.top_k)
        self.qa = create_chain(self.retriever, self.temperature, self.chain_type)

    def conversation(self, query):
        if query:
            result = self.qa({"question": query, "chat_history": self.chat_history})
            self.chat_history.extend([(HumanMessage(content=query).content, AIMessage(content=result['answer']).content)])
            self.db_query = result["generated_question"]
            self.db_response = result["source_documents"]
            self.answer = result['answer'] 
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
        rlist=[pn.Row(pn.pane.HTML("<b>Relevant chunks from DB:</b>", styles={"font_size": "16px", "margin": "5px 10px"}))]
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
    
    @param.depends('data')
    def count_tokens(self):
        chunks = [tiktoken_len(doc.page_content) for doc in self.data]
        return pn.Column(
            pn.pane.HTML("Documents were splitted into chunks.", styles={"margin-bottom": "20px", "font-size": "16px"}),
            pn.pane.HTML(f"<b>Selected chunk size</b> = 2000 / {self.top_k} = {2000 / self.top_k}", styles={"margin-bottom": "20px", "font-size": "16px"}),
            pn.pane.HTML(f"<b>Min</b> = {min(chunks)}", styles={"margin-bottom": "20px", "font-size": "16px"}),
            pn.pane.HTML(f"<b>Avg</b> = {int(sum(chunks) / len(chunks))}", styles={"margin-bottom": "20px", "font-size": "16px"}),
            pn.pane.HTML(f"<b>Max</b> = {max(chunks)}", styles={"margin-bottom": "20px", "font-size": "16px"}),
            pn.pane.HTML(f"<b>List of the chunks:</b> {chunks}"),
        )
    
    @param.depends('conversation')
    def get_history(self):
        if not self.chat_history:
            return pn.Column(
                pn.pane.HTML("<h2>There is no chat history</h2>", width=820, styles={"text-align": "center"}),
                pn.pane.HTML("<h2>Please start conversation</h2>", width=820, styles={"text-align": "center"}),
                pn.pane.Image("assets/thinking.png", width=100, height=100, styles={"margin": "0 auto"}),
            )
        rlist=[]
        for elem in self.chat_history:
            rlist.append(pn.pane.HTML(
                f"{elem}", 
                styles={
                    "padding": "5px",
                    "background-color": "#fff", 
                    "border-radius": "5px", 
                    "border": "1px gray solid"
                }
            ))
        rlist.append(pn.pane.HTML("<b>Current chat history variable:</b>", styles={"font_size": "16px", "margin": "5px 10px"}))
        return pn.Column(*rlist[::-1])

cbn = None

# Callback to create a CBN object
def start(event):
    for widget in [menu, question, send_button, save_button, select_temperature, select_chain_type, select_search_type, select_top_k]:
        widget.disabled = not widget.disabled

    global cbn
    cbn = Chatbot(select_temperature.value, select_chain_type.value, select_search_type.value, select_top_k.value)
    chat_box = pn.bind(cbn.conversation, question)
    chat[1] = pn.panel(chat_box, loading_indicator=True, height=300)
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