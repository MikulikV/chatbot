from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationSummaryBufferMemory
from langchain.agents import Tool
from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyMuPDFLoader
from langchain.document_loaders import DirectoryLoader
from langchain.schema import HumanMessage, AIMessage
import tiktoken

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
    options=["Conversation", "Database", "Steps", "Splitter", "Memory"],
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
# Agent's steps
steps = pn.Row(pn.pane.HTML())
# Database window
database = pn.Row(pn.pane.HTML())
# Splitter window
splitter = pn.Row(pn.pane.HTML())
# Chat history window
memory = pn.Row(pn.pane.HTML())

# CLASS CHATBOT

# Define LLM
def llm(temperature):
    return ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=temperature,
    )


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


# Define propmpts 
chain_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""Answer the question based only on the following pieces of context. 
If the question cannot be answered using the information provided answer with "I don't know".

{context}

Question: {question}

Answer:"""
)
chain_type_kwargs = {"prompt": chain_prompt}

system_message = """
You are Personal Assistant named Gizmo as a character from the SuperBook.

Gizmo is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Gizmo is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

Gizmo is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Gizmo is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.     

Gizmo answers questions as a deep believer in God and in no other way, for example Gizmo doesn't answer like pirate.

Overall, Gizmo is a powerful system that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Gizmo is here to assist.
"""

def agent_prompt(cbn_agent, tools):
    return cbn_agent.agent.create_prompt(
        system_message=system_message,
        tools=tools
    )


# Define chain
def create_chain(llm, retriever, chain_type):    
    # create a chatbot chain. Memory is managed externally.
    chain = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type=chain_type, 
        retriever=retriever, 
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs,
    )
    return chain


# Define memory
def conversational_memory(llm):
    return ConversationSummaryBufferMemory(
        llm=llm, 
        memory_key="chat_history", 
        input_key='input', 
        output_key="output", 
        return_messages=True, 
        max_token_limit=650
    )


# Define tools
def agent_tools(cbn_chain):
    tools = [
        Tool(
            name="CBN",
            func=cbn_chain,
            description="use this tool when you need to answer all the questions about CBN, Bible, characters from the Bible and Superbook and its characters"
        ),
    ]

    return tools


# Initialize agent
def agent(llm, tools, memory):
    agent = initialize_agent(
        agent='chat-conversational-react-description',
        tools=tools,
        llm=llm,
        max_iterations=3,
        memory=memory,
        return_intermediate_steps=True,
        handle_parsing_errors="Check your output and make sure it conforms!",
    )

    return agent


# Define CBN class
class Chatbot(param.Parameterized):
    answer = param.String("")
    panels = param.List([])
    db_query = param.List([])
    db_response = param.List([])
    chat_history = param.List([])
    steps = param.List([])
    
    def __init__(self, t, c, s, k, **params):
        super(Chatbot, self).__init__(**params)
        self.temperature = t
        self.chain_type = c
        self.search_type = s
        self.top_k = k
        self.source_directory = "docs"
        self.llm = llm(self.temperature)
        self.data = load_data(self.source_directory, self.top_k)
        self.vector_store = create_vector_store(self.data)
        self.retriever = create_retriever(self.vector_store, self.search_type, self.top_k)
        self.qa = create_chain(self.llm, self.retriever, self.chain_type)
        self.memory = conversational_memory(self.llm)
        self.tools = agent_tools(self.qa)
        self.agent = agent(self.llm, self.tools, self.memory)
        self.agent.agent.llm_chain.prompt = agent_prompt(self.agent, self.tools)

    def conversation(self, query):
        if query:
            response = self.agent({"input": query})
            self.chat_history = response["chat_history"]
            self.steps = response["intermediate_steps"]
            self.db_query = []
            self.db_response = []
            if self.steps:
                qlist = []
                rlist = []
                for step in self.steps:
                    if not isinstance(step[1], str):
                        qlist.append(step[1]["query"])
                        rlist.append(step[1]["source_documents"])
                self.db_query = qlist
                self.db_response = rlist
            self.answer = response['output'] 
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
    
    @param.depends('steps')
    def get_steps(self):
        if not self.steps:
            return pn.Column(
                pn.pane.HTML("<h2>There is no Agent steps</h2>", width=820, styles={"text-align": "center"}),
                pn.pane.HTML("<h2>That means llm answered your question without using your database or you didn't start conversation yet</h2>", width=820, styles={"text-align": "center"}),
                pn.pane.Image("assets/thinking.png", width=100, height=100, styles={"margin": "0 auto"}),
            )
        rlist = []
        for step in self.steps:
            rlist.append(
                pn.pane.HTML(f"{step[0].log[1:-2].replace(',', ',<br>')}", 
                styles={
                    "font-size": "16px",
                    "color": "green",
                    "padding": "5px",
                    "background-color": "#fff", 
                    "border-radius": "5px", 
                    "border": "1px gray solid"
                }))
        return pn.Column(*rlist)

    
    @param.depends('db_query')
    def get_last_question(self):
        if not self.db_query:
            return pn.Column(
                pn.pane.HTML("<h2>There is no information retrieved from your database</h2>", width=820, styles={"text-align": "center"}),
                pn.pane.HTML("<h2>That means llm answered your question without using your database or you didn't start conversation yet</h2>", width=820, styles={"text-align": "center"}),
                pn.pane.Image("assets/thinking.png", width=100, height=100, styles={"margin": "0 auto"}),
            )
        rlist = [pn.pane.HTML("<b>DB query:</b>", styles={"font_size": "16px", "margin": "5px 10px"})]
        for i in range(len(self.db_query)):
            rlist.append(pn.pane.HTML(f"{self.db_query[i]}\n", styles={"font_size": "16px"}))
        return pn.Row(*rlist)

    @param.depends('db_response')
    def get_sources(self):
        if not self.db_response:
            return 
        for i in range(len(self.db_response)):
            rlist=[pn.Row(pn.pane.HTML(f"<b>Relevant chunks from DB for query:</b> {self.db_query[i]}", styles={"font_size": "16px", "margin": "5px 10px"}))]
            for doc in self.db_response[i]:
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
        df = pd.DataFrame({'Token Count': chunks})
        # Create a histogram of the token count distribution
        fig = plt.figure(figsize=(3.5, 2.5))
        ax = fig.add_subplot()
        df.hist(column='Token Count', bins=40, ax=ax)
        plt.close(fig)  # Close the figure to prevent it from being displayed immediately
        histogram_widget = pn.pane.Matplotlib(fig)
        return pn.Row(
            pn.Column(
                pn.pane.HTML("Documents were splitted into chunks.", styles={"margin-bottom": "20px", "font-size": "16px"}),
                pn.pane.HTML(f"<b>Selected chunk size</b> = 2000 / {self.top_k} = {2000 / self.top_k}", styles={"margin-bottom": "20px", "font-size": "16px"}),
                pn.pane.HTML(f"<b>Min</b> = {min(chunks)}", styles={"margin-bottom": "20px", "font-size": "16px"}),
                pn.pane.HTML(f"<b>Avg</b> = {int(sum(chunks) / len(chunks))}", styles={"margin-bottom": "20px", "font-size": "16px"}),
                pn.pane.HTML(f"<b>Max</b> = {max(chunks)}", styles={"margin-bottom": "20px", "font-size": "16px"}),
            ),
            histogram_widget,
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

cbn = None

# Callback to create a CBN object
def start(event):
    for widget in [menu, question, send_button, save_button, select_temperature, select_chain_type, select_search_type, select_top_k]:
        widget.disabled = not widget.disabled

    global cbn
    cbn = Chatbot(select_temperature.value, select_chain_type.value, select_search_type.value, select_top_k.value)
    chat_box = pn.bind(cbn.conversation, question)
    chat[0] = pn.panel(chat_box, loading_indicator=True, height=335)
    database[0] = pn.Column(
        pn.panel(cbn.get_last_question),
        pn.panel(cbn.get_sources),
    )
    splitter[0] = pn.Column(
        pn.panel(cbn.count_tokens)
    )
    steps[0] = pn.Column(
        pn.panel(cbn.get_steps)
    )
    memory[0] = pn.Column(
        pn.panel(cbn.get_history),
    )
    menu.value = menu.value
 
save_button.on_click(start)

# Callback to switch between chat, database and history
def switch_ui(event):
    ui[0] = chat if event.new == 'Conversation' else (database if event.new == 'Database' else (splitter if event.new == 'Splitter' else (steps if event.new == 'Steps' else memory)))

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