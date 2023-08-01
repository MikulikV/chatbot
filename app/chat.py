from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores.chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
import panel as pn
import param
import os
from dotenv import load_dotenv 

# OPENAI_API_KEY
load_dotenv()
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

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
    end=5, 
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
    options=["Conversation", "Database", "Memory"],
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
# Chat history window
chat_history = pn.Row(pn.pane.HTML())

# CLASS CHATBOT
# Define LLM
def llm(temperature):
    return ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=temperature,
    )


# Define vector store
def get_vector_store():
    # define embedding
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    # create vector database from data
    vector_store = Chroma(
        collection_name="Database",
        embedding_function=embeddings,
        persist_directory="docs/chroma",
    )
    
    return vector_store


# Define retriever
def create_retriever(vector_store, search_type, k):
    if search_type == "score_threshold":
        retriever = vector_store.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": k})
    else:
        retriever = vector_store.as_retriever(search_type=search_type, search_kwargs={"k": k})

    return retriever


# Define propmpt
prompt = PromptTemplate(
    input_variables=["history", "context", "question"],
    template="""
% INSTRUCTIONS
- You are personal assistant named CBN Assistant who is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics.
- You are able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. 
- Always answer as helpfully as possible only in the manner of a deep believer, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
- Answer in question's language.
- Use the following chain of thought for specific questions:

Data: [chat history, context you are provided, your own knowledge base]
Question: the input you must answer
Thought: you should always think about what to do
Action: look for the answer in one of the Data elements.
Observation: the result of the action
... (repeat this Thought/Action/Observation for all the elements in Data)
Action: evaluate the answers and choose the best one
Thought: I now know the final answer
Final Answer: the final answer to the original input question

- Your output is the final answer.

% YOUR TASK
1. Answer a follow up question as best you can. You have access to the following conversation (delimited by <hs></hs>) and context (delimited by <ctx></ctx>):
------
<ctx>
{context}
</ctx>
------
<hs>
{history}
</hs>
------
Follow Up Input: {question}
Answer:

2. If necessary you can provide a link after answer the question related to the Bible or CBN to learn more. For example, if the question about faith: https://www2.cbn.com/search/faith?search=faith".
For questions about the SuperBook you can provide https://us-en.superbook.cbn.com/. 
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
        self.vector_store = get_vector_store()
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
            height=280
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
    chat_history[0] = pn.Column(
        pn.panel(cbn.get_history),
    )
    menu.value = menu.value
 
save_button.on_click(start)

# Callback to switch between chat, database and history
def switch_ui(event):
    ui[0] = chat if event.new == 'Conversation' else (database if event.new == 'Database' else chat_history)

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
    main_max_width="100%",
)
app.main[:5, :] = pn.Column(menu, ui)
app.servable()
# python -m panel serve app/chat.py --show --autoreload