from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.chains import RetrievalQA,  ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyMuPDFLoader
from langchain.document_loaders import DirectoryLoader

import panel as pn
import param

from dotenv import load_dotenv 


load_dotenv()

# WIDGETS

# Sidebar
# menu to switch main template
menu = pn.widgets.RadioButtonGroup(
    name="Menu", 
    options=['Conversation', 'Database', "Chat history"],
    button_type="primary",
    button_style="outline",
    styles={"margin-bottom": "20px"}
)
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
    options=["similarity", "MMR"],
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
    name="Save settings", 
    button_type="success", 
    width=120,
    height=35
)
edit_button = pn.widgets.Button(
    name="Edit settings", 
    button_type="warning", 
    width=120,
    height=35,
    disabled=True
)

# Main layout
title = pn.pane.HTML(object=f"<h2>{menu.value}</h2>")
# Conversation window
question = pn.widgets.TextInput(
    value="", placeholder="Send a message", width=720, height=40, disabled=True
)
send_button = pn.widgets.Button(name="Send", width=80, height=40, disabled=True)
clearhistory_button = pn.widgets.Button(name="Clear History", button_type="warning")

def save(event):
    for widget in [question, send_button, save_button, edit_button, select_temperature, select_chain_type, select_search_type, select_top_k]:
        widget.disabled = not widget.disabled

save_button.on_click(save)
edit_button.on_click(save)


# Create a CBN Chat chain
def load_db(source_directory, search_type, chain_type, k, temperature):
    # load documents
    loader = DirectoryLoader(source_directory, glob="**/*.pdf", loader_cls=PyMuPDFLoader)
    documents = loader.load()
    # split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)
    # define embedding
    embeddings = OpenAIEmbeddings()
    # create vector database from data
    db = DocArrayInMemorySearch.from_documents(docs, embeddings)
    # define retriever
    retriever = db.as_retriever(search_type=search_type, search_kwargs={"k": k})
    # create a chatbot chain. Memory is managed externally.
    qa = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=temperature), 
        chain_type=chain_type, 
        retriever=retriever, 
        return_source_documents=True,
        return_generated_question=True,
    )
    return qa


class Chatbot(param.Parameterized):
    chat_history = param.List([])
    answer = param.String("")
    panels = param.List([])
    db_query = param.String("")
    db_response = param.List([])
    
    def __init__(self, t, c, s, k, **params):
        super(Chatbot, self).__init__(**params)
        self.temperature = t
        self.chain_type = c
        self.search_type = s
        self.top_k = k
        self.source_directory = "docs"
        self.qa = load_db(self.source_directory, self.search_type, self.chain_type, self.top_k, self.temperature)

    def conversation(self, query):
        if query:
            result = self.qa({"question": query, "chat_history": self.chat_history})
            self.chat_history.extend([(query, result["answer"])])
            self.db_query = result["generated_question"]
            self.db_response = result["source_documents"]
            self.answer = result['answer'] 
            self.panels.extend([
                {"You": query},
                {"Gizmo": self.answer},
            ])

        return pn.widgets.ChatBox(
            value=self.panels,
            message_icons={
                "You": "assets/user.png",
                "Gizmo": "assets/gizmo.png",
            },
            show_names=False,
            allow_input=False,
        )


cbn = None



def start(event):
    global cbn
    if event.new:
        cbn = Chatbot(select_temperature.value, select_chain_type.value, select_search_type.value, select_top_k.value)
        chat_box = pn.bind(cbn.conversation, question)
        chat[1] = pn.panel(chat_box, loading_indicator=True, height=300)
    else:
        cbn = None
        chat[1] = pn.pane.HTML(f"{cbn}")

save_button.param.watch(start, 'disabled')

chat = pn.Column(pn.Row(question, send_button), pn.pane.HTML(object="To start conversation set up your settings"))

ui = chat

picker = pn.widgets.ColorPicker(name='Color Picker', value='#99ef78')

app = pn.template.FastGridTemplate(
    title="CBN Chat",
    favicon="assets/gizmo.png",
    sidebar=[
        pn.Column(
            menu,
            select_temperature,
            pn.pane.HTML("Chain type:", styles={"font-size": "16px", "margin-bottom": "0"}),
            select_chain_type,
            pn.pane.HTML("Search type:", styles={"font-size": "16px", "margin-bottom": "0"}),
            select_search_type,
            select_top_k,
            pn.Row(edit_button, save_button, styles={"margin": "0 auto"}),
            margin=10
        )
    ],
    main_max_width="900px",
)
app.main[:3, :6] = ui
app.servable()