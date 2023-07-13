import panel as pn

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
select__temperature = pn.widgets.FloatSlider(
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
select__top_k = pn.widgets.IntSlider(
    name="Number of relevant chunks",
    start=1, 
    end=10, 
    step=1, 
    value=4, 
    styles={"font-size": "16px"}
)

# Main layout
# Conversation window
question = pn.widgets.TextInput(
    value="", placeholder="Send a message ...", width=720, height=40, toolbar=False,
)
send_button = pn.widgets.Button(name="Send", width=80, height=40)




title = pn.pane.HTML(
    object=f"<h2>{menu.value}</h2>"
)

chat_box = pn.Column(
    pn.Row(question, send_button),
    pn.widgets.ChatBox(
        value=[
            {"You": "Hello!"},
            {"Gizmo": "Hi, I'm Gizmo, how can I help you today!"},
        ],
        message_icons={
            "You": "assets/user.png",
            "Gizmo": "assets/gizmo.png",
        },
        show_names=False,
        allow_input=False,
    )
)
database = pn.widgets.ColorPicker(name='Color Picker', value='#99ef78')
chat_history = pn.widgets.ColorPicker(name='Color Picker', value='#000')


layout = pn.Column(
    title,
    chat_box,
)

def switch_layout( event):
    layout[0].object = f"<h2>{event.new}</h2>"
    layout[1] = chat_history if event.new == "Chat history" else (database if event.new == "Database" else chat_box)

watcher = menu.param.watch(switch_layout, 'value')



template = pn.template.FastGridTemplate(
    title="CBN Chat",
    favicon="assets/gizmo.png",
    sidebar=[
        pn.Column(
            menu,
            select__temperature,
            pn.pane.HTML("Chain type:", styles={"font-size": "16px", "margin-bottom": "0"}),
            select_chain_type,
            pn.pane.HTML("Search type:", styles={"font-size": "16px", "margin-bottom": "0"}),
            select_search_type,
            select__top_k,
            margin=10
        )
    ],
    main_max_width="900px",
)

template.main[:3, :6] = layout

template.servable()

# panel serve cbn.py --show --autoreload