import os
from langchain.document_loaders import PyMuPDFLoader


files_dir = os.getcwd() + "/docs"
os.chdir(files_dir)
txt_dir = os.getcwd() + "/docs/txt"

# iterate through all file
for filename in os.listdir(files_dir):
    # Check whether file is in pdf format or not
    if filename.endswith(".pdf"):
        pdf_reader = PyMuPDFLoader(filename).load()
        text = ''
        for doc in pdf_reader:
            text += doc.page_content + ' '
        txt_filename = os.path.splitext(filename)[0] + '.txt'
        txt_file = open(os.path.join(txt_dir, txt_filename), 'w')
        txt_file.write(text)
        txt_file.close()
