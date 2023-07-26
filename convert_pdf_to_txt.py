import os
import re
from langchain.document_loaders import PyMuPDFLoader, Docx2txtLoader


files_dir = os.getcwd() + "/docs"
os.chdir(files_dir)
txt_dir = os.getcwd() + "/docs/txt"

# # iterate through all file
# for filename in os.listdir(files_dir):
#     # Check whether file is in pdf format or not
#     if filename.endswith(".pdf"):
#         pdf_file = open(os.path.join(files_dir, filename), 'rb')
#         pdf_reader = PyPDF2.PdfReader(pdf_file)
#         text = ''
#         for i in range(len(pdf_reader.pages)):
#             page = pdf_reader.pages[i]
#             text += page.extract_text()
#             text += ' '
#         processed_text = text.replace('\n\n', '\n')
#         processed_text = processed_text.replace('\n', ' ')
#         cleaned_text = clean_text(text, cleaning_functions)
#         txt_filename = os.path.splitext(filename)[0] + '.txt'
#         txt_file = open(os.path.join(txt_dir, txt_filename), 'w')
#         txt_file.write(processed_text)
#         pdf_file.close()
#         txt_file.close()

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
    # elif filename.endswith(".docx"):
    #     docx_reader = Docx2txtLoader(filename).load()
    #     text = ''
    #     for doc in docx_reader:
    #         text += doc.page_content
    #     text = text.replace('\n', ' ') # makes the text in one string
    #     cleaned_text = clean_text(text, cleaning_functions)
    #     cleaned_text = re.sub(" +", " ", cleaned_text) # remove multiple spaces
    #     txt_filename = os.path.splitext(filename)[0] + '.txt'
    #     txt_file = open(os.path.join(txt_dir, txt_filename), 'w')
    #     txt_file.write(cleaned_text)
    #     txt_file.close()
