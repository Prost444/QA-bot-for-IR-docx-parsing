# UI
import streamlit as st

# ML
import torch
from langchain.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
import spacy

# LIB
from lib.multi_gpu_embeddings import *
from lib.llm import LLM
from lib.docx_to_lang import docx2faiss

vectorstore, nlp, embedding_model, llm = None, None, None, None

def init():
	global vectorstore, nlp, embedding_model, llm
	
	embedding_model = MultiGPUHuggingFaceEmbeddings(
		model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
		half=True,
		model_kwargs={"device": "cuda" if torch.cuda.is_available() else "mps" },
		encode_kwargs={"normalize_embeddings": True},
		multi_process=False
	)

	llm = LLM("t-tech/T-lite-it-1.0",True)

	nlp = spacy.load('ru_core_news_sm')

	vectorstore = create_or_load_faiss_index("fiass",embedding_model)
	
	print("Готов!")

init()

# Заголовок и описание
st.title("Чилловые парни")
st.write("Загрузи документ, задай вопрос - получи ответ.")

# Загрузка файла
uploaded_file = st.file_uploader("Загрузи документ (.docx)", type=("docx"), accept_multiple_files=True)

# Вопрос
question = st.text_area(
    "Задай вопрос",
    placeholder = "Что вообще происходит?",
#    disabled = not uploaded_file,
)

if uploaded_file:
	docx2faiss(uploaded_file,vectorstore,nlp)

if st.button("Показать ответ"):
	if question:
		docs = vectorstore.similarity_search(question, k=1)

		text = llm.answer(docs[0].page_content,question)

		st.write(text)
	else:
		st.write("Запрос пуст!")