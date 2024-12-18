# UI
import streamlit as st

# ML
import torch
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
import spacy

# LIB
from lib.embedding import *
from lib.llm import *
from lib.nlp import *

if "upload_id" not in st.session_state:
	st.session_state["upload_id"] = 0

@st.cache_resource(show_spinner=False)
def init():
	print("[1/4]\tLoading Emedding model..")
	embed = EmbeddingModel(
		model_name="_models/sentence/",
		half=True
	)

	print("[2/4]\tLoading LLM model..")
	llm = LLM("_models/llm/")
	
	print("[3/4]\tLoading NLP model..")
	nlp = spacy.blank("ru")
	nlp.from_disk("_models/nlp/")
	nlp.add_pipe('sentencizer')

	print("[4/4]\tLoading FIASS vectorstore..")
	vs = create_or_load_faiss_index("_database/ru_wikipedia",embed)

	print("Done init!")
	return vs,nlp,embed,llm

def nice_format_table(content : str) -> list[list]:
	context = []
	table = []
	lines = content.split("\n")
	start = next(i for i, line in enumerate(lines) if line and "|" in line)
	for i in range(start):
		context.append(lines[i])
	if start+1 < len(lines) and lines[start] == lines[start + 1]:
		start+=1
	for line in lines[start:]:
		if line.strip():
			row = [col.strip() for col in line.split("|") if col.strip()]
			table.append(row)

	return '\n'.join(context),table

st.title("Чилловые парни")

with st.spinner("Инициализация.."):
	vector_store, nlp, embedding_model, llm = init()

st.write("Загрузи документ, задай вопрос - получи ответ.")

uploaded_files = st.file_uploader("Загрузи документ (.docx)", type=("docx"),
								  accept_multiple_files=True,
								  key=st.session_state["upload_id"])

if uploaded_files:
	with st.spinner("Загрузка документов.."):
		filenames = []

		for uploaded in uploaded_files:
			print(f"Uploading file: {uploaded.name}..")

			with open("_upload/" + uploaded.name, "wb") as file:
				file.write(uploaded.getbuffer())

			filenames.append("_upload/" + uploaded.name)

		docx2faiss(filenames,vector_store,nlp)
		
	st.session_state["upload_id"] += 1

	st.rerun()

if "upload_id" in st.session_state and st.session_state["upload_id"] > 0 and not uploaded_files:
	st.success("Файлы загружены!")

question = st.text_area("Задай вопрос", placeholder = "Что вообще происходит?",)

if question and vector_store.index.ntotal <= 0:
	st.warning("Сначала загрузите файлы!")

if st.button("Показать ответ",disabled=not question):
	if vector_store.index.ntotal > 0:
		print(f"(1/3)\tSearching for tables: \"{question}\"..")

		with st.spinner("Поиск подходящих таблиц.."):
			docs = vector_store.similarity_search(question, k=1)
		
		print(f"(2/3)\tFound table: \"{docs[0].metadata}\". Summarizing..")

		with st.spinner("Таблица найдена. Суммаризация ответа.."):
			text = llm.answer(docs[0].page_content,question)

		print(f"(3/3)\tGot response: \"{text}\"")

		st.write("Суммаризация:")
		st.info(text)

		st.write("Найденный контекст:")
		st.write(nice_format_table(docs[0].page_content)[0])
		st.table(nice_format_table(docs[0].page_content)[1])
	else:
		st.error("База данных пуста!")