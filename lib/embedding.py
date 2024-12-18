import faiss
import torch
from torch import nn
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores.faiss import FAISS
from transformers import AutoTokenizer, AutoModel
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.docstore.in_memory import InMemoryDocstore
from pydantic import PrivateAttr
import os

class EmbeddingModel(Embeddings):
	_tokenizer: AutoTokenizer = PrivateAttr()
	_model: nn.Module = PrivateAttr()
	_device: str = PrivateAttr()
	_half: bool = PrivateAttr()
	_normalize_embeddings: bool = PrivateAttr(default=False)

	def __init__(self, model_name: str, half: bool = True):
		self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self._tokenizer = AutoTokenizer.from_pretrained(model_name)
		self._model = AutoModel.from_pretrained(model_name)
		self._model.eval()
		self._model.to(self._device)

		if half and 'cuda' == self._device:
			self._model.half()

		self._half = half

	def embed_documents(self, texts: list[str]) -> list[list[float]]:
		# Используем mean pooling для получения sentence embeddings
		batch_size = 256
		embeddings = []
		for i in range(0, len(texts), batch_size):
			batch_texts = texts[i:i+batch_size]
			inputs = self._tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True, max_length=512)
			inputs = {k: v.to(self._device) for k,v in inputs.items()}

			with torch.no_grad():
				outputs = self._model(**inputs)
				last_hidden_state = outputs.last_hidden_state
				attention_mask = inputs['attention_mask']

				# Mean Pooling
				input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
				sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
				sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
				cls_emb = (sum_embeddings / sum_mask).float().cpu().numpy()

				if self._normalize_embeddings:
					norm = (cls_emb**2).sum(axis=1, keepdims=True)**0.5
					cls_emb = cls_emb / norm
				embeddings.extend(cls_emb.tolist())
		return embeddings

	def embed_query(self, text: str) -> list[float]:
		return self.embed_documents([text])[0]

def create_empty_faiss_index(embedding_model : Embeddings, distance_strategy : int = DistanceStrategy.COSINE) -> FAISS:
    # Получаем размерность через фиктивный текст
    dummy_emb = embedding_model.embed_documents(["hello"])
    dim = len(dummy_emb[0])

    if distance_strategy == DistanceStrategy.COSINE:
        index = faiss.IndexFlatIP(dim)
    else:
        index = faiss.IndexFlatL2(dim)

    docstore = InMemoryDocstore({})
    vectorstore = FAISS(
        index=index,
        docstore=docstore,
        index_to_docstore_id={},
        embedding_function=embedding_model,
        distance_strategy=distance_strategy
    )
    return vectorstore

def create_or_load_faiss_index(checkpoint_name : str, embedding_model : Embeddings) -> FAISS:
	if os.path.exists(checkpoint_name):
		vectorstore = FAISS.load_local(checkpoint_name, embedding_model, allow_dangerous_deserialization=True)
	else:
		vectorstore = create_empty_faiss_index(embedding_model, distance_strategy = DistanceStrategy.COSINE)

	return vectorstore
