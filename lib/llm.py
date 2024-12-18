import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoConfig

class LLM:
	def __init__(self, model_name : str,store : bool = False) -> None:
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.tokenizer = AutoTokenizer.from_pretrained(model_name)
		self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)

		if store:
			self.model.save_pretrained("_llm/")

	def answer(self, context : str, question : str) -> str:
		prompt_in_chat_format = [
    {
        "role": "system",
        "content": """Используя информацию, содержащуюся в контексте,
дайте исчерпывающий ответ на вопрос.
Отвечайте только на заданный вопрос, ответ должен быть кратким и релевантным.
Укажите номер исходного документа, если это уместно.
Если ответ нельзя вывести из контекста, дайте информацию на основе собственных знаний.""",
    },
    {
        "role": "user",
        "content": f"""Контекст:
{context}
---
Теперь вот вопрос, на который нужно ответить.

Вопрос: {question}""",
    },
]
		
		# Преобразование шаблона в текст
		text = self.tokenizer.apply_chat_template(
			prompt_in_chat_format,
			tokenize=False,
			add_generation_prompt=True
		)
		model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

		# Генерация ответа
		generated_ids = self.model.generate(
			**model_inputs,
			max_new_tokens=256
		)
		
		# Удаление токенов входного текста из сгенерированного
		generated_ids = [
			output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
		]

		response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
		return response[0]