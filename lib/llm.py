import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

class LLM:
	def __init__(self, model_name : str,store : bool = False) -> None:
		"""
		Инициализация LLM с указанной моделью.

		:param model_name: str, имя модели для загрузки
		"""
		bnb_config = BitsAndBytesConfig(
			load_in_4bit=True,
			bnb_4bit_quant_type="nf4",
			bnb_4bit_compute_dtype=torch.float16, # Используем torch.float16 вместо 'fp16'
			bnb_4bit_use_double_quant=True
		)

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.tokenizer = AutoTokenizer.from_pretrained(model_name)
		self.model = AutoModelForCausalLM.from_pretrained(
			model_name,
			quantization_config=bnb_config
		).to(self.device)

		if store:
			self.model.save_pretrained("./")

	def answer(self, context : str, question : str) -> str:
		"""
		Генерация ответа на вопрос с использованием предоставленного контекста.

		:param context: str, контекст для ответа
		:param question: str, вопрос для ответа
		:return: str, сгенерированный ответ
		"""
		prompt_in_chat_format = [
			{
				"role": "system",
				"content": """Using the information contained in the context,
	give a comprehensive answer to the question.
	Respond only to the question asked, response should be concise and relevant to the question.
	Provide the number of the source document when relevant.
	If the answer cannot be deduced from the context, give information based on your own knowledge.""",
			},
			{
				"role": "user",
				"content": f"""Context:
	{context}
	---
	Now here is the question you need to answer.

	Question: {question}""",
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