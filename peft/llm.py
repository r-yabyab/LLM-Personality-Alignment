import torch
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

class LocalLLM:
    def __init__(self, model_path, max_seq_length=2048):
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=True,
        )

        self.tokenizer = get_chat_template(
            self.tokenizer,
            chat_template="llama",
        )

    def generate(self, messages, temperature=0.8, max_new_tokens=300):
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            tokenize=True,
        ).to("cuda")

        attention_mask = (inputs != self.tokenizer.pad_token_id).long()

        outputs = self.model.generate(
            input_ids=inputs,
            attention_mask=attention_mask,
            temperature=temperature,
            top_p=0.95,
            top_k=64,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )

        return self.tokenizer.decode(
            outputs[0][inputs.shape[-1]:],
            skip_special_tokens=True
        )