from unsloth import FastLanguageModel
import torch
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
logging_dir = f"outputs-full/logs_{timestamp}"

max_seq_length = 512 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/mistral-7b-bnb-4bit",
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/Meta-Llama-3.1-8B-bnb-4bit",
] # More models at https://huggingface.co/unsloth

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3.1-8B-bnb-4bit",
    # model_name = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit", # Choose ANY! eg teknium/OpenHermes-2.5-Mistral-7B
    # model_name = "unsloth/mistral-7b-v0.3", # Choose ANY! eg teknium/OpenHermes-2.5-Mistral-7B
     max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

from unsloth.chat_templates import get_chat_template

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "llama", # Supports zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, unsloth
    # chat_template = "mistral", # Supports zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, unsloth
    # mapping = {"role" : "from", "content" : "value", "user" : "human", "assistant" : "gpt"}, # ShareGPT style
    # map_eos_token = True, # Maps <|im_end|> to </s> instead
)

def formatting_prompts_func(examples):
    convos = examples["conversations"]
    
    # texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
    
    
    texts = [tokenizer.apply_chat_template(
        [msg for pair in convo for msg in pair],
        tokenize = False, add_generation_prompt = False
    ) for convo in convos]
    
    return { "text" : texts, }
pass

from datasets import load_dataset
dataset = load_dataset("json", data_files="../data/data.jsonl", split = "train")

split_dataset = dataset.train_test_split(test_size=0.2, seed=3407)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]



train_dataset = train_dataset.map(formatting_prompts_func, batched = True,)
eval_dataset = eval_dataset.map(formatting_prompts_func, batched = True,)

# Print a sample to verify formatting
print("Sample formatted prompt:")
print(train_dataset[0]["text"])
print(f"Train dataset size: {len(train_dataset)}")
print(f"Eval dataset size: {len(eval_dataset)}")


from transformers import TrainerCallback
from torch.utils.tensorboard import SummaryWriter

class EvalLossSpikeCallback(TrainerCallback):
    def __init__(self, eval_dataset, tokenizer, threshold=0.05, top_k=3, max_seq_length=512, logging_dir=None):
        self.prev_eval_loss = None
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.threshold = threshold  # relative increase, e.g. 0.05 = 5%
        self.top_k = top_k
        self.max_seq_length = max_seq_length
        self.tb_writer = SummaryWriter(logging_dir) if logging_dir else None

    def on_evaluate(self, args, state, control, metrics=None, model=None, **kwargs):
        current_loss = metrics.get("eval_loss")
        if current_loss is None:
            return

        if self.prev_eval_loss is not None:
            increase = (current_loss - self.prev_eval_loss) / self.prev_eval_loss
            if increase > self.threshold:
                print(f"\n[SPIKE] eval_loss: {self.prev_eval_loss:.4f} -> {current_loss:.4f} (+{increase*100:.1f}%)")
                self._log_worst_samples(model, state.global_step)

        self.prev_eval_loss = current_loss

    def _log_worst_samples(self, model, step):
        import torch
        model.eval()
        device = next(model.parameters()).device
        losses = []

        with torch.no_grad():
            for i, sample in enumerate(self.eval_dataset):
                text = sample["text"]
                enc = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_seq_length,
                ).to(device)
                labels = enc["input_ids"].clone()
                outputs = model(**enc, labels=labels)
                losses.append((i, outputs.loss.item(), text))

        losses.sort(key=lambda x: x[1], reverse=True)
        print(f"  Top {self.top_k} highest-loss eval samples (step {step}):")
        for rank, (idx, loss, text) in enumerate(losses[:self.top_k]):
            preview = text[:400].replace("\n", " ")
            print(f"  [{rank+1}] idx={idx} loss={loss:.4f} | {preview}\n")
            if self.tb_writer:
                self.tb_writer.add_text(
                    f"spike/top{rank+1}_loss_{loss:.4f}",
                    f"**idx**: {idx}  **loss**: {loss:.4f}\n\n```\n{text[:800]}\n```",
                    global_step=step,
                )

        model.train()

from trl import SFTConfig, SFTTrainer
spike_cb = EvalLossSpikeCallback(
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    threshold=0.05,
    top_k=3,
    max_seq_length=max_seq_length,
    logging_dir=logging_dir,
)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    # train_dataset = dataset,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    packing = False, # Can make training 5x faster for short sequences.
    callbacks = [spike_cb],
    args = SFTConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        num_train_epochs = 1,
        # max_steps = 60,
        eval_strategy="steps",
        eval_steps=10, # this goes through the eval data every n steps
        # save_strategy="steps",
        # save_steps=10,
        learning_rate = 2e-4,
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs-full",
        # report_to = "none", # Use this for WandB etc
        logging_dir=logging_dir,
        report_to = "tensorboard",
    ),
)

trainer_stats = trainer.train()