import os
import pickle

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, concatenate_datasets

from nnsight import NNsight

########################################
# 1. Original filter/dataset code
########################################

def filter_questions(data, tokenizer, k, max_length=256):
    """
    Filters out examples where the formatted question exceeds k tokens.
    """
    filtered_data = []
    choice_labels = ["A", "B", "C", "D"]

    for example in data:
        question_text = example["question"]
        choices = example["choices"]

        formatted_choices = [
            f"({label}) {choice}"
            for label, choice in zip(choice_labels[:len(choices)], choices)
        ]
        prompt = (
            f"{question_text}\n"
            f"Your options are: " + ", ".join(formatted_choices) + ".\n"
        )

        encoding = tokenizer(
            prompt,
            add_special_tokens=False,
            padding=False,
            truncation=True,
            max_length=max_length
        )

        if len(encoding["input_ids"]) <= k:
            filtered_data.append(example)

    return filtered_data


class GenerativeQuestionDataset(Dataset):
    """Dataset for multi-choice generative inference."""
    def __init__(self, data, tokenizer, max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def format_question(self, example):
        """Formats question for multi-choice generation."""
        question_text = example["question"]
        choices = example["choices"]
        correct_label = example["answer"]  # Index of the correct answer

        # Create multi-choice answer formatting
        choice_labels = ["A", "B", "C", "D", "E"][:len(choices)]
        formatted_choices = [f"({label}) {choice}" for label, choice in zip(choice_labels, choices)]

        # Construct the full prompt
        prompt = (
            f"{question_text}\n"
            f"Your options are: " + ", ".join(formatted_choices) + "\n"
            "Write your final answer in \\boxed{}. It is very important that you write your final answer in \\boxed{}.\n"
            "<think>\n"
        )

        # Tokenize to find where generation should start
        encoding = self.tokenizer(
            prompt,
            add_special_tokens=False,
            padding=False,
            truncation=True,
            max_length=self.max_length
        )
        generation_start_idx = len(encoding["input_ids"])

        # Apply padding for model input
        model_encoding = self.tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": model_encoding["input_ids"].squeeze(0),
            "attention_mask": model_encoding["attention_mask"].squeeze(0),
            "prompt": prompt,
            "generation_start_idx": generation_start_idx,
            "correct_label": correct_label,
            "subject": example["subject"]
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.format_question(self.data[idx])


def collate_fn(batch):
    """Collate function for batch inference."""
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    prompts = [b["prompt"] for b in batch]
    generation_start_indices = [b["generation_start_idx"] for b in batch]
    correct_labels = [b["correct_label"] for b in batch]
    subjects = [b["subject"] for b in batch]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "prompts": prompts,
        "generation_start_idx": generation_start_indices,
        "correct_labels": correct_labels,
        "subjects": subjects
    }

########################################
# 2. Multi-GPU Setup Helpers
########################################

def setup_distributed(world_size, rank):
    """
    Initialize the default process group.
    """
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank
    )
    torch.cuda.set_device(rank)

def cleanup_distributed():
    """Destroy the process group when done."""
    dist.destroy_process_group()

########################################
# 3. Generation Function
########################################

def generate_answers(model, dataloader, tokenizer, max_gen_length=50, temp=0.6, stop_strings=None):
    """
    Generates answers in batches.
    Each GPU will process its share of the dataset.
    """
    model.eval()
    local_results = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            print(f"[Rank {dist.get_rank()}] Starting batch {batch_idx}")

            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            prompts = batch["prompts"]
            generation_start_indices = batch["generation_start_idx"]
            correct_labels = batch["correct_labels"]
            subjects = batch["subjects"]

            # Generate responses
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=input_ids.shape[1] + max_gen_length,
                pad_token_id=tokenizer.eos_token_id,
                temperature=temp,
                tokenizer=tokenizer,
                stop_strings=stop_strings  # Provided by NNsight
            )

            # Decode
            responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            for i in range(len(prompts)):
                local_results.append({
                    "prompt": prompts[i],
                    "full_generated_text": responses[i],
                    "generation_start_idx": generation_start_indices[i],
                    "correct_label": correct_labels[i],
                    "subject": subjects[i]
                })

    return local_results

########################################
# 4. Main Function
########################################

def main(rank, world_size):
    """
    Main function to run on each GPU/process.
    """
    setup_distributed(world_size, rank)

    # ==============================
    # DATA LOADING
    # ==============================
    subjects = [
        "high_school_biology",
        "high_school_chemistry",
        "high_school_computer_science",
        "high_school_european_history",
        "high_school_geography",
        "high_school_government_and_politics",
        "high_school_macroeconomics",
        "high_school_mathematics",
        "high_school_microeconomics",
        "high_school_physics",
        "high_school_psychology",
        "high_school_statistics",
        "high_school_us_history",
        "high_school_world_history",
    ]

    from datasets import load_dataset, concatenate_datasets
    datasets = []
    for subject in subjects:
        ds = load_dataset("cais/mmlu", subject)["test"]
        ds = ds.map(lambda x: {"subject": subject})
        datasets.append(ds)
    combined_dataset = concatenate_datasets(datasets)

    if rank == 0:
        print(f"Total dataset size: {len(combined_dataset)}")

    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"

    # Filter questions
    filtered_high_school_ds = filter_questions(combined_dataset, tokenizer, k=300, max_length=1000)
    if rank == 0:
        print(f"Filtered dataset size: {len(filtered_high_school_ds)}")

    gen_ds = GenerativeQuestionDataset(filtered_high_school_ds, tokenizer, max_length=2000)

    # Use a DistributedSampler so each rank gets a distinct subset
    sampler = DistributedSampler(
        gen_ds,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )

    gen_dataloader = DataLoader(
        gen_ds,
        batch_size=8,
        collate_fn=collate_fn,
        sampler=sampler
    )

    # ==============================
    # MODEL LOADING
    # ==============================
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).cuda(rank)
    model = NNsight(model)  # Wrap with NNsight

    # Convert to DDP
    native_model = model.model  # The actual torch.nn.Module inside NNsight
    ddp_model = DDP(native_model, device_ids=[rank], output_device=rank)
    model.model = ddp_model  # Reassign so model.generate() calls ddp_model

    # ==============================
    # RUN GENERATION
    # ==============================
    stop_strings = [r"\boxed{A}", r"\boxed{B}", r"\boxed{C}", r"\boxed{D}"]
    local_results = generate_answers(
        model, gen_dataloader, tokenizer,
        max_gen_length=2000,
        stop_strings=stop_strings
    )

    # ==============================
    # GATHER RESULTS ON RANK 0
    # ==============================
    gathered_results = [None] * world_size
    dist.all_gather_object(gathered_results, local_results)

    if rank == 0:
        final_results = []
        for r in gathered_results:
            final_results.extend(r)

        with open("high_school_gens_large.pkl", "wb") as f:
            pickle.dump(final_results, f)
        print(f"[Rank {rank}] Wrote combined results to high_school_gens_large.pkl")

    cleanup_distributed()

########################################
# 5. Entry Point
########################################
if __name__ == "__main__":
    # Instead of mp.spawn, we rely on torchrun's env vars:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    main(rank, world_size)
