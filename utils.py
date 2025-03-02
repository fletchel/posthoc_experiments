import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
import re

def get_pre_post_cot_logits(datapoint, model, tokenizer):
    """
    Analyzes a model generation that ends with 'The answer is \\boxed{...}'
    and does NOT have 'The answer is \\boxed{<...>}' before the <think> token.
    
    Specifically, it does the following:
      1) Extracts the final answer from "The answer is \\boxed{...}" at the end.
      2) Gets the logits for the token that follows "The answer is \\boxed{" 
         in the already-generated text.
      3) Teacher-forces "The answer is \\boxed{<final_answer>}" immediately 
         before <think>, and returns the logits corresponding to those 
         forced tokens (the model's distributions when it 'predicts' each 
         token in that snippet).
    
    Returns a dictionary with:
        {
            "final_answer": str or None,
            "logits_after_boxed_prefix": torch.Tensor or None,
            "forced_snippet_logits": torch.Tensor or None
        }
    """

    generation_text = datapoint['full_generated_text']

    relevant_ids = tokenizer.convert_tokens_to_ids(['A', 'B', 'C', 'D'])

    # ----------------------------------------------------------------
    # 1) Extract the final answer from the generation
    # ----------------------------------------------------------------
    answer_match = re.search(r"The answer is\s*\\boxed\{([^}]+)\}", generation_text)
    final_answer = answer_match.group(1) if answer_match else None

    # ----------------------------------------------------------------
    # 2) Get logits AFTER "The answer is \\boxed{"
    #    We'll find the prefix in the generation and then run the model
    #    on everything up to and including that prefix, returning the
    #    next-token logits.
    # ----------------------------------------------------------------
    logits_after_boxed_prefix = None
    prefix_match = re.search(r"\\boxed\{[A-D]\}", generation_text)
    if prefix_match:
        prefix_end_idx = prefix_match.end()  # Index right after 'The answer is \boxed{'
        prefix_text = generation_text[:prefix_end_idx-2]
        prefix_ids = tokenizer(prefix_text, return_tensors="pt").input_ids.to('cuda')
        with torch.no_grad():
            prefix_out = model(prefix_ids)
        # The final logits row is for the position where the next token is predicted
        logits_after_boxed_prefix = prefix_out.logits[0, -1, relevant_ids]

    # ----------------------------------------------------------------
    # 3) Teacher-force "The answer is \\boxed{<final_answer>}" before <think>
    #    and capture the logits for precisely those forced tokens.
    # ----------------------------------------------------------------

    # Look for <think> in the generation
    think_match = re.search(r"<think>", generation_text)
    if think_match:
        start_think_idx = think_match.start()

        # We'll construct the forced snippet: "The answer is \boxed{FINAL_ANSWER}"
        forced_snippet = f"The answer is \\boxed{{"

        # Partition the text: everything before <think>, then forced snippet, then everything after <think>
        prefix_text = generation_text[:start_think_idx]
        suffix_text = generation_text[start_think_idx:]  # includes <think>

        # Tokenize prefix without special tokens
        prefix_ids = tokenizer(prefix_text, return_tensors="pt", add_special_tokens=False).input_ids
        # Tokenize forced snippet without special tokens
        forced_ids = tokenizer(forced_snippet, return_tensors="pt", add_special_tokens=False).input_ids
        # Combine
        combined_ids = torch.cat([prefix_ids, forced_ids], dim=1).to('cuda')

        # Run the model on just prefix+forced_snippet (not the entire original text)
        with torch.no_grad():
            out = model(combined_ids)
        
        # out.logits shape: [batch_size=1, seq_length, vocab_size]
        # The snippet tokens start at index offset = prefix_ids.size(1).
        # The snippet length = forced_ids.size(1).
        # Because logits[i] is the distribution that predicts the i-th token in the input
        # (shifted by 1 in typical causal LM fashion).
        # We'll extract the logits for the snippet portion:
        offset = prefix_ids.size(1)
        length = forced_ids.size(1)

        # The logits that "produced" the forced snippet tokens:
        forced_snippet_logits = out.logits[0, -1, relevant_ids]

        # Explanation:
        # - If offset=prefix_len, then to get the model's distribution
        #   for token offset, we look at out.logits[:, offset-1, :].
        # - So for snippet tokens from offset to offset+length-1,
        #   we slice logits from offset-1 to offset-1+length.

    return {
        "final_answer": final_answer,
        "pre_logits": forced_snippet_logits,
        "post_logits": logits_after_boxed_prefix,
        "subject":datapoint['subject']
    }


class TextClassificationDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        """
        Args:
            data (list of dicts): A list containing data entries in the provided format.
            tokenizer (transformers.PreTrainedTokenizer): A tokenizer for encoding text.
            max_length (int): Maximum length for tokenized input.
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Ensure the tokenizer uses left padding
        self.tokenizer.padding_side = "left"
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        entry = self.data[idx]
        generated_text = entry["full_generated_text"]
        correct_label = entry["correct_label"]
        predicted_label = entry["predicted_label"]
        answer_start_idx = entry["answer_token_index"]
        generation_start_idx = entry["generation_start_idx"]
        
        # Tokenize with left padding
        encoding = self.tokenizer(
            generated_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        pad_token_id = self.tokenizer.pad_token_id
        
        # Count how many leading pad tokens there are
        leading_pad_count = 0
        for token_id in input_ids:
            if token_id.item() == pad_token_id:
                leading_pad_count += 1
            else:
                break
        
        # The index at which padding ends and real tokens start
        padding_start_idx = leading_pad_count if leading_pad_count < self.max_length else self.max_length
        
        # Adjust indices to account for the left-padded offset
        adjusted_answer_start_idx = answer_start_idx + leading_pad_count
        adjusted_generation_start_idx = generation_start_idx + leading_pad_count

        # Convert labels to tensors
        correct_label_tensor = torch.tensor(correct_label, dtype=torch.long)
        predicted_label_tensor = torch.tensor(predicted_label, dtype=torch.long)
        answer_start_idx_tensor = torch.tensor(adjusted_answer_start_idx, dtype=torch.long)
        generation_start_idx_tensor = torch.tensor(adjusted_generation_start_idx, dtype=torch.long)
        padding_start_idx_tensor = torch.tensor(padding_start_idx, dtype=torch.long)

        # change below to change probe idx, hacky but works

        probe_idx_tensor = torch.tensor(padding_start_idx_tensor, dtype=torch.long)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "correct_label": correct_label_tensor,
            "predicted_label": predicted_label_tensor,
            "answer_start_idx": answer_start_idx_tensor,
            "generation_start_idx": generation_start_idx_tensor,
            "padding_start_idx": padding_start_idx_tensor,
            "probe_position": probe_idx_tensor
        }

import torch

def get_probe_features(model, dataloader, hidden_dim,
                      layer_indices=[4, 16, 27], 
                      max_batches=50):
    """
    Processes the dataset to extract features from specified transformer layers
    at specific positions:
      1) padding_start, padding_start+1, padding_start+2
      2) generation_start ± 2
      3) answer_start ± 2
      4) 20%, 40%, 60%, 80% through the generation span

    Parameters:
    - model: The transformer model
    - dataloader: The DataLoader for the dataset
    - layer_indices: List of transformer layer indices to extract residuals from
    - max_batches: Maximum number of batches to process

    Returns:
    - all_residuals: Tensor of shape (total_examples, num_positions, num_layers, hidden_dim)
    - all_labels: Tensor of corresponding labels
    - all_gen_positions: Tensor of generation start indices
    - all_ans_positions: Tensor of answer start indices
    - all_padding_positions: Tensor of padding start indices
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    all_residuals = []
    all_labels = []
    all_gen_positions = []
    all_ans_positions = []
    all_padding_positions = []
    
    for i, batch in enumerate(dataloader):
        print(f"Batch {i}")
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["predicted_label"]
        gen_index = batch["generation_start_idx"]
        answer_index = batch["answer_start_idx"]
        padding_index = batch["padding_start_idx"]

        batch_size, seq_len = batch['input_ids'].shape
        num_layers = len(layer_indices)

        with torch.no_grad(), model.trace(input_ids, attention_mask):
            # Collect the outputs from the specified layers
            layer_outputs = []
            for layer_idx in layer_indices:
                # shape [batch_size, seq_len, hidden_dim]
                layer_outputs.append(model.model.layers[layer_idx].output[0])
            
            # Stack -> [batch_size, seq_len, num_layers, hidden_dim]
            stacked_outputs = torch.stack(layer_outputs, dim=2)

            # We'll store the slice for each example, then cat them
            batch_resid_positions = []
            
            
            for b in range(batch_size):
                p_start = padding_index[b].item()
                g_start = gen_index[b].item()
                a_start = answer_index[b].item()

                # Ensure indices are within sequence bounds
                def clamp(x):
                    return max(0, min(x, seq_len - 1))

                # 1) Padding start, start+1, start+2
                padding_positions = [clamp(p_start + j) for j in range(3)]

                # 2) Gen start ± 2
                gen_positions = [clamp(g_start + j) for j in range(-2, 3)]

                # 3) Answer start ± 2
                ans_positions = [clamp(a_start + j) for j in range(-2, 3)]

                # 4) 20%, 40%, 60%, 80% through generation
                # generation goes [g_start, a_start) (assuming a_start > g_start)
                gen_length = max(0, a_start - g_start)
                # get positions only if the span is non-zero
                frac_positions = []
                if gen_length > 0:
                    for frac in [0.2, 0.4, 0.6, 0.8]:
                        offset = int(round(frac * gen_length))
                        frac_positions.append(clamp(g_start + offset))


                # Combine them all into one unique set (keep order with list->dict->list trick)
                positions = padding_positions + gen_positions + ans_positions + frac_positions
                # [seq_len, num_layers, hidden_dim]
                example_stacked = stacked_outputs[b]
                # Gather only the desired positions -> shape [num_positions, num_layers, hidden_dim]
                positions_tensor = torch.tensor(positions, dtype=torch.long, device=model.device)
                example_subset = example_stacked.index_select(0, positions_tensor)
                
                batch_resid_positions.append(example_subset.unsqueeze(0)) 
                #print(batch_resid_positions[-1].shape)
                
            # shape [batch_size, num_positions, num_layers, hidden_dim]
            batch_resid_positions = torch.cat(batch_resid_positions, dim=0).cpu().save()
        
        all_residuals.append(batch_resid_positions)
        all_labels.append(labels.cpu())
        all_gen_positions.append(gen_index.cpu())
        all_ans_positions.append(answer_index.cpu())
        all_padding_positions.append(padding_index.cpu())

        if i >= max_batches:
            break
    
    # Concatenate across all batches
    # all_residuals -> shape [total_examples, num_positions, num_layers, hidden_dim]
    all_residuals = torch.cat(all_residuals, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_gen_positions = torch.cat(all_gen_positions, dim=0)
    all_ans_positions = torch.cat(all_ans_positions, dim=0)
    all_padding_positions = torch.cat(all_padding_positions, dim=0)

    return (
        all_residuals, 
        all_labels, 
        all_gen_positions, 
        all_ans_positions, 
        all_padding_positions
    )

def train_linear_classifier_with_positions(
    train_residuals: torch.Tensor,
    train_positions: torch.Tensor,
    train_labels: torch.Tensor,
    test_residuals: torch.Tensor,
    test_positions: torch.Tensor,
    test_labels: torch.Tensor,
    layer_idx: int,
    num_classes: int,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
):
    """
    Trains an nn.Linear classifier on a specific layer from residual streams,
    using per-example positions for training/test.

    Args:
        train_residuals: 4D tensor, shape [train_size, seq_len, num_layers, hidden_dim]
        train_positions: 1D tensor, shape [train_size], specifying which position each train example should use
        train_labels: 1D tensor, shape [train_size]
        test_residuals: 4D tensor, shape [test_size, seq_len, num_layers, hidden_dim]
        test_positions: 1D tensor, shape [test_size], specifying which position each test example should use
        test_labels: 1D tensor, shape [test_size]
        layer_idx: integer index for which layer to train on (0-based within the stacked dimension)
        num_classes: number of label classes (for nn.Linear output dim)
        epochs: total epochs to train
        batch_size: batch size
        lr: learning rate
        device: device for training

    Returns:
        model: trained nn.Linear model
    """

    # ---- 1) Extract single layer & per-example position ----
    # train_positions[i] is the index of the token in sequence i
    # advanced indexing: [batch_index, position_index, layer_index, :]
    train_size = train_residuals.size(0)
    test_size = test_residuals.size(0)
    
    # This turns [train_size, seq_len, num_layers, hidden_dim] into [train_size, hidden_dim]
    # by picking the position from train_positions and the layer from layer_idx
    X_train = train_residuals[
        torch.arange(train_size),  # all example indices
        train_positions,           # per-example positions
        layer_idx,                 # single layer
        :
    ]
    y_train = train_labels  # shape [train_size]

    # Similarly for test set
    X_test = test_residuals[
        torch.arange(test_size),
        test_positions,
        layer_idx,
        :
    ]
    y_test = test_labels  # shape [test_size]

    # ---- 2) Build Datasets and Loaders ----
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # ---- 3) Define model, loss, and optimizer ----
    # Hidden dimension is the last dimension of X_train
    hidden_dim = X_train.size(-1)
    model = nn.Linear(hidden_dim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # ---- 4) Training loop ----
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

        # ---- (Optional) Evaluate on test set each epoch ----
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                logits = model(batch_X)
                preds = logits.argmax(dim=1)
                correct += (preds == batch_y).sum().item()
                total += batch_y.size(0)

        accuracy = correct / total if total > 0 else 0.0

    return model, accuracy

def balance_classes(res, labels):
    """
    Balances the dataset by randomly discarding excess samples from overrepresented labels.

    Parameters:
    - res: Tensor of residuals
    - labels: Tensor of labels
    - gen_pos: Tensor of generation start indices
    - ans_pos: Tensor of answer start indices
    - pad_end_pos: Tensor of padding end positions

    Returns:
    - Balanced tensors for residuals, labels, gen positions, ans positions, and padding positions
    """
    unique_labels, counts = torch.unique(labels, return_counts=True)
    min_count = counts.min().item()  # Find the smallest class count

    balanced_indices = []
    
    for label in unique_labels:
        # Get indices of all samples with this label
        label_indices = torch.where(labels == label)[0].tolist()
        
        # Randomly select min_count samples to balance the dataset
        selected_indices = np.random.choice(label_indices, min_count, replace=False)
        balanced_indices.extend(selected_indices)
    
    # Shuffle to avoid label ordering biases
    np.random.shuffle(balanced_indices)
    
    # Subset tensors to keep only the balanced indices
    balanced_res = res[balanced_indices]
    balanced_labels = labels[balanced_indices]


    return balanced_res, balanced_labels
