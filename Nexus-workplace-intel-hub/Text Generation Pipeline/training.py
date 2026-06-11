import os
import math
import json
import random
from tqdm import tqdm

import torch 
import torch.nn as nn 
import torch.optim as optim 
from torch.utils.data import Dataset, DataLoader, random_split

from datasets import load_dataset
 
from model import MultiTaskTransformer
from tokenizer import (
    build_training_corpus,
    train_sentencepiece,
    load_tokenizer,
    pad_sequence,
    tokenize_text,
    VOCAB_SIZE,
)
from postprocessing import (
    decode_summary_ids,
    inspect_model_epoch,
)

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
RANDOM_SEED = 42  
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

MAX_LEN = 64            # Max length for encoder/decoder
EMBED_DIM = 256
D_MODEL = EMBED_DIM
NHEAD = 8               # Number of attention heads in the Transformer 
NUM_ENCODER_LAYER = 4
NUM_DECODER_LAYERS = 4

BATCH_SIZE = 4
EPOCHS = 100
LR = 3e-4               # Learning rate for the optimizer (0.0005)
WEIGHT_DECAY = 1e-2
GRAD_ACCUM_STEPS = 1    # effective batch size ~= BATCH_SIZE * GRAD_ACCUM_STEPS
PRINT_EVERY = 100

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------
# LOAD DATASET
# ----------------------------------------------------------------------- 
print("Loading Dolly dataset (this may take some time)...")

ds = load_dataset("databricks/databricks-dolly-15k", split="train")  # subset for testing
ds_qa = ds.filter(lambda ex: ex["category"] == "closed_qa")

max_examples = 32
if len(ds_qa) < max_examples:
    raise ValueError(f"Not enough QA examples in dataset: found {len(ds_qa)}")

dolly = ds_qa.select(range(max_examples))

instructions = dolly["instruction"]
contexts = dolly["context"]
responses = dolly["response"]

def build_source_text(inst, ctx):
    inst = (inst or "").strip()
    ctx = (ctx or "").strip()
    if ctx:
        return f"Instruction: {inst}\nContext: {ctx}"
    else:
        return f"Instruction: {inst}"
    
src_texts = [build_source_text(i, x) for i, x in zip(instructions, contexts)]
tgt_texts = responses

# --------------------------------------------------------------------------------------
# SENTENCEPIECE TOKENIZER SETUP
# --------------------------------------------------------------------------------------
build_training_corpus(src_texts, tgt_texts)           # Build corpus file for SentencePiece

train_sentencepiece()        # Train SentencePiece model
sp, vocab_size, PAD_IDX, UNK_IDX, SOS_IDX, EOS_IDX = load_tokenizer()  # Load tokenizer + special token IDs

print("SentencePiece vocab size:", vocab_size)

# -------------------------------------------------------------------------
# ENCODE DATA 
# -------------------------------------------------------------------------
def encode_and_pad_src(text_list, max_len=MAX_LEN):
    encoded = []
    for t in text_list:
        ids = tokenize_text(sp, t)        # list of int token IDs
        ids = ids[: max_len - 2]               # truncate
        encoded.append(ids)
    return encoded

src_ids_list = encode_and_pad_src(src_texts, MAX_LEN)
src_ids_list = [pad_sequence(ids, MAX_LEN, PAD_IDX) for ids in src_ids_list]

tgt_ids_list = []
for t in tgt_texts:
    ids = tokenize_text(sp, t)
    ids = ids[: MAX_LEN - 2]
    seq = [SOS_IDX] + ids + [EOS_IDX]
    seq = pad_sequence(seq, MAX_LEN, PAD_IDX)
    tgt_ids_list.append(seq)
# ----------------------------------------------------------------------------------------
# DATASET / DATALOADER
# ----------------------------------------------------------------------------------------
class Seq2SeqDataset(Dataset):
    def __init__(self, src, tgt):
        assert len(src) == len(tgt)
        self.src = torch.tensor(src, dtype=torch.long)
        self.tgt = torch.tensor(tgt, dtype=torch.long)

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return self.src[idx], self.tgt[idx]

dataset = Seq2SeqDataset(src_ids_list, tgt_ids_list)

train_size = 30
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(RANDOM_SEED),
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

# ----------------------------------------------------------------------------------------------
# INITIALIZING MODEL, LOSSES, OPTIMIZER
# ----------------------------------------------------------------------------------------------
model = MultiTaskTransformer(vocab_size=vocab_size, d_model=D_MODEL, 
                             nhead=NHEAD, num_encoder_layer=NUM_ENCODER_LAYER,
                             num_decoder_layer=NUM_DECODER_LAYERS, pad_idx=PAD_IDX).to(DEVICE)

optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

num_training_steps = (len(train_loader) * EPOCHS) // GRAD_ACCUM_STEPS
warmup_steps = int(0.03 * num_training_steps)

def lr_lambda(current_step):
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    return max(
        0.0,
        float(num_training_steps - current_step)
        / float(max(1, num_training_steps - warmup_steps)),
    )

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

# ------------------------------------------------------------------------------------------------
# TRAIN & EVAL
# ------------------------------------------------------------------------------------------------
def train_one_epoch(epoch_idx):
    model.train()
    total_loss = 0.0
    step_count = 0
    optimizer.zero_grad()

    for step, (src_batch, tgt_batch) in enumerate(
        tqdm(train_loader, desc=f"Epoch {epoch_idx+1}/{EPOCHS}")
    ):
        src_batch = src_batch.to(DEVICE)
        tgt_batch = tgt_batch.to(DEVICE)

        tgt_in = tgt_batch[:, :-1].contiguous()
        tgt_out = tgt_batch[:, 1:].contiguous()

        tgt_len = tgt_in.size(1)
        tgt_mask = torch.triu(
            torch.full((tgt_len, tgt_len), float("-inf"), device=DEVICE),
            diagonal=1,
        )

        with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda")):
            logits = model.summarizer(
                src_batch,
                tgt_in,
                src_key_padding_mask=(src_batch == PAD_IDX),
                tgt_key_padding_mask=(tgt_in == PAD_IDX),
                tgt_mask=tgt_mask,
            )
            loss = loss_fn(logits.view(-1, vocab_size), tgt_out.view(-1))
            loss = loss / GRAD_ACCUM_STEPS

        scaler.scale(loss).backward()
        total_loss += loss.item() * GRAD_ACCUM_STEPS
        step_count += 1

        if (step + 1) % GRAD_ACCUM_STEPS == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        if (step + 1) % PRINT_EVERY == 0:
            avg_loss_so_far = total_loss / step_count
            current_lr = scheduler.get_last_lr()[0]
            print(
                f"  Step {step+1}/{len(train_loader)} "
                f"- Loss: {avg_loss_so_far:.4f} - LR: {current_lr:.6f}"
            )

    avg_loss = total_loss / max(1, step_count)
    return avg_loss

def evaluate():
    model.eval()
    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for src_batch, tgt_batch in val_loader:
            src_batch = src_batch.to(DEVICE)
            tgt_batch = tgt_batch.to(DEVICE)

            tgt_in = tgt_batch[:, :-1].contiguous()
            tgt_out = tgt_batch[:, 1:].contiguous()

            tgt_len = tgt_in.size(1)
            tgt_mask = torch.triu(
                torch.full((tgt_len, tgt_len), float("-inf"), device=DEVICE),
                diagonal=1,
            ).to(DEVICE)

            with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda")):
                logits = model.summarizer(
                    src_batch,
                    tgt_in,
                    src_key_padding_mask=(src_batch == PAD_IDX),
                    tgt_key_padding_mask=(tgt_in == PAD_IDX),
                    tgt_mask=tgt_mask,
                )
                loss = loss_fn(logits.view(-1, vocab_size), tgt_out.view(-1))

            total_loss += loss.item()
            count += 1

    avg_loss = total_loss / max(1, count)
    return avg_loss

# ------------------------------------------------------------------------------------------------
# MAIN TRAINING LOOP
# ------------------------------------------------------------------------------------------------
print("\nStarting traing on device...\n")
best_val_loss = float("inf")

for epoch in range(EPOCHS):
    print(f"\n========== EPOCH {epoch+1}/{EPOCHS} ==========")
    train_loss = train_one_epoch(epoch)
    val_loss = evaluate()
    print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

    # ------------------------ DEBUG AFTER EPOCH ------------------------
    model.eval()
    with torch.no_grad():
        # Grab one batch from *train_loader* or *val_loader*
        src_batch, tgt_batch = next(iter(train_loader))
        src_batch = src_batch.to(DEVICE)
        tgt_batch = tgt_batch.to(DEVICE)

        tgt_in = tgt_batch[:, :-1].contiguous()
        tgt_out = tgt_batch[:, 1:].contiguous()

        # Just the first example
        src = src_batch[0:1]
        tgt = tgt_in[0:1]
        tgt_gold = tgt_out[0:1]

        tgt_len = tgt.size(1)
        tgt_mask = torch.triu(
            torch.full((tgt_len, tgt_len), float("-inf"), device=DEVICE),
            diagonal=1,
        ).to(DEVICE)

        logits = model.summarizer(
            src,
            tgt,
            src_key_padding_mask=(src == PAD_IDX),
            tgt_key_padding_mask=(tgt == PAD_IDX),
            tgt_mask=tgt_mask,
        )

        pred_ids = torch.argmax(logits, dim=-1)[0].cpu().tolist()
        gold_ids = tgt_gold[0].cpu().tolist()

        print("\n[DEBUG: Teacher-forced inspection on 1 train example]")
        print("Pred IDs: ", pred_ids[:40])
        print("Gold IDs: ", gold_ids[:40])
        print("Pred pieces:", [sp.id_to_piece(i) for i in pred_ids[:40]])
        print("Gold pieces:", [sp.id_to_piece(i) for i in gold_ids[:40]])
        print("-" * 60)
    # -------------------------------------------------------------------

    if val_loss < best_val_loss:
        best_val_loss = val_loss

# --------------------------------------------------------------------------------------------------
# SAVE ARTIFACTS
# --------------------------------------------------------------------------------------------------
os.makedirs("artifacts", exist_ok=True)
torch.save(model.state_dict(), "artifacts/model.pth")

import shutil
shutil.copy("spm_mtl.model", "artifacts/spm_mtl.model")
shutil.copy("spm_mtl.vocab", "artifacts/spm_mtl.vocab")

print("\n Model training complete and saved successfully!")