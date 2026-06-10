import torch  
import os 
import sentencepiece as spm

from model import MultiTaskTransformer
from postprocessing import decode_summary_ids

ARTIFACT_DIR = "artifacts"
MODEL_PATH   = os.path.join(ARTIFACT_DIR, "model.pth")
SPM_MODEL    = os.path.join(ARTIFACT_DIR, "spm_mtl.model")

MAX_LEN = 80        
EMBED_DIM = 256
D_MODEL = EMBED_DIM
NHEAD = 8
NUM_ENCODER_LAYER = 4
NUM_DECODER_LAYERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load Tokenizer ---
if not os.path.exists(SPM_MODEL):
    raise FileNotFoundError(f"SentencePiece model not found at {SPM_MODEL}")

sp = spm.SentencePieceProcessor()
sp.load(SPM_MODEL)

PAD_IDX = sp.pad_id()
UNK_IDX = sp.unk_id()
SOS_IDX = sp.bos_id()
EOS_IDX = sp.eos_id()

vocab_size = sp.get_piece_size()

# --- Load Model ---
model = MultiTaskTransformer(vocab_size=vocab_size, d_model=EMBED_DIM,
                             nhead=NHEAD, num_encoder_layer=NUM_ENCODER_LAYER, num_decoder_layer=NUM_DECODER_LAYERS,
                             pad_idx=PAD_IDX).to(DEVICE)

state = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state, strict=False)
model.eval()

# --- Helper Functions ---
def pad_sequence(ids, max_len, pad_idx):
    if len(ids) >= max_len:
        return ids[:max_len]
    return ids + [pad_idx] * (max_len - len(ids))

def encode_text(text: str, add_special: bool = False):
    """
    Encode text to IDs using SentencePiece.
    """
    ids = sp.encode(text, out_type=int)

    # If add_special=True, we wrap with BOS/SOS and EOS.
    if add_special:
        ids = [SOS_IDX] + ids + [EOS_IDX]   
    return ids

def decode_ids(ids):
    """
    Decode a list of IDs back to text,
    stopping at EOS and skipping PAD/BOS/EOS appropriately.
    """
    cleaned = []
    for i in ids:
        if i == PAD_IDX:
            continue
        if i == SOS_IDX:
            continue
        if i == EOS_IDX:
            break
        cleaned.append(i)
    return sp.decode_ids(cleaned)

# --- Main Generation Loop ---
@torch.no_grad()
def greedy_generate(
    instruction: str,
    context: str | None = None,
    max_new_tokens: int = 80,
) -> str:
    """
    Main generation function.
    """

    # Build the same style of source text trained on.
    instruction = (instruction or "").strip()
    context = (context or "").strip() if context is not None else ""

    if context:
        src_text = f"Instruction: {instruction}\nContext: {context}"
    else:
        src_text = f"Instruction: {instruction}"

    # Encode source, pad to MAX_LEN
    src_ids = sp.encode(src_text, out_type=int)
    src_ids = src_ids[:MAX_LEN]
    src_ids = pad_sequence(src_ids, MAX_LEN, PAD_IDX)

    src_tensor = torch.tensor([src_ids], dtype=torch.long, device=DEVICE)
    src_key_padding_mask = (src_tensor == PAD_IDX)

    # Run encoder to get memory
    memory = model.forward_encoder(src_tensor, src_key_padding_mask=src_key_padding_mask)

    # Start target sequence with SOS
    generated = [SOS_IDX]

    for _ in range(max_new_tokens):
        tgt_in = torch.tensor([generated], dtype=torch.long, device=DEVICE)

        # Build causal mask for decoder
        tgt_len = tgt_in.size(1)
        tgt_mask = torch.triu(
            torch.full((tgt_len, tgt_len), float("-inf"), device=DEVICE),
            diagonal=1,
        )

        tgt_key_padding_mask = (tgt_in == PAD_IDX)

        # Run one step of the decoder
        logits = model.summarizer(
            src_ids=src_tensor,
            tgt_ids=tgt_in,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            tgt_mask=tgt_mask,
        )  # [1, tgt_len, vocab_size]

        # Take the last time step
        next_token_logits = logits[0, -1, :]
        next_id = int(torch.argmax(next_token_logits).item())

        generated.append(next_id)

        if next_id == EOS_IDX:
            break

    print("Generated IDs:", generated)
    print("Generated pieces:", [sp.id_to_piece(i) for i in generated])
    # Decode generated IDs to text
    decoded = decode_summary_ids(
        sp,
        generated,
        sos_idx=SOS_IDX,
        eos_idx=EOS_IDX,
        pad_idx=PAD_IDX,
    )
    return decoded

if __name__ == "__main__":
    while True:
        print("\n---")
        instruction = input("Instruction (or 'exit'): ").strip()
        if instruction.lower() == "exit":
            break

        use_ctx = input("Add context? (y/n): ").strip().lower()
        if use_ctx == "y":
            context = input("Context: ")
        else:
            context = None

        output = greedy_generate(instruction, context=context, max_new_tokens=80)
        print("\nModel output:")
        print(output)