import torch

def decode_summary_ids(sp, ids, sos_idx, eos_idx, pad_idx):
    """
    Takes a 1D list/tensor of token IDs, removes special tokens,
    and decodes to text using SentencePiece.
    """
    if torch.is_tensor(ids):
        ids = ids.tolist()

    cleaned = []
    for i in ids:
        if i == pad_idx:
            continue
        if i == sos_idx:
            continue
        if i == eos_idx:
            break
        cleaned.append(i)

    # SentencePiece can decode IDs directly
    return sp.decode_ids(cleaned)

def inspect_model_epoch(
    model,
    loader,
    sp,
    PAD_IDX,
    SOS_IDX,
    EOS_IDX,
    MAX_LEN,
    DEVICE,
):
    """
    Run a small qualitative inspection at the end of an epoch:
    - Print one true/pred intent
    - Print one source article (truncated) and its generated summary
    """
    model.eval()
    with torch.no_grad():
        src_batch, tgt_batch = next(iter(loader))
        src_batch = src_batch.to(DEVICE)
        tgt_batch = tgt_batch.to(DEVICE)

        src_example = src_batch[0:1]  # [1, seq_len]
        src_ids = src_example[0].cpu().tolist()
        src_text = decode_summary_ids(
            sp, src_ids, SOS_IDX, EOS_IDX, PAD_IDX
        )

        print("\n[FLAN Inspection]")
        print("  SOURCE (truncated):")
        print(" ", src_text[:400], "...\n")

        # Greedy decoding
        generated = [SOS_IDX]
        for _ in range(MAX_LEN):
            tgt_in = torch.tensor(
                [generated], dtype=torch.long, device=DEVICE
            )

            tgt_len = tgt_in.size(1)
            tgt_mask = torch.triu(
                torch.full((tgt_len, tgt_len), float("-inf"), device=DEVICE),
                diagonal=1,
            )

            logits_step = model.summarizer(
                src_example,
                tgt_in,
                src_key_padding_mask=(src_example == PAD_IDX),
                tgt_key_padding_mask=(tgt_in == PAD_IDX),
                tgt_mask=tgt_mask,
            )
            next_id = int(torch.argmax(logits_step[0, -1, :]).item())
            generated.append(next_id)
            if next_id == EOS_IDX:
                break

        gen_text = decode_summary_ids(
            sp, generated, SOS_IDX, EOS_IDX, PAD_IDX
        )
        print("  GENERATED:")
        print(" ", gen_text)
        print("-" * 80)

    model.train()