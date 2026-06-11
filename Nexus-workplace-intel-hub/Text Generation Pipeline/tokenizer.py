import os
import sentencepiece as spm

SPM_MODEL_PREFIX = "spm_mtl"
TRAIN_TEXT_PATH = "spm_training_corpus.txt"
VOCAB_SIZE = 8000  

def build_training_corpus(src_text, tgt_texts):
    """
    Build a plain text file for training SentencePiece, if it doesn't exist yet.
    """
    if not os.path.exists(TRAIN_TEXT_PATH):
        print("Building SentencePiece training corpus from FLAN data...")
        with open(TRAIN_TEXT_PATH, "w", encoding="utf-8") as f:
            for s in src_text:
                f.write(str(s).strip() + "\n")
            for t in tgt_texts:
                f.write(str(t).strip() + "\n")


def train_sentencepiece():
    """
    Train SentencePiece model if files don't already exist.
    """
    model_path = SPM_MODEL_PREFIX + ".model"
    vocab_path = SPM_MODEL_PREFIX + ".vocab"

    if not (os.path.exists(model_path) and os.path.exists(vocab_path)):
        print("Training SentencePiece model...")
        spm.SentencePieceTrainer.Train(
            input=TRAIN_TEXT_PATH,
            model_prefix=SPM_MODEL_PREFIX,
            vocab_size=VOCAB_SIZE,
            model_type="bpe",
            character_coverage=0.9995,
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
        )
        print("SentencePiece model trained.")
    else:
        print("SentencePiece model already exists. Skipping training.")


def load_tokenizer():
    """
    Load the trained SentencePiece tokenizer and return:
      - sp: the SentencePieceProcessor
      - vocab_size
      - special token IDs: PAD_IDX, UNK_IDX, SOS_IDX, EOS_IDX
    """
    sp = spm.SentencePieceProcessor()
    sp.load(SPM_MODEL_PREFIX + ".model")

    vocab_size = sp.get_piece_size()
    PAD_IDX = sp.pad_id()
    UNK_IDX = sp.unk_id()
    SOS_IDX = sp.bos_id()
    EOS_IDX = sp.eos_id()

    return sp, vocab_size, PAD_IDX, UNK_IDX, SOS_IDX, EOS_IDX


def pad_sequence(seq, max_len, pad_idx):
    if len(seq) >= max_len:
        return seq[:max_len]
    return seq + [pad_idx] * (max_len - len(seq))


def tokenize_text(sp, sentence):
    """
    Tokenize a sentence with the given SentencePieceProcessor and return IDs.
    """
    return sp.encode(sentence, out_type=int)