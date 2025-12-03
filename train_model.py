import pandas as pd
import json
import os
import torch
import numpy as np
import evaluate
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from transformers import (
    MBartForConditionalGeneration, 
    MBart50TokenizerFast,
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)
from nltk.translate.gleu_score import corpus_gleu

# --- FIX untuk Windows Environment ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- KONFIGURASI FINAL ---
# Pastikan file saltik.json ada di folder ini atau sesuaikan path-nya
DATASET_FILE = "dataset/saltik.json" 
MODEL_CHECKPOINT = "indobenchmark/indobart-v2"
OUTPUT_DIR = "./indobart-correction-saltik"

# Settingan KUAT RTX 3050 (Tetap dipertahankan)
BATCH_SIZE = 8            
GRAD_ACCUMULATION = 8    
EPOCHS = 5                

# Load Metric
try:
    bleu_metric = evaluate.load("sacrebleu")
except Exception as e:
    print(f"Warning: Gagal load sacrebleu. {e}")

def prepare_data():
    print("--- 1. Membaca dan Memproses Data JSON (Saltik) ---")
    
    if not os.path.exists(DATASET_FILE):
        raise FileNotFoundError(f"File dataset tidak ditemukan di: {DATASET_FILE}")

    # Membaca File JSON
    with open(DATASET_FILE, 'r', encoding='utf-8') as f:
        try:
            raw_data = json.load(f)
        except json.JSONDecodeError:
            raise ValueError("Format file JSON tidak valid.")

    # Flattening JSON ke List
    # Struktur JSON: {"kata_benar": [{"typo": "kata_salah", ...}, ...]}
    data_pairs = []
    
    print("Sedang mengekstrak pasangan kata (Typo -> Benar)...")
    for correct_word, typo_list in raw_data.items():
        for item in typo_list:
            typo_word = item.get('typo')
            if typo_word:
                # Format: [Input (Salah), Target (Benar)]
                data_pairs.append([typo_word, correct_word])

    if not data_pairs:
        raise ValueError("Gagal mengekstrak data dari JSON.")

    # Konversi ke DataFrame
    df_clean = pd.DataFrame(data_pairs, columns=['input_text', 'target_text'])
    
    # Pastikan tipe data string
    df_clean['input_text'] = df_clean['input_text'].astype(str)
    df_clean['target_text'] = df_clean['target_text'].astype(str)

    # ==========================================
    # FULL DATASET
    # ==========================================
    print(f"Total Pasangan Kata Siap Training: {len(df_clean)}")
    # ==========================================
    
    # Split Data (Validasi 5%)
    train_df, val_df = train_test_split(df_clean, test_size=0.05, random_state=42)
    
    return DatasetDict({
        'train': Dataset.from_pandas(train_df),
        'validation': Dataset.from_pandas(val_df)
    })

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Menggunakan Device: {device.upper()} ---")
    
    dataset = prepare_data()
    
    print("--- 2. Load Model & Tokenizer ---")
    tokenizer = MBart50TokenizerFast.from_pretrained(MODEL_CHECKPOINT, src_lang="id_ID", tgt_lang="id_ID")
    model = MBartForConditionalGeneration.from_pretrained(MODEL_CHECKPOINT)
    
    # Resize Embeddings
    print(f"Ukuran Vocab Awal: {model.config.vocab_size}")
    model.resize_token_embeddings(len(tokenizer))
    print(f"Ukuran Vocab Baru: {model.config.vocab_size}")
    model.to(device)

    # Optimasi Panjang Kalimat
    # Karena ini level KATA (bukan kalimat panjang), 64 sudah sangat cukup & lebih hemat memori
    # Namun 128 tetap aman jika ingin konsisten.
    max_len = 64 
    
    def preprocess_function(examples):
        inputs = examples["input_text"]
        targets = examples["target_text"]
        model_inputs = tokenizer(inputs, max_length=max_len, truncation=True)
        labels = tokenizer(text_target=targets, max_length=max_len, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_datasets = dataset.map(preprocess_function, batched=True)

    # --- FUNGSI METRIK ---
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]
        
        # BLEU
        decoded_labels_bleu = [[l] for l in decoded_labels]
        bleu_score = bleu_metric.compute(predictions=decoded_preds, references=decoded_labels_bleu)
        
        # GLEU
        ref_tokens = [[tokenizer.tokenize(l)] for l in decoded_labels]
        pred_tokens = [tokenizer.tokenize(p) for p in decoded_preds]
        gleu_score = corpus_gleu(ref_tokens, pred_tokens)
        
        return {
            "bleu": bleu_score["score"],
            "gleu": gleu_score * 100 
        }

    # Training Arguments
    args = Seq2SeqTrainingArguments(
        output_dir="checkpoints_saltik",
        
        eval_strategy="epoch", 
        save_strategy="epoch",
        
        learning_rate=1e-4,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION, 
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=EPOCHS,
        predict_with_generate=True, 
        fp16=True if device == "cuda" else False, 
        logging_dir='./logs',
        logging_steps=100,
        dataloader_num_workers=0,
        generation_max_length=64 
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics 
    )

    print(f"--- 3. Mulai Training (Dataset Saltik | Batch {BATCH_SIZE}) ---")
    trainer.train()

    print("--- Menghitung Metrik Akhir ---")
    metrics = trainer.evaluate()
    print(metrics)

    print("--- 4. Menyimpan Model Final ---")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model tersimpan di folder: {OUTPUT_DIR}")

    # Quick Test (Disesuaikan dengan contoh Saltik)
    print("\n--- 5. QUICK TEST ---")
    # Contoh typo dari dataset saltik (misal 'tyng' harusnya 'yang')
    test_words = ["tyng", "mknan", "Sya", "kmren"] 
    
    model.eval() 
    print(f"{'Input':<15} -> {'Output Model':<15}")
    print("-" * 35)
    
    for word in test_words:
        inputs = tokenizer(word, return_tensors="pt", max_length=64, truncation=True).to(device)
        with torch.no_grad():
            outputs = model.generate(inputs["input_ids"], max_length=64, num_beams=4, early_stopping=True)
        corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"{word:<15} -> {corrected_text:<15}")

if __name__ == "__main__":
    main()