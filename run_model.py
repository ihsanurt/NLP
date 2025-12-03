import json
import random
import os
import torch
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

# --- KONFIGURASI ---
DATASET_FILE = "dataset/saltik.json"       # Dataset untuk bahan generator
MODEL_PATH = "./-correction-sindobartaltik" # Model hasil training

# ==========================================
# BAGIAN 1: FUNGSI DATASET (GENERATOR)
# ==========================================
def load_corpus():
    """
    Membaca dataset JSON dan mengubahnya menjadi list pasangan untuk generator.
    """
    print(f"--- Membaca Dataset dari {DATASET_FILE} ... ---")
    
    if not os.path.exists(DATASET_FILE):
        print(f"Error: File {DATASET_FILE} tidak ditemukan.")
        return []

    with open(DATASET_FILE, 'r', encoding='utf-8') as f:
        try:
            raw_data = json.load(f)
        except json.JSONDecodeError:
            print("Error: Format JSON tidak valid.")
            return []

    corpus_pairs = []
    
    # Flattening data
    for correct_word, typo_list in raw_data.items():
        for item in typo_list:
            typo_word = item.get('typo')
            if typo_word:
                corpus_pairs.append((typo_word, correct_word))
    
    print(f"--- Dataset dimuat! Total variasi typo tersedia: {len(corpus_pairs)} kata. ---")
    return corpus_pairs

def generate_sentence_pair(corpus, num_words):
    """
    Memilih kata acak dan membuat kalimat typo beserta kunci jawabannya.
    """
    if not corpus:
        return "", ""
        
    selected_pairs = [random.choice(corpus) for _ in range(num_words)]
    
    typo_words = [p[0] for p in selected_pairs]
    correct_words = [p[1] for p in selected_pairs]
    
    return " ".join(typo_words), " ".join(correct_words)

# ==========================================
# BAGIAN 2: FUNGSI MODEL (KOREKTOR)
# ==========================================
def load_model():
    print(f"--- Memuat Model AI dari {MODEL_PATH} ... ---")
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Folder model {MODEL_PATH} tidak ditemukan. Pastikan training sudah selesai.")
        exit()

    try:
        tokenizer = MBart50TokenizerFast.from_pretrained(MODEL_PATH)
        model = MBartForConditionalGeneration.from_pretrained(MODEL_PATH)
    except Exception as e:
        print(f"Error saat load model: {e}")
        exit()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    
    print(f"--- Model Siap! (Running on {device.upper()}) ---")
    return model, tokenizer, device

def correct_sentence(text, model, tokenizer, device):
    """
    Memperbaiki kalimat dengan memecahnya per kata (Batch Processing).
    """
    words = text.split()
    if not words: return ""

    # Tokenisasi batch
    inputs = tokenizer(words, return_tensors="pt", padding=True, truncation=True, max_length=64).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_length=64,
            num_beams=5, 
            early_stopping=True,
            no_repeat_ngram_size=2
        )
        
    corrected_words = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return " ".join(corrected_words)

# ==========================================
# MAIN MENU
# ==========================================
def main():
    # 1. Load Resource
    corpus = load_corpus()
    model, tokenizer, device = load_model()

    while True:
        print("\n" + "="*50)
        print("   APLIKASI PENGUJIAN MODEL TYPO CORRECTION")
        print("="*50)
        print("1. Generate Kalimat Random (Cek performa model)")
        print("2. Input Manual (Ketik kalimat sendiri)")
        print("3. Keluar")
        print("-" * 50)
        
        pilihan = input("Pilih menu (1/2/3): ").strip()

        if pilihan == '1':
            # --- MODE GENERATOR ---
            if not corpus:
                print("Dataset kosong, tidak bisa generate.")
                continue
                
            raw_num = input("Masukkan jumlah kata per kalimat (contoh: 5): ")
            if not raw_num.isdigit():
                print("Input harus angka.")
                continue
                
            num = int(raw_num)
            
            # Generate Soal
            soal_typo, kunci_jawaban = generate_sentence_pair(corpus, num)
            
            # Model Mengerjakan Soal
            jawaban_model = correct_sentence(soal_typo, model, tokenizer, device)
            
            print("\n--- HASIL PENGUJIAN ---")
            print(f"Generate Kalimat : {soal_typo}")
            print(f"Jawaban Model    : {jawaban_model}")

        elif pilihan == '2':
            # --- MODE MANUAL ---
            text = input("\nMasukkan kalimat typo: ")
            if not text.strip(): continue
            
            hasil = correct_sentence(text, model, tokenizer, device)
            
            print(f"Original : {text}")
            print(f"Koreksi  : {hasil}")

        elif pilihan == '3':
            print("Terima kasih. Sampai jumpa!")
            break
        
        else:
            print("Pilihan tidak valid.")

if __name__ == "__main__":
    main()