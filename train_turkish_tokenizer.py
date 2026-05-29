from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from datasets import load_dataset
import time

def train_turkish_tokenizer():
    print("Tokenizer modeli başlatılıyor (BPE Byte-Level)...")
    
    # 1. Tokenizer'ı yapılandır (BPE + Byte-Level)
    # Byte-Level, nadir karakterleri byte'lara böldüğü için <unk> (bilinmeyen kelime) sorununu ortadan kaldırır.
    tokenizer = Tokenizer(BPE(unk_token=None))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False, use_regex=True)
    tokenizer.decoder = ByteLevelDecoder()

    # İstenen kelime dağarcığı: 128.000
    vocab_size = 128000
    
    # Standart LLM özel tokenleri (DeepSeek / Llama benzeri)
    special_tokens = [
        "<|begin_of_sentence|>", 
        "<|end_of_sentence|>", 
        "<|pad|>",
        "<|unk|>"
    ]
    
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=True
    )

    print("\nHugging Face üzerinden Türkçe Wikipedia veri seti (streaming olarak) yükleniyor...")
    # Sadece 16GB RAM'iniz olduğu için streaming (akış) modunda yüklüyoruz.
    # Böylece veriler RAM'i doldurmadan parça parça işleniyor.
    dataset = load_dataset("wikimedia/wikipedia", "20231101.tr", split="train", streaming=True)

    def batch_iterator(batch_size=2000, max_docs=150000):
        batch = []
        print(f"Toplam {max_docs} civarında kaliteli makale taranacak...")
        for i, doc in enumerate(dataset):
            if i >= max_docs:
                break
            batch.append(doc["text"])
            
            # Belirlenen boyuta ulaşıldığında tokenizer'a gönder
            if len(batch) == batch_size:
                yield batch
                batch = []
                
        # Geriye kalan son partiyi de gönder
        if batch:
            yield batch

    start_time = time.time()
    print("\nEğitim Başladı! İşlemcinizin tüm çekirdekleri şu an çalışıyor.")
    print("Lütfen bekleyin (Sisteminizin i5-12450H gücüne bağlı olarak yaklaşık tahmini 10-20 dakika sürecektir)...")
    
    # Eğitimi Başlat
    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)
    
    end_time = time.time()
    print(f"\nEğitim başarıyla tamamlandı! Toplam süre: {round((end_time - start_time)/60, 2)} dakika.")

    # Kaydet
    output_path = "turkish_tokenizer.json"
    tokenizer.save(output_path)
    print(f"Tokenizer başarıyla kaydedildi: {output_path}")

if __name__ == "__main__":
    train_turkish_tokenizer()
