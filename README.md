---
language:
- tr
tags:
- tokenizer
- bpe
- turkce
- turkish
---

# Turkish BPE Tokenizer (128k Vocab)

Bu model, Türkçe metinler için sıfırdan eğitilmiş **Byte-Level BPE (Byte Pair Encoding)** tabanlı bir Tokenizer modelidir. 

## Özellikler
- **Vocabulary Size (Kelime Dağarcığı):** 128.000
- **Eğitim Verisi:** Hugging Face `wikimedia/wikipedia` (20231101.tr) - ~150.000 Türkçe Makale
- **Tokenizer Tipi:** BPE, Byte-Level Fallback özelliğine sahip (Bilinmeyen `<unk>` token sorunu yaşanmaz).
- **Özel Tokenler:** `<|begin_of_sentence|>`, `<|end_of_sentence|>`, `<|pad|>`, `<|unk|>`

## Kullanım (Python)

```python
from tokenizers import Tokenizer

# Tokenizer'ı doğrudan yükleyin
tokenizer = Tokenizer.from_file("turkish_tokenizer.json")

# Encode işlemi
text = "Yapay zeka Türkçe dilinde de çok güçlü."
encoded = tokenizer.encode(text)

print(encoded.tokens)
```

## Neden 128k?
Modern LLM modelleri (Örn: Llama-3, DeepSeek) yüksek işlem hızı ve dile özgü kavrayış için büyük kelime dağarcıkları (100k - 128k) kullanır. Bu tokenizer da aynı vizyonla, Türkçe dilinin eklemeli yapısını daha iyi temsil edebilmesi için 128.000 tokenlik bir kapasiteyle eğitilmiştir.
