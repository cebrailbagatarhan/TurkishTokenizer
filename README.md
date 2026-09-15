---
language:
- tr
tags:
- tokenizer
- bpe
- turkce
- turkish
---

# Turkish tokenizer benchmark

Bu depo, Türkçe için Byte-Level BPE tokenizer eğitimi ve karşılaştırması yapmaya yarayan deneysel bir projedir. Büyük vocabulary'nin otomatik olarak daha iyi olduğu varsayılmaz; 32K, 64K ve 128K seçenekleri aynı corpus üzerinde ölçülür.

> [!NOTE]
> Kökteki `turkish_tokenizer.json`, daha önce üretilmiş 128K artifact'idir. Bu dosyayla birlikte corpus hash'i, dataset revision'ı ve eğitim manifesti yayımlanmadığı için tek başına “128K daha iyi” sonucunu desteklemez. Karşılaştırılabilir yeni artifact'ler aşağıdaki akışla üretilmelidir.

## Ölçülen sorular

| Metrik | Ne anlatır? |
| --- | --- |
| Token / kelime | Türkçe kelimelerin ortalama kaç parçaya bölündüğü |
| Token / karakter | Corpus uzunluğuna göre token yoğunluğu |
| Encode karakter/s | Aynı makinedeki tokenizer encode throughput'u |
| Ek tek-token oranı | Yaygın Türkçe eklerinin tam olarak tek token span'iyle temsil edilme oranı |
| Ek başına örtüşen token | Bir ekin ortalama kaç token parçasıyla örtüştüğü |
| Vocabulary size | Bellek/çıktı katmanı maliyetine etki eden gerçek vocabulary boyutu |

Token/kelime tek başına kalite metriği değildir. Daha büyük vocabulary; embedding ve LM head parametrelerini büyütür, nadir tokenları artırabilir ve aynı corpus üzerinde encode hızını değiştirebilir. Sonuçlar bu trade-off'larla birlikte yorumlanmalıdır.

## Kurulum

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Gated bir Llama tokenizer'ına erişilecekse Hugging Face erişimini önceden yapılandırın veya `HF_TOKEN` ortam değişkenini kullanın. Tokenı komut satırı argümanı olarak vermeyin.

## 1. Corpus'u bir kez hazırla

Bu adım dataset revision'ını SHA'ya çözer, deterministic streaming shuffle uygular ve eğitim/benchmark splitlerini JSONL olarak kaydeder.

```bash
python prepare_corpus.py \
  --dataset wikimedia/wikipedia \
  --dataset-config 20231101.tr \
  --seed 42 \
  --train-docs 150000 \
  --benchmark-docs 5000 \
  --output-dir data/turkish-wikipedia
```

Çıktılar:

- `data/turkish-wikipedia/train.jsonl`
- `data/turkish-wikipedia/benchmark.jsonl`
- `data/turkish-wikipedia/manifest.json`

Manifest; resolved dataset SHA, seed, belge/karakter/byte sayıları ve iki dosyanın SHA-256 hash'ini içerir. Farklı vocabulary eğitimleri aynı `train.jsonl` dosyasını kullanmalıdır.

## 2. 32K, 64K ve 128K modelleri eğit

```bash
python train_turkish_tokenizer.py --corpus data/turkish-wikipedia/train.jsonl --corpus-manifest data/turkish-wikipedia/manifest.json --vocab-size 32000
python train_turkish_tokenizer.py --corpus data/turkish-wikipedia/train.jsonl --corpus-manifest data/turkish-wikipedia/manifest.json --vocab-size 64000
python train_turkish_tokenizer.py --corpus data/turkish-wikipedia/train.jsonl --corpus-manifest data/turkish-wikipedia/manifest.json --vocab-size 128000
```

Her komut tokenizer JSON'unun yanında bir eğitim manifesti üretir. Varsayılan yollar:

- `artifacts/turkish_bpe_32000.json`
- `artifacts/turkish_bpe_64000.json`
- `artifacts/turkish_bpe_128000.json`

Script sabit süre tahmini yazmaz; gerçek eğitim süresini, corpus hash'ini ve gerçek vocabulary boyutunu kaydeder.

## Tokenizer yayın doğrulaması

Eğitim için `--corpus-manifest` zorunludur. Eğitim yalnız manifestteki `train`
splitini kabul eder; eksik/geçersiz hash, farklı corpus veya eğitim sırasında
değişen girdiler işlemi durdurur. Eğitim manifestinin v2 şeması tokenizer,
corpus, corpus manifesti ve eğitim kodunun hash'lerini birbirine bağlar.

Aynı corpus ve manifestle, aynı parametreler ve Python/tokenizers sürümleriyle
ikinci kez eğitim çalıştırın. İlk ve ikinci çalışmayı ayrı dizinlerde tutun:

```bash
python train_turkish_tokenizer.py --corpus data/turkish-wikipedia/train.jsonl --corpus-manifest data/turkish-wikipedia/manifest.json --vocab-size 32000 --output artifacts/run-a/tokenizer.json
python train_turkish_tokenizer.py --corpus data/turkish-wikipedia/train.jsonl --corpus-manifest data/turkish-wikipedia/manifest.json --vocab-size 32000 --output artifacts/run-b/tokenizer.json
python verify_tokenizer.py \
  --tokenizer artifacts/run-a/tokenizer.json \
  --manifest artifacts/run-a/tokenizer.manifest.json \
  --repeat-tokenizer artifacts/run-b/tokenizer.json \
  --repeat-manifest artifacts/run-b/tokenizer.manifest.json \
  --corpus data/turkish-wikipedia/train.jsonl \
  --corpus-manifest data/turkish-wikipedia/manifest.json \
  --output artifacts/release-evidence.json
```

Kontrol; iki tokenizer dosyasının SHA-256 eşitliğini, eğitim girdilerinin
eşleşmesini, Türkçe Unicode, bozuk JSON, yarım kod, kontrol karakterleri,
özel token metni ve sentetik secret örneklerinin encode/decode sonucunu sınar.
Örüntünün kaç tokena bölündüğü JSON raporuna yazılır; BPE'de parçalanma tek
başına hata veya secret sızıntısı sayılmaz. Gerçek credential kullanılmaz.
İkinci çalışma eksikse sonuç `incomplete`, doğrulama başarısızsa `deny` olur;
her iki durumda da komut sıfırdan farklı çıkış kodu döndürür.

Yükleme komutu da aynı kontrolleri ağ işleminden önce çalıştırır. Önce kuru
çalıştırma yapın; yayımlama için `--dry-run` seçeneğini kaldırın:

```bash
python upload_to_hf.py --repo-id OWNER/MODEL \
  --tokenizer artifacts/run-a/tokenizer.json \
  --manifest artifacts/run-a/tokenizer.manifest.json \
  --repeat-tokenizer artifacts/run-b/tokenizer.json \
  --repeat-manifest artifacts/run-b/tokenizer.manifest.json \
  --corpus data/turkish-wikipedia/train.jsonl \
  --corpus-manifest data/turkish-wikipedia/manifest.json --dry-run
```

Tokenizer, iki manifest, README ve kontrol raporu tek Hub commitinde yüklenir;
corpus metni yüklenmez. Hata halinde aday yayımlanmaz. Geri dönüş için son
doğrulanmış Hub commitini sabitleyin ve o committeki tokenizer ile manifest
paketini birlikte kullanın. Manifesti olmayan kök artifact yayın kontrolünden
geçmez; kaynak corpus ile yeniden eğitilmelidir.

Bu kontrol imza veya güvenilir bir bağımsız eğitim kanıtı sağlamaz: bir dosyanın
kopyası ikinci eğitim gibi sunulabilir. İki eğitimi güvenilen bir CI/operatör
çalıştırmalı, kanıt paketini saklamalıdır. Hash kontrolü, güvenilmeyen bir kişinin
artifact ile manifesti birlikte değiştirmesini tek başına engellemez. Aynı
dosya hash'i farklı paket sürümleri veya platformlar arasında vaat edilmez.

Çevrimdışı testler küçük sentetik corpus kullanır; 32K/64K/128K kalite veya
performans benchmark'ı yerine geçmez:

```bash
python -m pip install -r requirements-security.txt
python -m unittest discover -s tests -v
```

CI aynı testleri Linux ve Windows üzerinde çalıştırır.

## 3. Yerel ve Hugging Face tokenizer'larını karşılaştır

```bash
python benchmark_tokenizers.py \
  --corpus data/turkish-wikipedia/benchmark.jsonl \
  --local bpe-32k=artifacts/turkish_bpe_32000.json \
  --local bpe-64k=artifacts/turkish_bpe_64000.json \
  --local bpe-128k=artifacts/turkish_bpe_128000.json \
  --hf qwen=Qwen/Qwen2.5-7B \
  --hf llama=meta-llama/Meta-Llama-3.1-8B \
  --output-json benchmark-results/tokenizers.json \
  --output-markdown benchmark-results/tokenizers.md
```

Hugging Face model revision'ları çalıştırma sırasında commit SHA'ya çözülür ve sonuç JSON'una yazılır. Llama erişimi yoksa ilgili satır `failed` olarak kaydedilir; diğer ölçümler devam eder.

## Sonuç durumu

Bu repoya henüz aynı held-out corpus, aynı makine ve aynı protokolle alınmış karşılaştırma sonucu commit edilmemiştir.

| Tokenizer | Revision / artifact hash | Vocab | Token/kelime | Token/karakter | Encode karakter/s | Durum |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Türkçe BPE 32K | — | — | — | — | — | not measured |
| Türkçe BPE 64K | — | — | — | — | — | not measured |
| Türkçe BPE 128K | — | — | — | — | — | not measured |
| Qwen | — | — | — | — | — | not measured |
| Llama | — | — | — | — | — | not measured |

Bu tablo yalnızca `benchmark-results` altındaki ham JSON sonucu da PR'a eklendiğinde doldurulmalıdır.

## Tekrar üretilebilirlik kuralları

- Dataset adı yeterli değildir; resolved dataset commit SHA raporlanır.
- Corpus dosyası SHA-256 ile doğrulanır.
- Eğitim ve benchmark splitleri bir kez hazırlanır; vocabulary boyutları için yeniden örneklenmez.
- Seed, shuffle buffer, belge sınırı, text column ve paket sürümleri manifestte saklanır.
- Benchmark corpus'u eğitim corpus'undan ayrıdır.
- Encode benchmark'ı aynı process içinde warm-up sonrası ve aynı batch boyutuyla çalışır.
- Sonuçlar farklı donanımlar arasında doğrudan hız karşılaştırması olarak kullanılmaz.

## Dosyalar

| Dosya | Rol |
| --- | --- |
| `prepare_corpus.py` | Pinned ve hash'li train/benchmark corpus üretir |
| `train_turkish_tokenizer.py` | İstenen vocabulary boyutunda BPE eğitir ve manifest yazar |
| `benchmark_tokenizers.py` | Yerel, Qwen ve Llama tokenizer ölçümlerini üretir |
| `upload_to_hf.py` | Açıkça seçilen artifact'leri güvenli kimlik doğrulamayla yükler |
| `turkish_tokenizer.json` | Manifesti olmayan legacy 128K artifact |

## Sınırlamalar

- Wikipedia tek başına bütün Türkçe kullanım alanlarını temsil etmez.
- Ek metriği morfolojik çözümleyici değildir; belirlenmiş yüzey eklerinin token offset'lerini ölçer.
- Vocabulary boyutu seçimi, tokenizer metriğinin yanında model parametre/RAM maliyetiyle birlikte yapılmalıdır.
- Benchmark tokenizer kalitesini tek başına kanıtlamaz; downstream dil modeli değerlendirmesi ayrı bir deneydir.
