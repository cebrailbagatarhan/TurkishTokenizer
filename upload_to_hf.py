from huggingface_hub import HfApi, login

print("=== Hugging Face Tokenizer Yükleme Aracı ===")
print("Eğer Hugging Face'e giriş yapmadıysanız https://huggingface.co/settings/tokens adresinden bir WRITE (Yazma) yetkili 'Access Token' alıp buraya yapıştırın.")
token = input("Hugging Face Access Token (Gizli görünmeyebilir, yapıştırıp Enter'a basın): ")

if token.strip():
    login(token=token.strip())

api = HfApi()

hf_username = input("Hugging Face Kullanıcı Adınız (Örn: cebrail): ")
repo_name = input("Oluşturulacak Modelin Adı (Örn: turkish-bpe-tokenizer-128k): ")

repo_id = f"{hf_username}/{repo_name}"

print(f"\n{repo_id} adlı depo (repo) oluşturuluyor...")
api.create_repo(repo_id=repo_id, exist_ok=True, repo_type="model")

print("Dosyalar yükleniyor (turkish_tokenizer.json, README.md, train_turkish_tokenizer.py)...")
api.upload_folder(
    folder_path=".", 
    allow_patterns=["*.json", "*.py", "*.md"], 
    repo_id=repo_id,
    repo_type="model"
)

print(f"\nHarika! Modeliniz Hugging Face'e başarıyla yüklendi 🚀")
print(f"Buradan ulaşabilirsiniz: https://huggingface.co/{repo_id}")
