# İddaa tahminleri · GitHub Pages

Bu proje, GitHub Pages üzerinde statik HTML arayüzü sunar. GitHub Actions, input klasöründeki Excel dosyasını JSON'a dönüştürür, RF/XGBoost/LightGBM/CatBoost birleşik tahminlerini tarih sırasıyla üretir ve siteyi yayınlar.

## GitHub'da ilk kurulum

1. GitHub'da yeni bir depo oluştur. Actions ve Pages kullanımına uygun olması için `main` dalını kullan.
2. Bu proje klasörünün **içindeki tüm dosya ve klasörleri** deponun köküne yükle. `.github/workflows/pages.yml` gizli klasörünün de yüklendiğini kontrol et.
3. Depoda `input/Excel-Açılış-Cep-Telefonu.xlsx` bulunduğundan emin ol. `input/matches.json` başlangıç verisi olarak depoda durur; workflow bu dosyayı Excel'den yeniden oluşturur ve günceller.
4. GitHub deposunda **Settings → Pages → Build and deployment → Source** bölümünden **GitHub Actions** seç.
5. **Actions** sekmesinden **Build predictions and deploy GitHub Pages** iş akışını açıp tamamlanmasını bekle. Model eğitimi veri büyüklüğüne göre birkaç dakika sürebilir.
6. Başarılı dağıtımdan sonra **Settings → Pages** bölümünde çıkan site bağlantısını aç.

GitHub Pages için uygun plan gerekir. GitHub Pages ile yayımlanan site herkese açık olabilir; Excel ve tahmin verisinin paylaşılmasında bunu göz önünde bulundur.

## Veriyi güncelleme

Yeni Excel'i aynı adla `input/Excel-Açılış-Cep-Telefonu.xlsx` üzerine yükleyip `main` dalına gönder. Workflow Excel'i `input/matches.json` dosyasına çevirir, JSON değiştiyse depoya commit eder, tüm tahminleri yeniden hesaplar ve Pages sitesini yayımlar. Actions tarafından yapılan JSON commit'i tek başına yeni workflow çalıştırmaz; aynı workflow run'ı siteyi doğrudan deploy eder.

GitHub Actions deposu ayarlarında `GITHUB_TOKEN` için **Read and write permissions** açık olmalı; workflow Excel'den oluşturulan JSON'u günceller. Site deploy adımı da GitHub Pages'in gerekli `pages: write` ve `id-token: write` yetkilerini kullanır.

## Yerel Excel → JSON dönüşümü

Proje klasöründe:

```bash
python -m pip install -r requirements.txt
python convert_excel_to_json.py
python generate_predictions.py
```

Son komut `docs/predictions.json` dosyasını üretir. `docs/index.html` bu JSON'u okur; yerel önizleme için `docs` klasörünü basit bir statik sunucuyla servis edebilirsin.

## Tahmin yöntemi

- Özellikler Excel'deki 36–74 numaralı kolonlarda bulunan tüm açılış market oranlarıdır.
- Marketler MS 1/X/2, 2,5 Üst, İY 1,5 Üst ve KG Var'dır.
- Her hedef için dört algoritma eğitilir. Olasılıklar birleştirilir ve maç için tek ana tahmin seçilir; diğer market tahminleri `+` alanında görünür.
- Her maç gününün tahmini yalnızca o tarihten önceki tamamlanmış maçlarla eğitilir. Günlük ve 30 günlük başarı oranları sadece ana tahminleri ve sonucu belli maçları sayar.
- Yeni maçlarda skorlar boş, oranlar dolu olmalıdır. İlk maç satırı 9. Excel satırıdır; skor hedefleri tam zamanlı goller 25–26 ve devre arası golleri 23–24 numaralı kolonlardan üretilir.

Tahminler istatistiksel olasılıklardır; sonuç veya kazanç garantisi vermez.
