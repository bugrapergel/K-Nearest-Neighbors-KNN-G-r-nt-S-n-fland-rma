# K-EN YAKIN KOMŞULUK (KNN) İLE GÖRÜNTÜ SINIFLANDIRMA ÖDEVİ

## Proje Amacı

Bu proje, K-En Yakın Komşuluk (K-Nearest Neighbors - KNN) algoritmasının temel mantığını uygulamak, L1 (Manhattan) ve L2 (Öklid) mesafe metriklerini karşılaştırmak ve Scikit-learn kütüphanesindeki yerleşik KNN sınıflandırıcısı ile performans karşılaştırması yapmayı amaçlamaktadır

Projede, Scikit-learn'ün **MNIST Digits** veri seti kullanılarak rakam görüntüleri sınıflandırılmıştır

## Dosya Yapısı

Proje, aşağıdaki Python dosyalarından oluşmaktadır:


knn_classifier.py`: K-En Yakın Komşuluk algoritmasının çekirdek implementasyonunu içerir (`fit`, `compute_distances`, `predict`, `score` fonksiyonları)

visualization.py`: Confusion Matrix, k-analizi grafiği ve örnek tahminler gibi görselleştirme fonksiyonlarını içerir

`experiments.py`: Deneyleri, parametre analizini ve Scikit-learn karşılaştırmasını içeren ana test kodudur

results/`: Çalıştırma sonucunda üretilen tüm görselleştirmelerin (`.png` dosyaları) kaydedildiği klasördür[
## Kurulum ve Çalıştırma

Kodu çalıştırmak için Python 3 ortamı ve aşağıdaki kütüphaneler gereklidir.

### 1. Kütüphane Kurulumu

Projenin bağımlılıklarını kurmak için, sanal ortamınız (`.venv`) aktifken aşağıdaki komutu çalıştırın:

```bash
pip install numpy scikit-learn matplotlib seaborn pandas tabulate jupyter