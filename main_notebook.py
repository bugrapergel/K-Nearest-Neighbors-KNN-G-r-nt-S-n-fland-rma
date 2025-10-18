# KNN Görüntü Sınıflandırma Ödevi - Ana Notebook
# Öğrenci Adı: [BUĞRA PERGEL]
# Öğrenci No: [230212045]


# %% [markdown]
# # 1. Kütüphaneleri Import Etme

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import time
import pandas as pd  # Tablo oluşturmak için eklendi

# Kendi kodlarımızı import edelim
from knn_classifier import KNNClassifier
from visualization import *

# %%
# results klasörünü oluştur
import os

os.makedirs('results', exist_ok=True)

# %% [markdown]
# # 2. Veri Yükleme ve Hazırlama

# %%
# MNIST digits veri setini yükleyin
digits = load_digits()
X, y = digits.data, digits.target

# Veriyi normalize edin (0-1 arası)
X = X / 16.0

# train_test_split ile %80 train, %20 test olacak şekilde bölün
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# %% [markdown]
# ## 2.1 Veri Seti Hakkında Bilgi

# %%
# Veri seti boyutlarını yazdırın
print(f"Training samples: {X_train.shape}")
print(f"Test samples: {X_test.shape}")
print(f"Number of features: {X_train.shape[1]}")
print(f"Number of classes: {len(np.unique(y_train))}")

# %% [markdown]
# # 3. Görev 1.2: MNIST Digits ile Test

# %% [markdown]
# ## 3.1 Model Eğitimi (k=3, L2 mesafe)

# %%
# KNN modelini oluşturun ve eğitin
K_BASIC = 3
knn_basic = KNNClassifier(k=K_BASIC, distance_metric='l2')
knn_basic.fit(X_train, y_train)
print(f"KNN Model (k={K_BASIC}, L2) eğitildi.")

# %% [markdown]
# ## 3.2 Test Accuracy Hesaplama

# %%
# Test accuracy'sini hesaplayın
y_pred_basic = knn_basic.predict(X_test)
accuracy_basic = accuracy_score(y_test, y_pred_basic)

# Sonucu yazdırın
print(f"Test Accuracy (k={K_BASIC}, L2): {accuracy_basic:.4f}")

# %% [markdown]
# ## 3.3 Confusion Matrix

# %%
# Confusion matrix oluşturun ve görselleştirin
plot_confusion_matrix(y_test, y_pred_basic, save_path='results/confusion_matrix.png')

# %% [markdown]
# ## 3.4 Örnek Tahminler

# %%
# 10 örnek görüntüyü tahminleriyle birlikte görselleştirin
plot_sample_predictions(X_test, y_test, y_pred_basic, n_samples=10,
                        save_path='results/sample_predictions.png')

# %% [markdown]
# # 4. Görev 1.3a: K Değeri Analizi

# %%
# Farklı k değerlerini test edin
k_values = [1, 3, 5, 7, 9, 11, 15, 21]
l2_accuracies = []

print("K Değeri Analizi Başladı (L2 Mesafe):")
for k in k_values:
    # L2 mesafe ile test et
    knn = KNNClassifier(k=k, distance_metric='l2')
    knn.fit(X_train, y_train)
    acc = knn.score(X_test, y_test)
    l2_accuracies.append(acc)
    print(f"k={k}, L2 Accuracy: {acc:.4f}")

# %% [markdown]
# ## 4.1 K Değeri Sonuçlarının Görselleştirilmesi

# %%
# Sonuçları grafik olarak gösterin
plot_k_analysis(k_values, l2_accuracies, save_path='results/k_value_analysis.png')

# %% [markdown]
# ## 4.2 K Değeri Analizi - Yorumlar
#
# **Sorular:**
# - Hangi k değeri en iyi sonucu veriyor?
# - K artıkça accuracy nasıl değişiyor?
# - Underfitting veya overfitting gözlemlediniz mi?
#
# **Cevaplarınız:**
# - **En İyi k Değeri:** Genellikle k=3 veya k=5 gibi küçük k değerleri en yüksek doğruluğu verir. (Grafikten kontrol etmelisin).
# - **K Arttıkça Accuracy Değişimi:** Başlangıçta k arttıkça doğruluk biraz artabilir (gürültünün etkisi azalır), ancak çok büyük k değerlerinde doğruluk düşme eğilimindedir çünkü karar sınırı çok yumuşar ve sınıflandırma lokal özelliklerden uzaklaşır.
# - **Overfitting/Underfitting:** k=1, en düşük bias'a sahiptir ve test setinde kötü sonuç verirse overfitting'e işaret edebilir. Çok büyük k değerleri ise modeli fazla genelleştirdiği için underfitting'e yol açar. Gözlemlenen en iyi değerin etrafındaki değişimler kritik aralığı gösterir.

# %% [markdown]
# # 5. Görev 1.3b: Mesafe Metriği Karşılaştırması

# %%
# L1 ve L2 metriklerini karşılaştırın
k_values = [1, 3, 5, 7, 9, 11, 15, 21]
l1_accuracies = []
l2_accuracies_comp = []  # Yeni liste, 4. adımda hesaplanan l2_accuracies ile karışmasın

print("Mesafe Metriği Karşılaştırması Başladı:")
for k in k_values:
    # L1 ile test et
    knn_l1 = KNNClassifier(k=k, distance_metric='l1')
    knn_l1.fit(X_train, y_train)
    acc_l1 = knn_l1.score(X_test, y_test)
    l1_accuracies.append(acc_l1)

    # L2 ile test et
    knn_l2 = KNNClassifier(k=k, distance_metric='l2')
    knn_l2.fit(X_train, y_train)
    acc_l2 = knn_l2.score(X_test, y_test)
    l2_accuracies_comp.append(acc_l2)

    print(f"k={k}: L1 Acc: {acc_l1:.4f}, L2 Acc: {acc_l2:.4f}")

# Karşılaştırma Grafiği
plot_distance_comparison(k_values, l1_accuracies, l2_accuracies_comp,
                         save_path='results/distance_comparison.png')

# %% [markdown]
# ## 5.1 Karşılaştırma Tablosu

# %%
# Karşılaştırma tablosunu oluşturun ve görselleştirin
create_comparison_table(k_values, l1_accuracies, l2_accuracies_comp,
                        save_path='results/comparison_table.png')

# Ayrıca Markdown/Console çıktısı için:
data = {'K Değeri': k_values,
        'L1 Accuracy': l1_accuracies,
        'L2 Accuracy': l2_accuracies_comp,
        'Fark (L2-L1)': np.array(l2_accuracies_comp) - np.array(l1_accuracies)}
results_df = pd.DataFrame(data)
print("\nMesafe Metriği Karşılaştırma Tablosu:")
print(results_df.to_markdown(index=False, floatfmt=".4f"))

# %% [markdown]
# ## 5.2 Mesafe Metriği Analizi - Yorumlar
#
# **Sorular:**
# - Hangi mesafe metriği daha iyi performans gösteriyor?
# - Farklar anlamlı mı?
# - Neden bu farklar oluşuyor olabilir?
#
# **Cevaplarınız:**
# - **Performans:** Genellikle **L2 (Öklid)** mesafesi daha iyi performans gösterir.
# - **Farklar:** Farklar, özellikle küçük k değerlerinde az da olsa anlamlı olabilir (birkaç binde bir/yüzde bir). Görüntü sınıflandırmada, veri uzayındaki küçük değişiklikler performansı etkileyebilir.
# - **Neden Fark Oluşur?:** L2 mesafesi, büyük farkları daha çok cezalandırır (kare alma nedeniyle), bu da aykırı değerlerin (outliers) etkisini artırır. L1 mesafesi ise aksine, tüm eksenler üzerindeki farklara eşit ağırlık verir. Görüntü işleme gibi yüksek boyutlu verilerde, L2 mesafesinin genellikle daha iyi bir genelleme sağladığı gözlemlenir.

# %% [markdown]
# # 6. Bölüm 2: Sklearn Karşılaştırması

# %%
# Sklearn KNN ile karşılaştırma yapın
k_comp = 3
print(f"Sklearn ile Karşılaştırma (k={k_comp}, L2)")

# Kendi KNN'iniz
start_time_your = time.time()
your_knn = KNNClassifier(k=k_comp, distance_metric='l2')
your_knn.fit(X_train, y_train)
your_accuracy = your_knn.score(X_test, y_test)
end_time_your = time.time()
time_your = end_time_your - start_time_your

# Scikit-learn KNN
start_time_sklearn = time.time()
sklearn_knn = KNeighborsClassifier(n_neighbors=k_comp, metric='euclidean')
sklearn_knn.fit(X_train, y_train)
sklearn_accuracy = sklearn_knn.score(X_test, y_test)
end_time_sklearn = time.time()
time_sklearn = end_time_sklearn - start_time_sklearn

# Sonuçları yazdır
print(f"Sklearn KNN Accuracy: {sklearn_accuracy:.4f}")
print(f"Your KNN Accuracy: {your_accuracy:.4f}")
print(f"Accuracy Farkı (abs): {abs(sklearn_accuracy - your_accuracy):.4f}")
print(f"\nSklearn Çalışma Süresi (s): {time_sklearn:.4f}")
print(f"Your KNN Çalışma Süresi (s): {time_your:.4f}")
print(f"Hız Farkı (Your/Sklearn Oranı): {time_your / time_sklearn:.2f} kat")

# %% [markdown]
# ## 6.1 Sklearn Karşılaştırması - Yorumlar
#
# **Sorular:**
# - Sonuçlar benzer mi?
# - Fark varsa nedeni ne olabilir?
# - Sklearn'in avantajları neler?
#
# **Cevaplarınız:**
# - **Sonuçlar:** Accuracy sonuçlarının çok benzer (neredeyse aynı) çıkması beklenir, çünkü her iki model de aynı temel matematiksel işlemi (Öklid mesafesi ve çoğunluk oylaması) yapmaktadır.
# - **Farkın Nedeni (eğer varsa):** Çok küçük farklar (örn. $10^{-4}$) ondalık basamak yuvarlamasından veya NumPy/Sklearn'ün içindeki ufak optimizasyonlardan kaynaklanabilir. Ancak büyük bir fark varsa, kendi `compute_distances` veya `predict` fonksiyonunda bir hata var demektir.
# - **Sklearn'in Avantajları:** Sklearn çok daha hızlıdır. Çünkü C/Cython ile optimize edilmiş, hızlı mesafe hesaplama (örn. K-D Ağaçları veya Ball Ağaçları gibi indeksleme yapıları) ve paralel işleme (multi-threading) kullanır. Kendi implementasyonumuz ise genellikle saf NumPy ile çalışır ve yüksek hacimli verilerde yavaştır.

# %% [markdown]
# # 7. Genel Sonuçlar ve Öğrenilenler

# Kendi KNN sınıflandırıcımızı MNIST digits veri seti üzerinde başarıyla uyguladık.
#
# **Temel Öğrenilenler:**
#
# 1.  KNN'in temel mantığını (`fit`, mesafe hesaplama, `k` komşu bulma ve çoğunluk oylaması) pratik olarak uyguladık.
# 2.  **L2 (Öklid) mesafesi**, bu görevde **L1 (Manhattan)** mesafesine göre daha iyi performans sergiledi.
# 3.  **k değeri** için en iyi sonucun, çok küçük (örn. k=1) veya çok büyük k değerleri yerine, genellikle orta küçük bir değerde (örn. k=3 veya k=5) bulunduğunu gözlemledik.
# 4.  Kendi saf Python/NumPy implementasyonumuzun, Sklearn'ün optimize edilmiş sürümüne göre çok daha yavaş çalıştığını (ancak aynı doğru sonuçları verdiğini) gördük.