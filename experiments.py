import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from knn_classifier import KNNClassifier
from visualization import *
from sklearn.neighbors import KNeighborsClassifier
import time
import os
import pandas as pd

# results klasörünü oluştur
os.makedirs('results', exist_ok=True)


def load_data():
    """
    MNIST Digits veri setini yükler ve train/test'e böler
    """
    # 1. load_digits() ile veriyi yükleyin
    digits = load_digits()
    X = digits.data
    y = digits.target

    # 3. Veriyi normalize edin (0-1 arası)
    # MNIST digits verisi 0-16 arasında değer alır, 16'ya bölerek normalize ediyoruz
    X = X / 16.0

    # 2. train_test_split() ile %80 train, %20 test olacak şekilde bölün
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test


def test_mnist_basic():
    """
    Görev 1.2: MNIST üzerinde temel KNN testi [cite: 27]
    """
    print("=" * 50)
    print("Görev 1.2: MNIST Digits Testi (k=3, L2)")
    print("=" * 50)

    # 1. Veriyi yükleyin
    X_train, X_test, y_train, y_test = load_data()

    # 2. KNN modelini oluşturun (k=3, L2 mesafe)
    knn = KNNClassifier(k=3, distance_metric='l2')

    # 3. Modeli eğitin
    knn.fit(X_train, y_train)

    # Tahmin yapın
    y_pred = knn.predict(X_test)

    # 4. Test accuracy'sini hesaplayın ve yazdırın
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy (k=3, L2): {accuracy:.4f}")

    # 5. Confusion matrix oluşturun (ve görselleştirin)
    plot_confusion_matrix(y_test, y_pred)

    # Örnek tahminleri görselleştirin
    plot_sample_predictions(X_test, y_test, y_pred, n_samples=10)

    return accuracy


def analyze_k_values():
    """
    Görev 1.3a: Farklı k değerlerinin etkisini analiz eder
    """
    print("\n" + "=" * 50)
    print("Görev 1.3a: K Değeri Analizi (L2)")
    print("=" * 50)

    X_train, X_test, y_train, y_test = load_data()

    # 1. k değerleri listesi oluşturun
    k_values = [1, 3, 5, 7, 9, 11, 15, 21]
    accuracies = []

    # 2. Her k değeri için test et
    for k in k_values:
        knn = KNNClassifier(k=k, distance_metric='l2')
        knn.fit(X_train, y_train)
        acc = knn.score(X_test, y_test)
        accuracies.append(acc)
        print(f"k={k}, L2 Accuracy: {acc:.4f}")

    # 3. Sonuçları tablo olarak yazdır (Pandas ile)
    results_df = pd.DataFrame({'k': k_values, 'Accuracy': accuracies})
    print("\nK Değeri Analiz Tablosu (L2 Mesafe):")
    print(results_df.to_markdown(index=False, floatfmt=".4f"))

    # 4. Grafik çiz (k vs accuracy)
    plot_k_analysis(k_values, accuracies)

    return k_values, accuracies


def compare_distance_metrics():
    """
    Görev 1.3b: L1 ve L2 mesafe metriklerini karşılaştırır


    """
    print("\n" + "=" * 50)
    print("Görev 1.3b: Mesafe Metriği Karşılaştırması")
    print("=" * 50)

    X_train, X_test, y_train, y_test = load_data()

    # 1. k değerleri listesi
    k_values = [1, 3, 5, 7, 9, 11, 15, 21]
    l1_accuracies = []
    l2_accuracies = []

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
        l2_accuracies.append(acc_l2)

        print(f"k={k}: L1 Acc: {acc_l1:.4f}, L2 Acc: {acc_l2:.4f}, Fark (L2-L1): {(acc_l2 - acc_l1):.4f}")

    # 3. Sonuçları tablo olarak yazdır
    data = {'K Değeri': k_values,
            'L1 Accuracy': l1_accuracies,
            'L2 Accuracy': l2_accuracies,
            'Fark (L2-L1)': np.array(l2_accuracies) - np.array(l1_accuracies)}
    results_df = pd.DataFrame(data)
    print("\nMesafe Metriği Karşılaştırma Tablosu:")
    print(results_df.to_markdown(index=False, floatfmt=".4f"))

    # 4. Karşılaştırma grafiği çiz
    plot_distance_comparison(k_values, l1_accuracies, l2_accuracies)
    create_comparison_table(k_values, l1_accuracies, l2_accuracies)


def compare_with_sklearn():
    """
    Bölüm 2: Sklearn KNN ile karşılaştırma
    """
    print("\n" + "=" * 50)
    print("Bölüm 2: Sklearn Karşılaştırması (k=3, L2)")
    print("=" * 50)

    X_train, X_test, y_train, y_test = load_data()

    k = 3

    # 1. Scikit-learn KNN
    start_time_sklearn = time.time()
    sklearn_knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    sklearn_knn.fit(X_train, y_train)
    sklearn_accuracy = sklearn_knn.score(X_test, y_test)
    end_time_sklearn = time.time()
    time_sklearn = end_time_sklearn - start_time_sklearn

    # 2. Kendi implementasyonunuz
    start_time_your = time.time()
    your_knn = KNNClassifier(k=k, distance_metric='l2')
    your_knn.fit(X_train, y_train)
    your_accuracy = your_knn.score(X_test, y_test)
    end_time_your = time.time()
    time_your = end_time_your - start_time_your

    # 3. Accuracy'leri karşılaştırın
    print(f"Sklearn KNN Accuracy: {sklearn_accuracy:.4f}")
    print(f"Your KNN Accuracy: {your_accuracy:.4f}")
    print(f"Accuracy Farkı (abs): {abs(sklearn_accuracy - your_accuracy):.4f}")

    # 4. Çalışma sürelerini ölçün ve yazdırın
    print(f"\nSklearn Çalışma Süresi (s): {time_sklearn:.4f}")
    print(f"Your KNN Çalışma Süresi (s): {time_your:.4f}")
    print(f"Hız Farkı (Your/Sklearn Oranı): {time_your / time_sklearn:.2f} kat")


if __name__ == "__main__":
    # Tüm deneyleri çalıştır
    test_mnist_basic()
    analyze_k_values()
    compare_distance_metrics()
    compare_with_sklearn()