import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd


def plot_confusion_matrix(y_true, y_pred, save_path='results/confusion_matrix.png'):
    """
    Confusion matrix görselleştirmesi [cite: 32]
    """
    # 1. confusion_matrix() ile matrix oluştur
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    # 2. seaborn heatmap ile görselleştir
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
    plt.title('Confusion Matrix')
    plt.xlabel('Tahmin Edilen Etiket')
    plt.ylabel('Gerçek Etiket')
    plt.tight_layout()
    # 3. Grafiği kaydet
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion Matrix kaydedildi: {save_path}")


def plot_sample_predictions(X_test, y_test, y_pred, n_samples=10,
                            save_path='results/sample_predictions.png'):
    """
    Örnek tahminleri görselleştirir [cite: 33]
    """
    # 1. Random n_samples örnek seç
    # Görüntü boyutunu bilmediğimiz için X_test'in 8x8 olduğunu varsayıyoruz (MNIST digits)
    indices = np.random.choice(len(X_test), n_samples, replace=False)

    # 3. Subplotlar halinde düzenle
    # 5x2 düzeni uygun
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()

    for i, idx in enumerate(indices):
        image = X_test[idx].reshape(8, 8)  # 8x8 reshape et
        true_label = y_test[idx]
        predicted_label = y_pred[idx]

        color = 'green' if true_label == predicted_label else 'red'  # Doğru yeşil, yanlış kırmızı

        ax = axes[i]
        # 2. Görüntüyü göster
        ax.imshow(image, cmap='gray')
        # Etiketi yaz
        ax.set_title(f"Gerçek: {true_label}\nTahmin: {predicted_label}",
                     color=color, fontsize=12)
        ax.axis('off')

    plt.suptitle(f'{n_samples} Örnek Görüntü Tahmini (Yeşil: Doğru, Kırmızı: Yanlış)', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    # 4. Grafiği kaydet
    plt.savefig(save_path)
    plt.close()
    print(f"Örnek Tahminler kaydedildi: {save_path}")


def plot_k_analysis(k_values, accuracies, save_path='results/k_value_analysis.png'):
    """
    K değeri analizi grafiği [cite: 69]
    """
    plt.figure(figsize=(10, 6))
    # 1. Line plot oluştur
    plt.plot(k_values, accuracies, marker='o', linestyle='-', color='blue')

    # En iyi k değerini bul
    best_k_index = np.argmax(accuracies)
    best_k = k_values[best_k_index]
    best_acc = accuracies[best_k_index]

    # 2. En iyi k değerini işaretle
    plt.plot(best_k, best_acc, 'ro', markersize=10, label=f'En İyi k={best_k} ({best_acc:.4f})')

    # 3. Eksen etiketlerini ve başlığı ekle
    plt.xlabel('k değeri')
    plt.ylabel('Accuracy (Doğruluk)')
    plt.title('K Değerinin Accuracy\'e Etkisi (L2 Mesafe)')
    plt.xticks(k_values)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    # 4. Grafiği kaydet
    plt.savefig(save_path)
    plt.close()
    print(f"K Değeri Analiz Grafiği kaydedildi: {save_path}")


def plot_distance_comparison(k_values, l1_accuracies, l2_accuracies,
                             save_path='results/distance_comparison.png'):
    """
    L1 ve L2 mesafe metriklerini karşılaştırır [cite: 70, 72]
    """
    plt.figure(figsize=(10, 6))
    # 1. İki line plot çiz (L1 ve L2)
    plt.plot(k_values, l1_accuracies, marker='o', label='L1 (Manhattan) Mesafe')
    plt.plot(k_values, l2_accuracies, marker='s', label='L2 (Euclidean) Mesafe')

    # 3. Eksen etiketlerini ve başlığı ekle
    plt.xlabel('k değeri')
    plt.ylabel('Accuracy (Doğruluk)')
    plt.title('L1 vs L2 Mesafe Metriği Karşılaştırması')
    plt.xticks(k_values)
    plt.grid(True)
    # 2. Legend ekle
    plt.legend()
    plt.tight_layout()
    # 4. Grafiği kaydet
    plt.savefig(save_path)
    plt.close()
    print(f"Mesafe Metriği Karşılaştırma Grafiği kaydedildi: {save_path}")


def create_comparison_table(k_values, l1_accuracies, l2_accuracies,
                            save_path='results/comparison_table.png'):
    """
    Karşılaştırma tablosu oluşturur [cite: 89]
    """
    # 1. Tablo verisi hazırla
    data = {'K Değeri': k_values,
            'L1 Accuracy': l1_accuracies,
            'L2 Accuracy': l2_accuracies,
            'Fark (L2-L1)': np.array(l2_accuracies) - np.array(l1_accuracies)}
    df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('tight')
    ax.axis('off')

    # Tabloyu görselleştir
    # Her sütun için format belirleme
    cell_text = []
    for row in df.itertuples(index=False):
        cell_text.append([str(row[0])] + [f'{val:.4f}' for val in row[1:]])

    table = ax.table(cellText=cell_text,
                     colLabels=df.columns,
                     loc='center',
                     cellLoc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.2)

    plt.title('L1 ve L2 Mesafe Metriği Karşılaştırma Tablosu', y=0.85)
    # 4. Tabloyu kaydet
    plt.savefig(save_path)
    plt.close()
    print(f"Karşılaştırma Tablosu kaydedildi: {save_path}")