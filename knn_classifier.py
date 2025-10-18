import numpy as np
from collections import Counter


class KNNClassifier:
    """
    K-En Yakın Komşuluk (K-Nearest Neighbors) Sınıflandırıcı
    """

    def __init__(self, k=3, distance_metric='l2'):
        """
        KNN sınıflandırıcıyı başlatır

        Parameters:
        -----------
        k : int
            Komşu sayısı
        distance_metric : str
            Mesafe metriği ('l1' veya 'l2')
        """
        self.k = k
        self.distance_metric = distance_metric.lower()  # Case-insensitive yapalım
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        Training verisini kaydeder

        Parameters:
        -----------
        X : numpy array, shape (n_samples, n_features)
            Training özellikleri
        y : numpy array, shape (n_samples,)
            Training etiketleri
        """
        # Training verisini self.X_train ve self.y_train'e kaydedin
        self.X_train = X
        self.y_train = y

    def compute_distances(self, X):
        """
        Test örnekleri ile training örnekleri arasındaki mesafeleri hesaplar

        Parameters:
        -----------
        X : numpy array, shape (n_test, n_features)
            Test örnekleri

        Returns:
        --------
        distances : numpy array, shape (n_test, n_train)
            Mesafe matrisi
        """
        # X: (n_test, n_features)
        # X_train: (n_train, n_features)
        n_test = X.shape[0]
        n_train = self.X_train.shape[0]
        distances = np.zeros((n_test, n_train))

        # Test verisi ile training verisi arasındaki fark matrisi
        # Fark matrisi boyutu: (n_test, n_train, n_features)

        # NumPy'ın broadcasting özelliği kullanılarak optimizasyon
        # (X_test[:, None, :] - X_train[None, :, :])
        diff = X[:, np.newaxis, :] - self.X_train[np.newaxis, :, :]

        if self.distance_metric == 'l1':
            # L1 (Manhattan) mesafe: Sum(|x1 - x2|)
            # diff'in mutlak değerlerinin son eksen boyunca toplamı
            distances = np.sum(np.abs(diff), axis=2)

        elif self.distance_metric == 'l2':
            # L2 (Euclidean) mesafe: sqrt(Sum((x1 - x2)^2))
            # diff'in karesinin son eksen boyunca toplamının karekökü
            distances = np.sqrt(np.sum(diff ** 2, axis=2))

        else:
            raise ValueError(f"Bilinmeyen mesafe metriği: {self.distance_metric}")

        return distances

    def predict(self, X):
        """
        Test örnekleri için tahmin yapar

        Parameters:
        -----------
        X : numpy array, shape (n_test, n_features)
            Test örnekleri

        Returns:
        --------
        predictions : numpy array, shape (n_test,)
            Tahmin edilen etiketler
        """
        # 1. compute_distances() ile mesafeleri hesaplayın
        distances = self.compute_distances(X)  # (n_test, n_train)
        n_test = X.shape[0]
        predictions = np.zeros(n_test, dtype=self.y_train.dtype)

        for i in range(n_test):
            # 2. Her test örneği için k en yakın komşuyu bulun
            # Mesafeleri sıralayıp, ilk k index'i al
            # argsort, sıralanmış dizinin orijinal indekslerini verir
            closest_k_indices = np.argsort(distances[i])[:self.k]

            # K en yakın komşuların etiketleri
            k_nearest_labels = self.y_train[closest_k_indices]

            # 3. Majority voting ile sınıf tahmini yapın
            # Counter ile etiket sayımlarını bul ve en çok tekrar edeni (most_common(1)) seç
            most_common = Counter(k_nearest_labels).most_common(1)
            predictions[i] = most_common[0][0]

        # 4. Tahminleri döndürün
        return predictions

    def score(self, X, y):
        """
        Model accuracy'sini hesaplar

        Parameters:
        -----------
        X : numpy array, shape (n_test, n_features)
            Test özellikleri
        y : numpy array, shape (n_test,)
            Gerçek etiketler

        Returns:
        --------
        accuracy : float
            Doğruluk skoru (0-1 arası)
        """
        # 1. predict() ile tahmin yapın
        y_pred = self.predict(X)

        # 2. Doğru tahmin sayısını / toplam örnek sayısını hesaplayın
        accuracy = np.mean(y_pred == y)
        return accuracy