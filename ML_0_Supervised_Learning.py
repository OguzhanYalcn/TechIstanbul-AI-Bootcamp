"""
ML(Machine Learning)
Supervised Learning (Denetimli Öğrenme= Feature(+) + Label(+))
Modelin hem giriş verileri(x)
hem de bu verilere ait doğruları cevapları (Etiketleri[Label]: Y)

Temel Yapı:
    X(Features/Özellikler) --> MODEL --> y (Label/Etiketler)

Örneğin:
- E-Posta spam mı değil mi ?
- Ev fiyatları ne kadar olur ?
- Bu müşteriye borç beyaz eşya verilir mi ?

Öğrenme türü olan LABEL vardır
"""

"""
Bir öğrencinin:
1-) Günlük çalışma saati
2-) Derse katılım yüzdesi
bu bilgilere bakarak sınavı geçip geçmeyeceğini tahmin edelim.

Label:
    0 : Kaldı
    1 : Geçti
    
Kullanılan algoritma:
    Logistic Regression  
    
    
Kurulum:
    pip install numpy scikit-learn
    python -c "import numpy; import sklearn; print('Kurulum başarılı')"
    
    python -m pip install -r requirements.txt
    
    -m → module
    -c → command
"""

import numpy as np
from sklearn.linear_model import LogisticRegression

def main():
    #  X(Features/Özellikler) --> MODEL --> y (Label/Etiketler)
    #  1= Günlük çalışma saati
    #  30= Derse katılım yüzdesi
    x = np.array([
        [1, 30],
        [2, 40],
        [2, 50],
        [3, 55],
        [4, 60],
        [5, 65],
        [6, 75],
        [7, 85],
        [8, 90],
        [9, 95]
    ])

    # y (Label/Etiketler)
    # 0 = Kaldı, 1= Geçti

    # NOT: Supervised Learning'in en önemli özelliği ==>
    # X verileriyle birlikte y etiketlerinin bulunmasıdır(SL)
    y = np.array([
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        1
    ])

    print("=== SUPERVISED LEARNING (Features(+) Label(+))")
    print("\nX - Öğrenci Özellikler(Features)")
    print(x)

    print("\ny - Label(Etiketler)")
    print(y)


    # Model Oluşturma
    """
     LogisticRegression bir sınıflandırma algoritmasıdır
     İki tane sınıf vardı
     0 --> Kaldı
     1 --> Geçti
     LogisticRegression, iki veya daha fazla sınıfın hangisine ait olduğunu tahmit etmek için kullanılan sınıfın algoritmasıdır
    """

    model =LogisticRegression()

    # Modeli Eğitim
    # Model hem özellikleri hemde dorğu cevpları görsün
    # Bu ilişkide çalışma saati + Katılım oranı --> Geçti/Kaldı
    model.fit(x,y)

    # Instance
    # Örnek: Öğrenci 6 saat çalışıyor, Derse katılım %80

    new_student = np.array([[6, 80]])

    # Tahmin
    prediction = model.predict(new_student)[0]

    # Tahmin olasılıkları
    probabilities = model.predict_proba(new_student)[0]

    print("\nYeni Öğrenci")
    print("Çalışma Saati: 6 saat")
    print("Derse katılım: %80")

    print("\nModel tahmini:", prediction)

    # Conditional
    if prediction ==1:
        print("Sonuç: Öğrencinin GEÇMESİ bekleniyor")
    else:
        print("Sonuç: Öğrencinin KALMASI bekleniyor")

    print("\nOlasılıklar:")
    print(f"Kalma olasılığı: , %{probabilities[0] * 100:.2f}")
    print(f"Geçme olasılığı: , %{probabilities[1] * 100:.2f}")

    # ============ OZET ===================
    print("\nÖZET")
    print("Supervised Learning LABEL vardır")
    print("Unutmaaaa:  Model, geçmişteki doğru cevapları öğrenir ve")
    print("Bu örnekte labek:0=Kaldı, 1=Geçti")
    print("\n Ağırlıkları: ",model.coef_)
    print("\n Bias: ",model.intercept_)

if __name__== "__main__":
    main()

"""
LogisticRegresssion:
z = 0.1422 * çalışma saati + 0.7122* katılım_yüzdesi - 45.1082
P(Geçti) =  1/(1+e^(-z))

P(Geçti) >=50 --> Geçti ==> z=0
%99,9997
"""