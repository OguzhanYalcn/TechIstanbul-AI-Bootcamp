# -*- coding: utf-8 -*-

"""
===============================================================================
MINI MACHINE LEARNING PROJESI - E-POSTA SPAM TAHMINI
===============================================================================

Bu proje, kullanicinin disaridan bir CSV dosyasi secmesini ve bu veri uzerinde
console/terminal araciligiyla temel Machine Learning adimlarini uygulamasini
saglar.

PROJENIN AMACI
--------------
Bir e-postanin SPAM olup olmadigini tahmin eden bir Classification uygulamasi
olusturmaktir.

ORNEK CSV SUTUNLARI
-------------------
kelime_sayisi
link_sayisi
buyuk_harf_orani
supheli_kelime_sayisi
gonderici_puani
ek_var
spam

Ornek:
kelime_sayisi,link_sayisi,buyuk_harf_orani,supheli_kelime_sayisi,gonderici_puani,ek_var,spam
120,0,0.05,0,92,0,0
45,6,0.72,5,18,1,1

spam:
    0 -> Normal e-posta
    1 -> Spam e-posta

ONEMLI
------
Bu proje icin hedef sutun otomatik olarak 'spam' kabul edilir.
'spam' disindaki sutunlar feature olarak kullanilir.

Program sayisal ve kategorik sutunlari otomatik algilar.
"""

# -----------------------------------------------------------------------------
# BU KOD NE ISE YARAR?
# -----------------------------------------------------------------------------
# Asagidaki import bolumu, projenin ihtiyac duydugu kutuphaneleri programa
# dahil eder.
#
# os / pathlib:
#   Dosya ve klasor islemleri icin kullanilir.
#
# pandas:
#   CSV dosyasini okumak, tablo halinde incelemek ve temizlemek icin kullanilir.
#
# numpy:
#   Sayisal islemlerde ve veri tipleriyle calisirken kullanilir.
#
# matplotlib:
#   Confusion Matrix grafigini PNG olarak kaydetmek icin kullanilir.
#
# scikit-learn:
#   Veriyi train/test olarak ayirmak, on isleme yapmak, model egitmek ve
#   Accuracy, Precision, Recall, F1 gibi metrikleri hesaplamak icin kullanilir.
# -----------------------------------------------------------------------------
from pathlib import Path
# from tkinter.ttk import Style
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
from colorama import Fore, Style
from sklearn.pipeline import Pipeline


# -----------------------------------------------------------------------------
# BU KOD NE ISE YARAR?
# -----------------------------------------------------------------------------
# AppState sinifi program boyunca kullanilan verileri tek bir yerde tutar.
#
# Neden gereklidir?
# Console uygulamalarinda kullanici once CSV yukler, sonra temizleme yapar,
# sonra model egitir. Her adimda ayni veriyi tekrar tekrar okumak yerine
# programin mevcut durumunu burada sakliyoruz.
#
# raw_df:
#   CSV dosyasindan ilk okunan, dokunulmamis orijinal veri.
#
# df:
#   Temizleme ve analiz islemlerinde kullanilan aktif veri.
#
# target_column:
#   Tahmin edilmek istenen hedef sutun.
#
# feature_columns:
#   Modelin tahmin yaparken kullanacagi giris sutunlari.
#
# best_model:
#   Egitilen modeller arasinda F1 skoruna gore en basarili model.
# -----------------------------------------------------------------------------
class AppState:
    def __init__(self):
        self.csv_path: Optional[Path] = None
        self.raw_df: Optional[pd.DataFrame] = None
        self.df: Optional[pd.DataFrame] = None

        self.target_column: Optional[str] = None
        self.feature_columns: List[str] = []

        self.best_model: Optional[Pipeline] = None
        self.best_model_name: Optional[str] = None

        self.X_test: Optional[pd.DataFrame] = None
        self.y_test: Optional[pd.Series] = None
        self.y_pred: Optional[np.ndarray] = None

        self.model_results: List[Dict[str, Any]] = []

        # Program ilk açıldığında yalnızca veri hazırlama adımları (1-6)
        # gösterilir. Veri temizleme başarıyla tamamlandığında ikinci aşama
        # yani Machine Learning seçenekleri açılır.
        self.preprocessing_completed: bool = False

        # Aktif console ekranini takip eder.
        # 1 = Veri Hazirlama, 2 = Machine Learning.
        self.current_step: int = 1

        # Aktif olarak hangi veriyle devam edildigini takip eder.
        # Degerler: 'original', 'cleaned' veya None
        self.active_data_source: Optional[str] = None
        self.cleaned_csv_path: Optional[Path] = None
        self.last_pdf_report_path: Optional[Path] = None


# -----------------------------------------------------------------------------
# BU KOD NE ISE YARAR?
# -----------------------------------------------------------------------------
# console ekranının daha okunabilir hale gelmesini sağlamak
# SOLID: Single Responsibility
def print_header(title:str) -> None: # 'başlangıçta bir şey yoksa None olarak al' diyoruz.
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# -----------------------------------------------------------------------------
# BU KOD NE ISE YARAR?
# -----------------------------------------------------------------------------
# Menü kullanıcının sonucu okuyabilmesi için ENTER
def pause() -> None:
    input("\nDevam etmek için lütfen ENTER tuşuna basınız...")

# -----------------------------------------------------------------------------
# BU KOD NE ISE YARAR?
# -----------------------------------------------------------------------------
# Menu yazısını farklı renklerde kullanmamızı sağlar.
def print_menu_option(text:str) -> None:
    print(Fore.LIGHTCYAN_EX + text + Style.RESET_ALL)

# -----------------------------------------------------------------------------
# BU KOD NE ISE YARAR?
# -----------------------------------------------------------------------------
# print_step_title fonksiyonu STEP başlıklarını menü seçeneklerinden ayırmak için parlak camgöbeği renkte gösterir.
def print_step_title(text:str) -> None:
    print(Fore.CYAN + Style.BRIGHT + text + Style.RESET_ALL)



# -----------------------------------------------------------------------------
# BU KOD NE ISE YARAR?
# -----------------------------------------------------------------------------
# CSV sütun adlarını daha düzenli hale getirmek
# Örnek:
# " KeliME Sayısi " → "kelime_sayisi"
# Bu sayede sütun isimlerindeki boşluk, büyük/küçük harf farklarından kaynaklanan hataları azaltmak
def normalize_column_name(name:str) -> str:
    value = str(name).replace("\ufeff", "").strip().lower()

    replacemenets = {
        "ç": "c",
        "ğ": "g",
        "ı": "i",
        "ö": "o",
        "ş": "s",
        "ü": "u",
        " ": "_",
        "-": "_",
        "/": "_",
        "\\": "_",
    }

    for old, new in replacemenets.items():
        value = value.replace(old, new)

    while "__" in value:
        value = value.replace("__", "_")

    return value.strip("_")



# -----------------------------------------------------------------------------
# BU KOD NE ISE YARAR?
# -----------------------------------------------------------------------------
# discover_csv_files fonksiyonu kullanıcının dosya seçebilmesi için CSV dosyalarını tarar ve sadece görünen CSV dosyalarını eklemeye yani dinamik olarak CSV dosyalarını seçmeye yarar.
def discover_csv_files() -> List[Path]:
    found: List[Path] = []

    search_dirs = [
        Path.cwd(),
        Path.cwd() / "data"
    ]

    for folder in search_dirs:
        if not folder.exists() or not folder.is_dir():
            continue

        for file_path in folder.glob("*.csv"):
            resolved = file_path.resolve()
            if resolved not in found:
                found.append(resolved)
    return sorted(found, key=lambda p:p.name.lower())


# -----------------------------------------------------------------------------
# BU KOD NE ISE YARAR?
# -----------------------------------------------------------------------------
# Manuel olarak (copy-paste) olarak girilen yolu seçmek
# Seçeneklerden
def choose_csv_path() -> Optional[Path]:
    print_header("CSV DOSYASINI SEÇ")

    # -----------------------------------------------------------------
    # BU MENU NE ISE YARAR?
    # -----------------------------------------------------------------
    # Kullanıcı CSV dosyasını iki farklı yöntemle seçebilir:
    #
    # 0 - Ana Menüye Dön
    #     CSV seçmeden STEP 1 ana menüsüne geri döner.
    #
    # 1 - Dosya Yolunu Manuel Gir
    #     Kullanıcı CSV dosyasının tam yolunu klavyeden yazar.
    #
    #     Örnek:
    #         E:\ML\veriler\spam.csv
    #
    # 2 - Dosya Yolunu Dosya Seçerek Gir
    #     Windows/Linux dosya seçme penceresi açılır.
    #     Kullanıcı CSV dosyasını tıklayarak seçer.
    # -----------------------------------------------------------------
    print_menu_option("0 - Ana Menüye Dön")
    print_menu_option("1 - Dosya Yolunu Manuel Gir")
    print_menu_option("2 - Dosya Yolunu Dosya Seçerek Gir")

    choice = input("\nSeçiminiz: ").strip()

    if choice == "0":
        return None

    if choice == "1":
        raw_path = input(
            "\nCSV dosyasının tam yolunu giriniz: "
        ).strip().strip('"')

        if not raw_path:
            print("\nHATA: Dosya yolu boş bırakılamaz.")
            return None

        # expanduser: kısayolları gerçek klasör yoluna çevirmeye yarar.
        # ~\Desktop\veri.csv C:\Users\Data\Desktop\veri.csv
        path = Path(raw_path).expanduser()

        if not path.exists():
            print("\nHATA: Girilen dosya bulunamadı.")
            return None

        if not path.is_file():
            print("\nHATA: Girilen yol dosya değil.")
            return None

        if path.suffix.lower() != ".csv":
            print("\nHATA: Girilen dosya CSV uzantılı değil.")
            return None

        print(f"\nSeçilen CSV dosyası:\n{path.resolve()}")
        return path.resolve()

    # -----------------------------------------------------------------------------
    # BU KOD NE ISE YARAR?
    # -----------------------------------------------------------------------------
    # tkinter: Python'ın standart kütüphanelerinden biridir.
    # Kullanıcının fare imleci ile dosya seçerek programa aktarılmasıdır.
    if choice == "2":
        try:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()

            # Dosya seçme penceresinin arkada kalmasını engellemeye çalışır.
            try:
                root.attributes("-topmost", True)
            except Exception:
                pass

            selected_file = filedialog.askopenfilename(
                title="CSV Dosyasını Seç",
                filetypes = [
                    ("CSV Dosyaları", "*.csv"),
                    ("Tüm Dosyalar", "*.txt"),
                ]
            )

            root.destroy()

            if not selected_file:
                print("\nDosya Seçimi İptal Edildi.")
                return None

            path = Path(selected_file)

            if not path.exists():
                print("\nHATA: Seçilen dosya bulunamadı.")
                return None

            if path.suffix.lower() != ".csv":
                print("\nLütfen CSV uzantılı bir dosya seçiniz.")
                return None

            print(f"\nSeçilen CSV dosyası:\n{path.resolve()}")
            return path.resolve()
        except ImportError:
            print(
                "\nHATA: Bu Python sürümünde tkinter bulunamadı.\n"
                "Alternatif olarak '1 - Dosya Yolunu Manuel Gir' seçeneğini de kullanabilirsiniz."
            )
            return None
    print("\nHATA: 0 <= X <= 2 arasında tam sayı seçmelisiniz. Yani 0,1,2 kullanabilirsiniz.")
    return None

# -----------------------------------------------------------------------------
# BU KOD NE ISE YARAR?
# -----------------------------------------------------------------------------
# CSV dosyasını pandas DataFrame formatında okur.
# Seperator: Virgül, noktalı virgül vb. pandas tarafından otomatik olarak tahmin etmesine yardımcı olan metriklerdir.
def read_csv_safely(path:Path) -> pd.DataFrame:
    encodings = ["utf-8", "utf-8-sig", "latin-1"]

    last_error = None

    for encoding in encodings:
        try:
            return pd.read_csv(
                path,
                sep=None,
                engine="python",
                encoding=encoding
            )
        except Exception as exc:
            last_error = exc

    # Zorlayarak Hata Fırlatma
    raise RuntimeError(
        f"CSV dosyası okunamadı. Son Hata: {last_error}"
    )

# -----------------------------------------------------------------------------
# BU KOD NE ISE YARAR?
# -----------------------------------------------------------------------------
# load_csv

def load_csv(state: AppState) -> None:
    path = choose_csv_path()

    if path is None:
        return

    try:
        df = read_csv_safely(path)

        if df.empty:
            print("\nHATA: CSV dosyası boş.")
            return

        df.columns = [normalize_column_name(col) for col in df.columns]

        state.csv_path = path
        state.raw_df = df.copy(deep=True)
        state.df = df.copy(deep=True)

        state.target_column = None
        state.feature_columns = []

        state.best_model = None
        state.best_model_name = None

        state.X_test = None
        state.y_test = None
        state.y_pred = None

        state.model_results = []

        state.current_step = 2
        state.preprocessing_completed = False
        state.cleaned_csv_path = None
        state.active_data_source = "original"

        print_header("CSV BAŞARIYLA YÜKLENDİ.")

        file_size_kb = path.stat().st_size / 1024
        missing_total = int(df.isna().sum().sum()) # toplamın toplamı
        duplicated_total = int(df.duplicated().sum())

        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = [
            column for column in df.columns
            if column not in numeric_columns
        ]

        target_column = "spam" if "spam" in df.columns else None
        features_columns = [
            column for column in df.columns
            if column != target_column
        ]

        print(f"Dosya Adı:                    {path.name}")
        print(f"Dosya Yolu:                   {path}")
        print(f"Dosya Boyutu:                 {file_size_kb:.2f} KB")
        print(f"Dosya Satır Sayısı:           {len(df)}")
        print(f"Dosya Sütun Sayısı:           {len(numeric_columns)}")
        print(f"Kategorik Sütun:              {len(categorical_columns)}")
        print(f"Eksik Değer:                  {missing_total}")
        print(f"Duplicate Satır:              {duplicated_total}")

        if target_column:
            print(f"Target / Label       : {target_column}")
            print(f"Problem Türü         : classification")
            print(f"Feature Sayısı       : {len(features_columns)}")

        else:
            print(f"Target / Label       : BULUNAMADI")
            print(f"Problem Türü         : BELİRTİLMEDİ")
            print(f"UYARI                : CSV içinde 'spam' sütunu bulunamadı.")

        print("\nFeature Sütunları")
        for column in features_columns:
            print(f"- {column}")

        if target_column:
            print("\nTarget / Label")
            print(f"- {target_column}")

        print("\nCSV Kullanıma Hazır.")
    except Exception as exc:
        print(f"\nHATA: CSV Yüklenemedi. \n{exc}")

# -----------------------------------------------------------------------------
# BU KOD NE ISE YARAR?
# -----------------------------------------------------------------------------
# CSV dosyası yüklenmeden menü çalışmasını engelle
def require_data(state: AppState) -> bool:
    if state.df is None:
        print("\nÖnce bir CSV dosyasını yüklemelisiniz.")
        return False
    return True