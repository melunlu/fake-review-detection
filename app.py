import streamlit as st
import pandas as pd
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import words
import nltk
import matplotlib.pyplot as plt
import numpy as np
import io
from deep_translator import GoogleTranslator

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('words', quiet=True)

english_vocab = set(w.lower() for w in words.words())


class TextPreprocessor:
    """
    Handles all text preprocessing operations including cleaning,
    validation, and translation for the fake review detection pipeline.
    """

    def __init__(self):
        """Initialize lemmatizer and English stopwords."""
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def clean(self, text: str) -> str:
        """
        Clean and normalize raw review text for model inference.

        Steps:
            1. Convert to lowercase
            2. Remove non-alphabetic characters
            3. Tokenize by whitespace
            4. Remove stopwords and apply lemmatization

        Args:
            text (str): Raw review text.

        Returns:
            str: Cleaned and lemmatized text ready for vectorization.
        """
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        tokens = text.split()
        tokens = [self.lemmatizer.lemmatize(word)
                  for word in tokens if word not in self.stop_words]
        return ' '.join(tokens)

    def is_valid(self, text: str) -> tuple:
        """
        Validate whether the input text is a meaningful review.

        Checks:
            - Minimum length (>= 6 characters, >= 2 words)
            - Not composed solely of numbers or symbols
            - Contains at least one alphabetic character
            - At least 35% of words exist in English vocabulary

        Args:
            text (str): Input text to validate (should be in English).

        Returns:
            tuple: (bool, str) — (is_valid, reason_message)
        """
        raw_text = str(text).strip()

        if len(raw_text) < 6 or len(raw_text.split()) < 2:
            return False, "Tekrar deneyiniz."

        if re.match(r'^[0-9\W_]+$', raw_text):
            return False, "Girdi sadece sayı veya sembollerden oluşamaz."

        cleaned_words = re.sub(r'[^a-z\s]', '', raw_text.lower()).split()
        if not cleaned_words:
            return False, "Anlamlı karakter bulunamadı."

        valid_word_count = sum(1 for word in cleaned_words if word in english_vocab)
        valid_ratio = valid_word_count / len(cleaned_words)

        if valid_ratio < 0.35:
            return False, "Girdi anlamsız karakterler veya kelimeler içeriyor."

        return True, "Geçerli"

    def translate_to_english(self, text: str) -> str:
        """
        Translate input text to English using Google Translate API.

        Args:
            text (str): Text in any language.

        Returns:
            str: Translated English text, or original text if translation fails.
        """
        try:
            return GoogleTranslator(source='auto', target='en').translate(text)
        except Exception:
            return text


class ReviewClassifier:
    """
    Loads the trained SVM model and TF-IDF vectorizer,
    and provides prediction functionality for single and batch reviews.
    """

    def __init__(self, model_path: str = 'model.pkl', tfidf_path: str = 'tfidf.pkl'):
        """
        Load model and TF-IDF vectorizer from disk.

        Args:
            model_path (str): Path to the pickled classification model.
            tfidf_path (str): Path to the pickled TF-IDF vectorizer.
        """
        self.model, self.tfidf = self._load(model_path, tfidf_path)

    @st.cache_resource
    def _load(_self, model_path: str, tfidf_path: str):
        """
        Load and cache model and vectorizer from pickle files.

        Args:
            model_path (str): Path to model pickle file.
            tfidf_path (str): Path to TF-IDF pickle file.

        Returns:
            tuple: (model, tfidf_vectorizer)
        """
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(tfidf_path, 'rb') as f:
            tfidf = pickle.load(f)
        return model, tfidf

    def predict(self, clean_text: str) -> tuple:
        """
        Predict whether a cleaned review text is fake or real.

        Args:
            clean_text (str): Preprocessed review text.

        Returns:
            tuple: (prediction: int, probabilities: np.ndarray)
                   prediction — 0 for fake, 1 for real
                   probabilities — [P(fake), P(real)]
        """
        vectorized = self.tfidf.transform([clean_text])
        prediction = self.model.predict(vectorized)[0]
        probability = self.model.predict_proba(vectorized)[0]
        return prediction, probability


# ── Initialize core objects ───────────────────────────────────────────────────
preprocessor = TextPreprocessor()
classifier = ReviewClassifier()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Sahte Yorum Tespit Sistemi", page_icon="🔍", layout="wide")

st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght=400;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        .stApp {
            background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        }

        h1, h2, h3, h4, p, div, label, span {
            color: white !important;
        }

        .hero-box {
            background: linear-gradient(135deg, #667eea, #764ba2);
            border-radius: 20px;
            padding: 40px;
            text-align: center;
            margin-bottom: 30px;
            box-shadow: 0 8px 32px rgba(102, 126, 234, 0.4);
        }
        .hero-box h1 { font-size: 2.8rem !important; font-weight: 700 !important; margin-bottom: 10px; }
        .hero-box p { font-size: 1.1rem !important; opacity: 0.9; }

        .result-card {
            border-radius: 15px;
            padding: 25px;
            text-align: center;
            margin: 15px 0;
            font-size: 1.4rem;
            font-weight: 700;
        }
        .result-fake {
            background: linear-gradient(135deg, #3a0ca3, #7209b7);
            box-shadow: 0 4px 20px rgba(114, 9, 183, 0.5);
        }
        .result-real {
            background: linear-gradient(135deg, #5e60ce, #9b5de5);
            box-shadow: 0 4px 20px rgba(155, 93, 229, 0.5);
        }

        .metric-card {
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.15);
            border-radius: 15px;
            padding: 20px;
            text-align: center;
            backdrop-filter: blur(10px);
        }
        .metric-number {
            font-size: 2rem !important;
            font-weight: 700 !important;
            background: linear-gradient(135deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .dosya-kart {
            background: rgba(60, 20, 120, 0.4);
            border: 1px solid rgba(114, 49, 191, 0.5);
            border-radius: 12px;
            padding: 14px 20px;
            margin-bottom: 10px;
            cursor: pointer;
            transition: all 0.2s ease;
        }
        .dosya-kart:hover {
            background: rgba(114, 49, 191, 0.5);
            border-color: rgba(150, 80, 220, 0.8);
        }
        .dosya-kart-aktif {
            background: linear-gradient(135deg, rgba(102,126,234,0.3), rgba(118,75,162,0.3));
            border: 1px solid rgba(150, 80, 220, 0.9);
            border-radius: 12px;
            padding: 14px 20px;
            margin-bottom: 10px;
        }

        .stTextArea textarea {
            background: rgba(60, 20, 120, 0.35) !important;
            border: 1px solid rgba(114, 49, 191, 0.35) !important;
            border-radius: 12px !important;
            color: white !important;
            font-size: 16px !important;
            caret-color: white !important;
            box-shadow: none !important;
        }
        .stTextArea textarea:focus {
            border: 1px solid rgba(150, 80, 220, 0.5) !important;
            box-shadow: 0 0 10px rgba(114, 49, 191, 0.2) !important;
            outline: none !important;
        }
        .stTextArea textarea:hover {
            border: 1px solid rgba(150, 80, 220, 0.5) !important;
        }
        .stTextArea textarea::placeholder {
            color: rgba(255, 255, 255, 0.4) !important;
        }

        [data-testid="stFileUploader"] {
            background: rgba(60, 20, 120, 0.35) !important;
            border: 1px solid rgba(114, 49, 191, 0.35) !important;
            border-radius: 16px !important;
            backdrop-filter: blur(10px);
        }
        [data-testid="stFileUploaderDropzone"] {
            background: transparent !important;
            border: none !important;
        }
        [data-testid="stFileUploader"] section {
            background: transparent !important;
        }
        [data-testid="stFileUploader"] button {
            background: linear-gradient(135deg, #667eea, #764ba2) !important;
            color: white !important;
            border: none !important;
        }

        .stSelectbox div[data-baseweb="select"] > div {
            background: rgba(60, 20, 120, 0.6) !important;
            border: 1px solid rgba(114, 49, 191, 0.5) !important;
            border-radius: 10px !important;
            color: white !important;
        }

        .stButton > button {
            background: linear-gradient(135deg, #667eea, #764ba2) !important;
            color: white !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 12px 40px !important;
            font-size: 16px !important;
            font-weight: 600 !important;
            width: 100% !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
        }
        .stButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6) !important;
        }

        .stTabs [data-baseweb="tab"] {
            background: rgba(255,255,255,0.05) !important;
            border-radius: 10px !important;
            color: white !important;
            font-size: 16px !important;
            font-weight: 600 !important;
            padding: 10px 25px !important;
        }
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #667eea, #764ba2) !important;
        }
        .stTabs [data-baseweb="tab-highlight"] {
            background-color: #764ba2 !important;
        }
        .stTabs [data-baseweb="tab-border"] {
            background-color: rgba(114, 49, 191, 0.3) !important;
        }

        .stDataFrame { background: rgba(255, 255, 255, 0.05) !important; border-radius: 12px !important; }

        .stDownloadButton > button {
            background: rgba(60, 20, 120, 0.6) !important;
            border: 1px solid rgba(114, 49, 191, 0.5) !important;
            border-radius: 10px !important;
            color: white !important;
            width: 100% !important;
        }
        .stDownloadButton > button:hover {
            background: rgba(114, 49, 191, 0.7) !important;
        }

        hr { border-color: rgba(255,255,255,0.1) !important; }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="hero-box">
        <h1>🔍 Sahte Yorum Tespit Sistemi</h1>
        <p>Yapay zeka destekli bu sistem, e-ticaret yorumlarının sahte mi gerçek mi olduğunu saniyeler içinde analiz eder.</p>
    </div>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if 'dosyalar' not in st.session_state:
    st.session_state.dosyalar = {}
if 'aktif_dosya' not in st.session_state:
    st.session_state.aktif_dosya = None

tab1, tab2 = st.tabs(["🔎 Tekli Analiz", "📂 Toplu Analiz"])

# ── TEKLİ ANALİZ ──────────────────────────────────────────────────────────────
with tab1:
    st.markdown("<br>", unsafe_allow_html=True)
    user_input = st.text_area("Yorumu buraya yazınız", height=150,
                               placeholder="Örnek: This product is amazing! Best purchase ever...")

    if st.button("🔍 Analiz Et"):
        if user_input.strip() == "":
            st.warning("Lütfen yorum girin")
        else:
            with st.spinner("🌍 Dil işleniyor..."):
                translated_text = preprocessor.translate_to_english(user_input)

            is_valid, error_msg = preprocessor.is_valid(translated_text)

            if not is_valid:
                st.error(f"⚠️ Geçersiz Yorum Algılandı: {error_msg}")
            else:
                progress = st.progress(0)
                status = st.empty()

                with st.spinner("🔄 Yorum analiz ediliyor..."):
                    status.write("Metin temizleniyor...")
                    progress.progress(25)
                    cleaned = preprocessor.clean(translated_text)

                    status.write("Vektörleştiriliyor...")
                    progress.progress(50)

                    status.write("Tahmin yapılıyor...")
                    progress.progress(80)
                    prediction, probability = classifier.predict(cleaned)

                    progress.progress(100)
                    status.success("✓ Analiz tamamlandı")
                    progress.empty()

                sahte_oran = probability[0] * 100
                gercek_oran = probability[1] * 100

                st.markdown("<br>", unsafe_allow_html=True)

                if prediction == 0:
                    st.markdown("""
                        <div class="result-card result-fake">
                            🚨 Bu yorum SAHTE görünüyor!
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                        <div class="result-card result-real">
                            ✅ Bu yorum GERÇEK görünüyor!
                        </div>
                    """, unsafe_allow_html=True)

                if user_input.lower().strip() != translated_text.lower().strip():
                    with st.expander("🌐 Çeviri Detayını Gör"):
                        st.write(f"**Orijinal Metin:** {user_input}")
                        st.write(f"**Analiz Edilen (İngilizce):** {translated_text}")

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("#### 📊 Sahtelik / Gerçeklik Oranı")

                bar_html = f"""
                <div style="margin: 10px 0 50px 0;">
                    <div style="
                        width: 100%;
                        height: 32px;
                        border-radius: 20px;
                        background: linear-gradient(to right, #10002b, #3c096c, #5a189a, #7b2fbe, #9d4edd, #c77dff, #e0aaff);
                        position: relative;
                        box-shadow: 0 4px 15px rgba(0,0,0,0.4);
                    ">
                        <div style="
                            position: absolute;
                            left: {gercek_oran}%;
                            top: -8px;
                            transform: translateX(-50%);
                            width: 5px;
                            height: 48px;
                            background-color: white;
                            border-radius: 3px;
                            box-shadow: 0 0 12px rgba(255,255,255,0.9);
                        "></div>
                        <div style="
                            position: absolute;
                            left: {gercek_oran}%;
                            top: 48px;
                            transform: translateX(-50%);
                            font-weight: 700;
                            font-size: 16px;
                            color: white;
                            background: rgba(114, 49, 191, 0.6);
                            padding: 2px 12px;
                            border-radius: 10px;
                            backdrop-filter: blur(5px);
                            border: 1px solid rgba(255,255,255,0.2);
                        ">%{gercek_oran:.1f}</div>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-top: 55px; font-size: 14px; color: rgba(255,255,255,0.6);">
                        <span>← Sahte</span>
                        <span>Gerçek →</span>
                    </div>
                </div>
                """
                st.markdown(bar_html, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"""
                        <div class="metric-card">
                            <div style="font-size:1rem; opacity:0.7;">🚨 Sahtelik Olasılığı</div>
                            <div class="metric-number">%{sahte_oran:.1f}</div>
                        </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                        <div class="metric-card">
                            <div style="font-size:1rem; opacity:0.7;">✅ Gerçeklik Olasılığı</div>
                            <div class="metric-number">%{gercek_oran:.1f}</div>
                        </div>
                    """, unsafe_allow_html=True)

# ── TOPLU ANALİZ ──────────────────────────────────────────────────────────────
with tab2:
    st.markdown("<br>", unsafe_allow_html=True)

    col_sol, col_sag = st.columns([1, 2])

    with col_sol:
        st.markdown("#### 📁 Yüklenen Dosyalar")

        uploaded_file = st.file_uploader(
            "Dosya ekle (CSV, TXT veya Excel)",
            type=["csv", "txt", "xlsx"],
            label_visibility="collapsed"
        )

        if uploaded_file is not None:
            dosya_adi = uploaded_file.name
            file_type = dosya_adi.split('.')[-1].lower()

            if dosya_adi not in st.session_state.dosyalar:
                if file_type == 'csv':
                    df_yeni = pd.read_csv(uploaded_file)
                    metin_sutun = df_yeni.columns[0]
                elif file_type == 'xlsx':
                    df_yeni = pd.read_excel(uploaded_file)
                    metin_sutun = df_yeni.columns[0]
                elif file_type == 'txt':
                    lines = uploaded_file.read().decode('utf-8').splitlines()
                    lines = [l.strip() for l in lines if l.strip() != ""]
                    df_yeni = pd.DataFrame(lines, columns=["yorum"])
                    metin_sutun = "yorum"

                if len(df_yeni) > 100:
                    st.error(f"⚠️ Dosya çok büyük ({len(df_yeni)} satır). API limitleri nedeniyle maksimum 100 satır yüklenebilir.")
                else:
                    with st.spinner("🌍 Çevriliyor ve Analiz ediliyor..."):

                        def toplu_islem(row_text):
                            """
                            Process a single review row for batch analysis.

                            Translates to English, validates, cleans,
                            and predicts fake/real label.

                            Args:
                                row_text: Raw review text from dataframe row.

                            Returns:
                                tuple: (label: str, english_text: str)
                            """
                            try:
                                t_text = preprocessor.translate_to_english(str(row_text))
                                valid, _ = preprocessor.is_valid(t_text)
                                if not valid:
                                    return "⚠️ Geçersiz", t_text
                                cleaned = preprocessor.clean(t_text)
                                pred, _ = classifier.predict(cleaned)
                                label = '✅ Gerçek' if pred == 1 else '🚨 Sahte'
                                return label, t_text
                            except Exception:
                                return "❌ Hata", str(row_text)

                        sonuclar = df_yeni[metin_sutun].apply(toplu_islem)
                        df_yeni['tahmin_label'] = [s[0] for s in sonuclar]
                        df_yeni['english_version'] = [s[1] for s in sonuclar]

                    st.session_state.dosyalar[dosya_adi] = {
                        'df': df_yeni,
                        'metin_sutun': metin_sutun
                    }
                    st.session_state.aktif_dosya = dosya_adi

        if st.session_state.dosyalar:
            for dosya_adi in st.session_state.dosyalar:
                aktif = dosya_adi == st.session_state.aktif_dosya
                css_class = "dosya-kart-aktif" if aktif else "dosya-kart"
                icon = "📂" if aktif else "📄"
                st.markdown(f'<div class="{css_class}">{icon} {dosya_adi}</div>', unsafe_allow_html=True)
                if st.button(f"Seç", key=f"btn_{dosya_adi}"):
                    st.session_state.aktif_dosya = dosya_adi
                    st.rerun()
        else:
            st.markdown("""
                <div style="color: rgba(255,255,255,0.4); font-size:14px; text-align:center; padding:20px;">
                    Henüz dosya yüklenmedi
                </div>
            """, unsafe_allow_html=True)

    with col_sag:
        if st.session_state.aktif_dosya and st.session_state.aktif_dosya in st.session_state.dosyalar:
            veri = st.session_state.dosyalar[st.session_state.aktif_dosya]
            df_goster = veri['df']
            metin_sutun = veri['metin_sutun']

            st.markdown(f"#### 📊 {st.session_state.aktif_dosya} — Analiz Sonuçları")

            sahte_sayi = (df_goster['tahmin_label'] == '🚨 Sahte').sum()
            gercek_sayi = (df_goster['tahmin_label'] == '✅ Gerçek').sum()
            gecersiz_sayi = (df_goster['tahmin_label'] == '⚠️ Geçersiz Girdi').sum()
            toplam = len(df_goster)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                    <div class="metric-card">
                        <div style="font-size:1rem; opacity:0.7;">📝 Toplam / Geçersiz</div>
                        <div class="metric-number">{toplam} <span style="font-size:14px; opacity:0.5;">({gecersiz_sayi} Geçersiz)</span></div>
                    </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                    <div class="metric-card">
                        <div style="font-size:1rem; opacity:0.7;">🚨 Sahte Yorum</div>
                        <div class="metric-number">{sahte_sayi}</div>
                    </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                    <div class="metric-card">
                        <div style="font-size:1rem; opacity:0.7;">✅ Gerçek Yorum</div>
                        <div class="metric-number">{gercek_sayi}</div>
                    </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            col_grafik, col_bos = st.columns([1, 1])
            with col_grafik:
                if sahte_sayi + gercek_sayi > 0:
                    fig2, ax2 = plt.subplots(figsize=(4, 4))
                    fig2.patch.set_alpha(0)
                    ax2.set_facecolor('none')
                    ax2.pie(
                        [sahte_sayi, gercek_sayi],
                        labels=[f'Sahte ({sahte_sayi})', f'Gerçek ({gercek_sayi})'],
                        colors=['#3c096c', '#9d4edd'],
                        autopct='%1.1f%%',
                        startangle=90,
                        textprops={'color': 'white', 'fontsize': 12}
                    )
                    ax2.set_title('Sahte / Gerçek Dağılımı', fontsize=13, color='white')
                    st.pyplot(fig2)
                else:
                    st.info("Grafik oluşturulacak geçerli veri bulunamadı.")

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### 📋 Yorum Detayları")
            st.dataframe(df_goster[[metin_sutun, 'tahmin_label']], use_container_width=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### 📥 Sonuçları İndir")
            col_dl1, col_dl2, col_dl3 = st.columns(3)

            csv_data = df_goster[[metin_sutun, 'tahmin_label']].to_csv(index=False).encode('utf-8')
            col_dl1.download_button(
                label="📄 CSV olarak indir",
                data=csv_data,
                file_name=f"{st.session_state.aktif_dosya}_sonuc.csv",
                mime="text/csv"
            )

            txt_data = "\n".join(
                f"{row[metin_sutun]} --> {row['tahmin_label']}"
                for _, row in df_goster.iterrows()
            ).encode('utf-8')
            col_dl2.download_button(
                label="📝 TXT olarak indir",
                data=txt_data,
                file_name=f"{st.session_state.aktif_dosya}_sonuc.txt",
                mime="text/plain"
            )

            excel_buffer = io.BytesIO()
            df_goster[[metin_sutun, 'tahmin_label']].to_excel(excel_buffer, index=False)
            col_dl3.download_button(
                label="📊 Excel olarak indir",
                data=excel_buffer.getvalue(),
                file_name=f"{st.session_state.aktif_dosya}_sonuc.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            st.markdown("""
                <div style="text-align: center; padding: 60px 20px;
                            color: rgba(255,255,255,0.4); font-size: 18px;">
                    📂 Sol taraftan bir dosya seçin veya yükleyin
                </div>
            """, unsafe_allow_html=True)