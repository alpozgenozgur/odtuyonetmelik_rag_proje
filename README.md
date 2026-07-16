# ODTÜ Yönetmelik Asistanı | METU Regulations Assistant

> A RAG-based chatbot that answers student questions about METU's Undergraduate Education Regulations, with article-level citations.

[Türkçe](#türkçe) | [English](#english)

---

## English

### Why?

University regulations are written in legal language: articles, sub-clauses, cross-references. Finding the answer to a simple question — "how many courses can I withdraw from in a semester?" — means scanning pages of dense text.

This bot reads the regulation, answers in plain Turkish, and **cites the article it relied on**, so you can verify it yourself.

### Example

**Question:** Bir yarıyılda kaç dersten çekilebilirim?

**Answer:** Madde 24'e göre, bir yarıyıl içinde en çok bir dersten çekilme işlemi yapabilirsiniz. Öğreniminiz boyunca ise toplamda en fazla altı dersten çekilebilirsiniz.

### How it works
Regulation page (oidb.metu.edu.tr)
↓  BeautifulSoup extracts only the regulation body (nav/footer stripped)
Raw text (~40,000 characters)
↓  Split into 1000-char chunks with 200-char overlap
~45 chunks
↓  Embedded with Gemini embedding model
ChromaDB (persisted to disk — processed once, not on every run)
↓  Top 4 relevant chunks retrieved per question
Retrieval
↓  Chunks + question injected into prompt
Gemini 2.5 Flash
↓
Plain-language answer with article citation

### Design decisions

**Grounded only in the regulation.** The prompt forces the model to rely solely on retrieved regulation text. If something isn't in the regulation, it doesn't invent an answer — it says so and points the user to the Student Affairs Office. For a legal document, a hallucinated answer is worse than no answer.

**Cites its source.** Every answer states which article it came from, so the user can verify against the original text.

**Stays in scope.** Off-topic questions are declined rather than answered from the model's general knowledge.

**Ingests once, reads from disk after.** Embeddings persist to `.chroma`. The first run scrapes and processes the regulation; subsequent runs load the existing vector store — faster, and no repeated API quota usage.

### Stack

- **LangChain** — RAG pipeline
- **Google Gemini 2.5 Flash** — answer generation
- **Gemini Embedding** — text vectorization
- **ChromaDB** — vector store with disk persistence
- **BeautifulSoup** — targeted web scraping

### Setup

```bash
git clone https://github.com/alpozgenozgur/odtu-yonetmelik-rag.git
cd odtu-yonetmelik-rag

python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

Create a `.env` file:
GOOGLE_API_KEY=your_key_here

Get a free API key from [Google AI Studio](https://aistudio.google.com).

Run:

```bash
python main.py
```

The first run downloads and processes the regulation (a few seconds). After that it loads instantly.

### Limitations

- If the regulation is amended, delete `.chroma` and re-run to re-ingest.
- Covers only the **Undergraduate** Education Regulation. Graduate studies, disciplinary rules, and other regulations are out of scope.
- Informational only — for official matters, consult the Student Affairs Office.

### Roadmap

- [ ] Streamlit web interface
- [ ] Additional METU regulations (routed to separate collections)
- [ ] English regulation support (bilingual)
- [ ] Live demo on Hugging Face Spaces

---

## Türkçe

### Neden?

Üniversite yönetmelikleri hukuki dille yazılıyor: maddeler, alt fıkralar, çapraz referanslar. Basit bir sorunun cevabını bulmak — "bir dönemde kaç dersten çekilebilirim?" — sayfalarca yoğun metin taramak demek.

Bu bot yönetmeliği okuyup soruyu sade Türkçeyle cevaplıyor ve **hangi maddeye dayandığını belirtiyor**, böylece cevabı kendiniz doğrulayabiliyorsunuz.

### Örnek

**Soru:** Bir yarıyılda kaç dersten çekilebilirim?

**Cevap:** Madde 24'e göre, bir yarıyıl içinde en çok bir dersten çekilme işlemi yapabilirsiniz. Öğreniminiz boyunca ise toplamda en fazla altı dersten çekilebilirsiniz.

### Nasıl çalışıyor?
Yönetmelik sayfası (oidb.metu.edu.tr)
↓  BeautifulSoup ile sadece yönetmelik metni çekiliyor (menü/footer eleniyor)
Ham metin (~40.000 karakter)
↓  1000 karakterlik parçalara bölünüyor (200 karakter örtüşmeli)
~45 parça
↓  Gemini embedding modeli ile vektörleştiriliyor
ChromaDB (diske kaydediliyor — bir kez işleniyor, her çalıştırmada değil)
↓  Her soruda en ilgili 4 parça getiriliyor
Retrieval
↓  Parçalar + soru prompt'a yerleştiriliyor
Gemini 2.5 Flash
↓
Madde referanslı, sade dilde cevap

### Tasarım kararları

**Sadece yönetmeliğe dayanır.** Prompt, modeli yalnızca getirilen yönetmelik metnine dayanmaya zorluyor. Yönetmelikte olmayan bir şey sorulursa cevap uydurmuyor — bilmediğini söyleyip kullanıcıyı Öğrenci İşleri'ne yönlendiriyor. Hukuki bir metinde uydurulmuş cevap, cevapsızlıktan daha kötüdür.

**Kaynağını gösterir.** Her cevapta hangi maddeye dayandığı belirtiliyor, kullanıcı orijinal metinden doğrulayabiliyor.

**Konu dışına çıkmaz.** Alakasız sorular, modelin genel bilgisinden cevaplanmak yerine reddediliyor.

**Bir kez işler, sonra diskten okur.** Embedding'ler `.chroma` klasörüne kaydediliyor. İlk çalıştırmada yönetmelik indirilip işleniyor, sonraki çalıştırmalarda mevcut vektör deposu yükleniyor — hem hızlı, hem tekrar API kotası harcamıyor.

### Teknolojiler

- **LangChain** — RAG zinciri
- **Google Gemini 2.5 Flash** — cevap üretimi
- **Gemini Embedding** — metin vektörleştirme
- **ChromaDB** — disk kalıcılığı olan vektör veritabanı
- **BeautifulSoup** — hedefli web scraping

### Kurulum

```bash
git clone https://github.com/alpozgenozgur/odtu-yonetmelik-rag.git
cd odtu-yonetmelik-rag

python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

`.env` dosyası oluşturun:
GOOGLE_API_KEY=buraya_kendi_anahtariniz

Ücretsiz API anahtarını [Google AI Studio](https://aistudio.google.com)'dan alabilirsiniz.

Çalıştırın:

```bash
python main.py
```

İlk çalıştırmada yönetmelik indirilip işlenir (birkaç saniye). Sonrasında anında açılır.

### Sınırlar

- Yönetmelik güncellenirse `.chroma` klasörünü silip yeniden çalıştırmak gerekir.
- Yalnızca **Lisans** Eğitim-Öğretim Yönetmeliği'ni kapsar. Lisansüstü, disiplin ve diğer yönetmelikler kapsam dışıdır.
- Yalnızca bilgilendirme amaçlıdır — resmi işlemler için Öğrenci İşleri'ne danışılmalıdır.

### Yol haritası

- [ ] Streamlit web arayüzü
- [ ] Diğer ODTÜ yönetmelikleri (ayrı koleksiyonlara yönlendirmeli)
- [ ] İngilizce yönetmelik desteği (çift dilli)
- [ ] Hugging Face Spaces'te canlı demo

---

**Hüseyin Özgür Alpözgen** — [GitHub](https://github.com/alpozgenozgur)
