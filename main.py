import os
import bs4
import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

URL = 'https://oidb.metu.edu.tr/tr/orta-dogu-teknik-universitesi-lisans-egitim-ogretim-yonetmeligi'
PERSIST_DIR = './.chroma'

# ---------- Sayfa ayarları ----------
st.set_page_config(
    page_title="ODTÜ Yönetmelik Asistanı",
    page_icon="📘",
    layout="centered"
)

st.title("📘 ODTÜ Yönetmelik Asistanı")
st.caption("ODTÜ Lisans Eğitim-Öğretim Yönetmeliği'ne dayanarak sorularınızı yanıtlar ve hangi maddeye dayandığını belirtir.")


# ---------- RAG kurulumu (bir kez çalışır, önbelleğe alınır) ----------
@st.cache_resource
def setup_rag():
    embeddings = GoogleGenerativeAIEmbeddings(model='models/gemini-embedding-001')
    llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash', temperature=0)

    vectorstore = Chroma(
        collection_name='metu-yonetmelik',
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
    )

    # Boşsa yönetmeliği indir ve işle
    if vectorstore._collection.count() == 0:
        loader = WebBaseLoader(
            web_path=URL,
            bs_kwargs=dict(
                parse_only=bs4.SoupStrainer(
                    class_=lambda c: c and 'field--name-body' in c
                )
            )
        )
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        vectorstore = Chroma.from_documents(
            documents=splits,
            collection_name='metu-yonetmelik',
            embedding=embeddings,
            persist_directory=PERSIST_DIR
        )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    prompt = ChatPromptTemplate.from_messages([
        ('human', """Sen ODTÜ Lisans Eğitim-Öğretim Yönetmeliği asistanısın. Öğrencilerin sorularını, aşağıdaki yönetmelik metnine dayanarak yanıtla.

Kurallar:
- Sadece verilen yönetmelik metnine dayan. Kendi bilgini kullanma, uydurma.
- Cevabın yönetmelikte yoksa "Bu konuda yönetmelikte bilgi bulamadım, Öğrenci İşleri'ne danışmanızı öneririm" de.
- Hangi maddeye dayandığını mutlaka belirt (örn: "Madde 24'e göre...").
- Yönetmelik dilini sadeleştir, öğrencinin anlayacağı şekilde açıkla.
- Yönetmelik dışı sorulara "Ben sadece ODTÜ lisans yönetmeliği hakkında yardımcı olabilirim" de.
- Kısa ve net cevapla.

Soru: {question}

Yönetmelik metni:
{context}

Cevap:""")
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {'context': retriever | format_docs, 'question': RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


# ---------- Uygulama ----------
with st.spinner("Yönetmelik yükleniyor..."):
    rag_chain = setup_rag()

# Sohbet geçmişini sakla
if "messages" not in st.session_state:
    st.session_state.messages = []

# Örnek sorular
with st.expander("💡 Örnek sorular"):
    st.markdown("""
    - Bir yarıyılda kaç dersten çekilebilirim?
    - Sınamalı öğrenci ne demek?
    - Mezun olmak için not ortalamam kaç olmalı?
    - NA notu ne zaman verilir?
    - Azami öğrenim sürem kaç yıl?
    - Kaç kez ders tekrarı yapabilirim?
    - Şeref öğrencisi olmak için ne gerekiyor?
    """)

# Geçmiş mesajları göster
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Kullanıcı girişi
if question := st.chat_input("Yönetmelikle ilgili sorunuzu yazın..."):
    # Kullanıcı mesajını göster ve kaydet
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Cevabı akarak göster
    with st.chat_message("assistant"):
        response = st.write_stream(rag_chain.stream(question))
    st.session_state.messages.append({"role": "assistant", "content": response})

# Alt bilgi
st.divider()
st.caption("⚠️ Bu asistan bilgilendirme amaçlıdır. Resmi işlemler için Öğrenci İşleri'ne danışınız.")