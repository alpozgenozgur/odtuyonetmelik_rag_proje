import bs4
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import os

load_dotenv()

llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash', temperature=0)

embeddings = GoogleGenerativeAIEmbeddings(model='models/gemini-embedding-001')

URL = 'https://oidb.metu.edu.tr/tr/orta-dogu-teknik-universitesi-lisans-egitim-ogretim-yonetmeligi'
PERSIST_DIR = './.chroma'


def ingest():
    loader = WebBaseLoader(
        web_path=URL,
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=lambda c: c and 'field--name-body' in c
            )
        )
    )
    docs = loader.load()
    print(f"Çekilen karakter sayısı: {len(docs[0].page_content)}")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    print(f"Parça sayısı: {len(splits)}")

    return Chroma.from_documents(
        documents=splits,
        collection_name='metu-yonetmelik',
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )


# Bir kez embed et, sonraki çalıştırmalarda kayıtlıdan oku
vectorstore = Chroma(
    collection_name='metu-yonetmelik',
    persist_directory=PERSIST_DIR,
    embedding_function=embeddings,
)

if vectorstore._collection.count() == 0:
    print("İlk çalıştırma: yönetmelik indiriliyor ve işleniyor...")
    vectorstore = ingest()
else:
    print("Kayıtlı veritabanı kullanılıyor.")

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


rag_chain = (
    {'context': retriever | format_docs, 'question': RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


if __name__ == '__main__':
    print("\nODTÜ Yönetmelik Asistanı (çıkmak için 'q')\n")
    while True:
        soru = input("Soru: ")
        if soru.lower() in ('q', 'quit', 'exit'):
            break
        print("\nCevap: ", end='')
        for chunk in rag_chain.stream(soru):
            print(chunk, end='', flush=True)
        print("\n" + "-" * 50 + "\n")