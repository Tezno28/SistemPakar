from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings

loader = PyPDFLoader("dataset.pdf")
data = loader.load()  # entire PDF is loaded as a single Document

#data
len(data)

from langchain.text_splitter import RecursiveCharacterTextSplitter

# split data
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
docs = text_splitter.split_documents(data)


print("Total number of documents: ",len(docs))
docs[7]

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from dotenv import load_dotenv
load_dotenv() 

#Get an API key: 
# Head to https://ai.google.dev/gemini-api/docs/api-key to generate a Google AI API key. Paste in .env file

# Embedding models: https://python.langchain.com/v0.1/docs/integrations/text_embedding/

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector = embeddings.embed_query("hello, world!")
vector[:5]
vector

vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

retrieved_docs = retriever.invoke("Apa itu dinamo?")
len(retrieved_docs)

print(retrieved_docs[5].page_content)

from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro",temperature=0.3, max_tokens=500)
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Set up the system prompt in Indonesian
system_prompt = (
    "Anda adalah asisten AI khusus untuk konsultasi kerusakan dinamo motor. "
            "Gunakan konteks yang diberikan untuk menjawab pertanyaan dengan akurat, singkat dan detail. "
            "Berikan solusi praktis dan tidak terlalu menggunakan penjelasan yang panjang dan aman untuk setiap permasalahan. "
            "Jika informasi tidak tersedia, sampaikan dengan jelas dan tidak terlalu panjang. "
            "Fokus pada keselamatan dan solusi teknis. "
    "\n\n"
    "{context}"
)


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)


question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)
response = rag_chain.invoke({"input": "Apa itu dinamo motor?"})

print(response["answer"])

# Retrieve documents with similarity scores
retrieved_docs_with_scores = vectorstore.similarity_search_with_score("Apa itu dinamo motor?", k=10)

# Print retrieved documents with their similarity scores
for i, (doc, score) in enumerate(retrieved_docs_with_scores):
    print(f"Document {i+1}:")
    print(f"Similarity Score: {score}")
