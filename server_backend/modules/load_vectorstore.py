import os 
import time 
from pathlib import Path 
from dotenv import load_dotenv 
from tqdm.auto import tqdm 
from pinecone import Pinecone,ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def clean_text(text):
    return text.encode("utf-8", "ignore").decode("utf-8").strip()

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = "us-east-1"
PINECONE_INDEX_NAME = "medical-index"

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

UPLOAD_DIR = "./uploaded_docs"
os.makedirs(UPLOAD_DIR,exist_ok=True)


#initialize pinecone instance 
pc = Pinecone(api_key=PINECONE_API_KEY)
spec = ServerlessSpec(cloud="aws",region=PINECONE_ENV)

existing_indexes = [i["name"] for i in pc.list_indexes()]
if PINECONE_INDEX_NAME in existing_indexes:
    print("Deleting old index...")
    pc.delete_index(PINECONE_INDEX_NAME)
    while PINECONE_INDEX_NAME in [i["name"] for i in pc.list_indexes()]:
        time.sleep(1)
print("creating new index")  
pc.create_index(
    name=PINECONE_INDEX_NAME,
    dimension=3072,   
    metric="dotproduct",
    spec=spec
)      

while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
        time.sleep(1)

index = pc.Index(PINECONE_INDEX_NAME)        


# load,split,embed and upsert pdf docs content 

def load_vectorstore(uploaded_files):
    embed_model = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key = GOOGLE_API_KEY
    )
    file_paths=[]

    #1. upload 
    for file in uploaded_files:
        save_path = Path(UPLOAD_DIR)/file.filename
        with open(save_path,"wb") as f:
            f.write(file.file.read())
        file_paths.append(str(save_path))
    #2. split
    for file_path in file_paths    :
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100)
        chunks = splitter.split_documents(documents)

        #chunking 
        texts = [clean_text(chunk.page_content)[:5000] for chunk in chunks]
        metadata = [chunk.metadata for chunk in chunks]
        ids = [f"{Path(file_path).stem}-{i}" for i in range(len(chunks))]


        # 3. Embedding 
        print(f"Embedding Chunks")
        embedding = embed_model.embed_documents(texts)

        # 4. Upsert 

        print("Upserting embeddings")

        with tqdm(total=len(embedding),desc = "Upserting to Pinecone") as progress:
            index.upsert(vectors=zip(ids,embedding,metadata))
            progress.update(len(embedding))

        print(f"Upload complete for {file_path}")    


