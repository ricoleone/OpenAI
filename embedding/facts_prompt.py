from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores.chroma import Chroma
from langchain.chat_models import ChatOpenAI
from facts_rdndnt_fltr_rtrvr import RedundantFilterRetriever
import langchain

langchain.debug = True

load_dotenv()

embeddings=OpenAIEmbeddings()

chat = ChatOpenAI()

db = Chroma(
    embedding_function=embeddings,
    persist_directory="emb",
)

retriever = RedundantFilterRetriever(
    embeddings=embeddings,
    chroma=db
)

chain = RetrievalQA.from_chain_type(
    llm=chat,
    retriever=retriever,
    chain_type="stuff"
)

results = chain.run("What is an interesting fact about the English language?")

for result in results:
    print("\n")
    print(result.page_content)