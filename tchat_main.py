from langchain.chat_models import ChatOpenAI
from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain
from langchain.memory import ConversationSummaryMemory, FileChatMessageHistory
from dotenv import load_dotenv

load_dotenv()

chat = ChatOpenAI()

#Note: as of 11/2023, ConversationSummaryMemory does not work as well with FileChatMessageHistory as does ConversationBufferMemory

memory = ConversationSummaryMemory(
    #chat_memory= FileChatMessageHistory("messages.json"),
    memory_key="messages",
    return_messages=True,
    llm=chat
    )

prompt = ChatPromptTemplate(
    input_variables=["content", "messages"],
    messages=[
        MessagesPlaceholder(variable_name="messages"),
        HumanMessagePromptTemplate.from_template("{content}")
    ]
)
chain = LLMChain(
    llm=chat,
    prompt = prompt,
    memory=memory,
    verbose=True
)

while True:
    content = input(">> ")
    result = chain({"content":content})
    print(result["text"])