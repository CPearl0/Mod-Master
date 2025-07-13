from langchain_community.chat_models import ChatZhipuAI
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain_core.vectorstores import InMemoryVectorStore
from bs4.filter import SoupStrainer

from modmaster.config import config

chat_model = ChatZhipuAI(
    model=config.get("zhipu", "model"),
    temperature=0.95,
)

embeddings = ZhipuAIEmbeddings(
    api_key=config.get("zhipu", "api-key"),
    model="embedding-3",
)

vector_store = InMemoryVectorStore(embeddings)


def load_documents():
    loader = WebBaseLoader(
        (
            "https://docs.neoforged.net/docs/1.21.1/concepts/registries",
            "https://docs.neoforged.net/docs/1.21.1/items/",
            "https://docs.neoforged.net/docs/1.21.1/blocks/",
            "https://docs.neoforged.net/docs/1.21.1/blocks/states",
            "https://docs.neoforged.net/docs/1.21.1/resources/",
        ),
        bs_kwargs=dict(
            parse_only=SoupStrainer(
                name=("h2", "h3", "p", "li", "div")
            )
        )
    )
    return loader.load()


class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"], 6)
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n--------\n".join(
        doc.page_content for doc in state["context"])

    prompt = ChatPromptTemplate([
        ("system", "你是一位Minecraft模组开发专家，请协助用户完成模组开发"),
        (
            "user",
            """
            使用以上下文来回答用户的问题。如果你不知道答案，就说你不知道。总是使用中文回答。
            问题: {question}
            可参考的上下文：
            --------
            {context}
            --------
            """
        )
    ])

    messages = prompt.invoke(
        {"question": state["question"], "context": docs_content})
    response = chat_model.invoke(messages)
    return {"answer": response.content}


def main():
    docs = load_documents()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200, chunk_overlap=200)
    all_splits = text_splitter.split_documents(docs)

    BATCH_SIZE = 64
    for i in range(0, len(all_splits), BATCH_SIZE):
        batch = all_splits[i:i+BATCH_SIZE]
        vector_store.add_documents(batch)

    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()

    question = input()
    response = graph.invoke(
        {"question": question, "context": [], "answer": ""})
    print(response["answer"])
