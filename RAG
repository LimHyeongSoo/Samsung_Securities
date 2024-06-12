# -*- coding: utf-8 -*-
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chainlit as cl
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain.text_splitter import TokenTextSplitter
from langchain.schema.runnable import RunnablePassthrough

# 텍스트 분할기 설정
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=150,
    chunk_overlap=40,
    encoding_name='cl100k_base'
)



@cl.on_chat_start
async def on_chat_start():
    files = None

    # Wait for the user to upload a file
    while files is None:
        files = await cl.AskFileMessage(
            content="샘숭증권 챗봇에 오신 것을 환영합니다!우선 분석하길 원하는 PDF 파일을 업로드해주세요!",
            accept=["application/pdf"],
            max_size_mb=20,
            timeout=180,
        ).send()

    file = files[0]
    print(file)

    msg = cl.Message(content=f"`{file.name}`파일을 처리중입니다....")
    await msg.send()

    # Read the PDF file
    loader = PyPDFLoader(file.path)
    pages = loader.load()

    # Split the text into chunks
    texts = text_splitter.split_documents(pages)

    # Create a FAISS vector store
    embeddings_model = HuggingFaceEmbeddings(
        model_name='./ko-sroberta-multitask',  # 로컬 경로로 설정
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True},
    )
    docsearch = await cl.make_async(FAISS.from_documents)(texts, embeddings_model)

    
    # base_retriever = docsearch.as_retriever()

    

    message_history = ChatMessageHistory()

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    llm = ChatOllama(model="Llama-3-Open-Ko-8B-Q8_0:latest")

    system_template = '''
        너는 주어진 문서를 바탕으로 사실에 기반한 금융 전문가야.
        3Q22는 2022년도 3분기를 의미해. 1Q21은 2021년도 1분기를 의미해. 
        따라서 Q는 분기를 의미해.
        문서의 모든 내용을 정확히 이해해서 관련 정보를 철저히 조사한 후 답변해줘.
        제공된 문서의 내용에 질문이 연관되어 있다면, 문서를 기반으로 상세하고 명확한 답변을 제공해야 해.
        질문에 특정 연도가 언급되면, 그 연도의 정확한 데이터를 제공해야 해.
        문서 외의 정보를 기반으로 답을 생성하지 않고, 정보를 찾을 수 없는 경우에는 반드시 '해당 문서에서 내용을 찾을 수 없습니다.'라고 알려줘.
        문서와 연관되지 않거나 불분명한 질문에 대해서는 '내용을 찾을 수 없습니다.'라고 답변해줘.
    '''

    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template("문맥: {context}\n\n질문: {question}\n\n한국어로 답변해 주세요:")
    ]
    chat_prompt = ChatPromptTemplate.from_messages(messages)

    # chain = LLMChain(
    #     llm=llm,
    #     prompt=chat_prompt,
    #     output_parser=StrOutputParser(),
    # )
    
    parent_store = InMemoryStore()
    child_splitter = TokenTextSplitter(chunk_size=128, chunk_overlap=0)
    parent_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=0)
    
    # vectorstore = FAISS()
    
    parent_retriever = ParentDocumentRetriever(
    vectorstore=docsearch,
    docstore=parent_store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
    )
    
    parent_retriever.add_documents(texts, ids=None)
    
    
    multi_parent_retriever = MultiQueryRetriever.from_llm(
    retriever=parent_retriever, llm=llm
    )

    multi_parent_chain = (
        {"context": multi_parent_retriever, "question": RunnablePassthrough()}
        | chat_prompt
        | llm
        | StrOutputParser()
    )
    

    cl.user_session.set("chain", multi_parent_chain)
    cl.user_session.set("retriever", parent_retriever)
    cl.user_session.set("memory", memory)

    msg.content = f"`{file.name}`처리 완료하였습니다! 제게 질문을 해주세요!"
    await msg.update()

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")
    retriever = cl.user_session.get("retriever")
    memory = cl.user_session.get("memory")

    query = message.content
    # docs = retriever.invoke(query)
    # print("확인용 : ",docs)
    # formatted_docs = "\n".join([doc.page_content for doc in docs])
    # print("확인용 : ",formatted_docs)

    result = await chain.ainvoke({
    "context" : retriever,
    "question": query,
    })
    
    # result = await chain.ainvoke(query)

    
    print("결과물 : ",result)

    # 필요한 부분만 출력
    # if isinstance(result, str):
    #     # 문자열 형식인 경우 그대로 출력
    #     response = result
    # elif isinstance(result, dict) and 'text' in result:
    #     # 딕셔너리 형식인 경우 'text' 키의 값만 출력
    #     response = result['text']
    # else:
    #     response = str(result)

    await cl.Message(content=f'질문에 대한 답변을 말씀드리겠습니다. \n{result}').send()

# Chainlit 서버 실행 명령
if __name__ == "__main__":
    cl.run()
