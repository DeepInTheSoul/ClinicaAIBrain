import os
import streamlit as st
from dotenv import dotenv_values
from langchain_community.embeddings.baidu_qianfan_endpoint import QianfanEmbeddingsEndpoint
from langchain_community.llms.baidu_qianfan_endpoint import QianfanLLMEndpoint
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredMarkdownLoader, PyPDFLoader, UnstructuredFileLoader
import codecs
import csv
config = dotenv_values(".env")


QIANFAN_AK = "rYndCW8UyNrh7ZIxAmxG0w1X"
QIANFAN_SK = "KovKWoaJeKYeIQwLgOUxFof5KI1ggTRq"
embeddings = QianfanEmbeddingsEndpoint(model='bge-large-zh', qianfan_ak=QIANFAN_AK, qianfan_sk=QIANFAN_SK)

def rag_page():
    st.title("📚知识库管理")
    files = st.file_uploader("上传知识文件(目前支持txt、pdf、md文件，建议将word转换为pdf文件)：")
    if files is not None:
        filepath = os.path.join('file', files.name)
        if filepath.lower().endswith(".csv"):
            num = 0
            with codecs.open(filepath) as f:
                new_json = []
                for row in csv.DictReader(f, skipinitialspace=True):
                    data = {}
                    data['question'] = row['ask']
                    data['answer'] = row['answer']
                    data_str = str(data)
                    num = num + 1
                    new_json.append(data_str)
                    if num > 1000:
                        break
                st.markdown("正在把数据传至向量模型编码, 请耐心等待, 暂时先别切换界面")
                vector_db = Chroma.from_texts(new_json, embedding=embeddings, persist_directory="./chroma_db")
                vector_db.persist()
                st.markdown("---------------已将文件装载进知识库中--------------------")
        else:
                docs = file_loader(files)
                konwlwdge_vec_store(docs)
                st.markdown("---------------已将文件装载进知识库中--------------------")

def file_loader(file):
    filepath = os.path.join('file', file.name)
    with open(filepath, 'wb') as f:
        f.write(file.getbuffer())
    if filepath.lower().endswith(".md"):
        loader = UnstructuredMarkdownLoader(filepath)
        docs = loader.load()
    elif filepath.lower().endswith(".pdf"):
        loader = PyPDFLoader(filepath)
        docs = loader.load()
    elif filepath.lower().endswith(".txt"):
        loader = UnstructuredFileLoader(filepath,encoding='utf8')
        docs = loader.load()
    else:
        loader = UnstructuredFileLoader(filepath, mode="elements")
        docs = loader.load()
    return docs

def konwlwdge_vec_store(docs):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    splits = text_splitter.split_documents(docs)

    vector_db = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory="./chroma_db")
    vector_db.persist()



def format_docs(docs):
    return "\n\n".join(doc for doc in docs)



def rag_chain(llm):

    vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    retriever = vector_db.as_retriever()

    # 加载一个预先定义的提示生成器，用于生成检索问题。
    template = """基于以下已知信息，请专业地回答用户的问题。
                不要乱回答，如果无法从已知信息中找到答案，请诚实地告诉用户。
                已知内容:   
                {context}
                问题:
                {question}"""

    prompt = ChatPromptTemplate.from_template(template)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
    )
    return rag_chain


def generate_initial_answers(llm, question, k=1):
    """使用LLM生成k个初步答案"""
    answers = []
    for _ in range(k):
        answer = llm.invoke(question)
        answers.append(answer)
    return answers

def vectorize_texts(embedding_function, texts):
    """将文本向量化"""
    return [embedding_function.embed_query(text) for text in texts]

def average_vectors(vectors):
    """平均化向量"""
    if not vectors:
        return []
    avg_vector = [sum(dim) / len(vectors) for dim in zip(*vectors)]
    return avg_vector


from typing import List, Dict
import json

def filter_relevant_strings(llm, strings: List[str], question: str) -> Dict:
    """
    使用LLM过滤与问题相关的字符串，并返回一个包含相关字符串列表、过滤标志、解释和相关分数的List。

    参数:
        llm: 已初始化的语言模型实例。
        strings (List[str]): 要过滤的字符串列表。
        question (str): 需要回答的问题。

    返回:
        包含相关字符串列表、是否过滤、解释和相关分数的JSON。
    """
    # 构建过滤提示
    filter_prompt = """给定以下字符串列表使用:
{strings}

请根据以下问题过滤相关的字符串，并返回一个包含相关字符串列表、是否过滤、解释和相关分数的List:
问题: {question}
格式要求:
[
    "filtered_strings": "相关字符串1",
    "filtered": true/false,
    "explanation": "解释内容",
    "scores": 分数1
,

    "filtered_strings": "相关字符串2",
    "filtered": true/false,
    "explanation": "解释内容",
    "scores": 分数2
]"""

    # 将字符串列表格式化为一个多行字符串
    formatted_strings = "\n".join(strings)

    # 使用提示生成器创建提示
    prompt = filter_prompt.format(strings="["+formatted_strings+"]", question=question)

    # 使用LLM生成过滤后的字符串
    response = llm.invoke(prompt)

    # 解析JSON响应
    # result = json.loads(response)

    return response.strip("'```json\n'")


def filter_result(llm, strings: List[str], question: str) -> Dict:
    """
    使用LLM过滤与问题相关的字符串，并返回一个包含相关字符串列表、过滤标志、解释和相关分数的List。

    参数:
        llm: 已初始化的语言模型实例。
        strings (List[str]): 要过滤的字符串列表。
        question (str): 需要回答的问题。

    返回:
        包含相关字符串列表、是否过滤、解释和相关分数的JSON。
    """
    # 构建过滤提示
    filter_prompt = """给定以下字符串列表使用:
{strings}

请根据以下问题过滤相关的字符串，并返回一个包含相关字符串列表、是否过滤、解释和相关分数的List:
问题: {question}
格式要求:
[
    "filtered_strings": "相关字符串1",
    "filtered": true/false,
    "explanation": "解释内容",
    "scores": 分数1
,

    "filtered_strings": "相关字符串2",
    "filtered": true/false,
    "explanation": "解释内容",
    "scores": 分数2
]"""

    # 将字符串列表格式化为一个多行字符串
    formatted_strings = "\n".join(strings)

    # 使用提示生成器创建提示
    prompt = filter_prompt.format(strings="["+formatted_strings+"]", question=question)

    # 使用LLM生成过滤后的字符串
    response = llm.invoke(prompt)

    # 解析JSON响应
    # result = json.loads(response)

    return response.strip("'```json\n'")


import random
import openai
import json

def generate_prompt_results(question, answers):
    # 构建评分提示
    prompt = f"""
我将提供一个问题和一些答案。请根据以下标准对每个答案进行评分：

1. **流畅性**：答案是否语法正确、表达清晰、易于理解。
2. **合理性**：答案是否与问题相关、提供的信息是否准确和有用。
3. **细节丰富度**：答案是否提供了充分的细节和解释。
4. **结构性**：答案是否有良好的逻辑结构，条理清晰。
5. **专业性**：答案是否显示出对主题的深刻理解和专业知识。

请为每个答案分别给出每项标准的评分（满分10分），并提供简短的解释。最终输出格式应如下：

问题:
{question}
"""
    for i, answer in enumerate(answers, start=1):
        prompt += f"""
答案{i}:
{answer}
流畅性评分: {{fluency_score{i}}}/10
合理性评分: {{relevance_score{i}}}/10
细节丰富度评分: {{detail_score{i}}}/10
结构性评分: {{structure_score{i}}}/10
专业性评分: {{expertise_score{i}}}/10
解释: {{explanation{i}}}
"""

    prompt += f"""
请根据以上标准对以下答案进行评分：

问题: {question}
答案:
"""
    for i, answer in enumerate(answers, start=1):
        prompt += f"答案{i}: {answer}\n"

    return prompt


import re

def result_filter(answer_text):

    # 定义正则表达式来匹配分数
    pattern = r"答案(\d+):\s*流畅性评分: (\d+)/10\s*合理性评分: (\d+)/10\s*细节丰富度评分: (\d+)/10\s*结构性评分: (\d+)/10\s*专业性评分: (\d+)/10"

    # 使用正则表达式查找所有匹配
    matches = re.findall(pattern, answer_text)

    # 初始化分数字典
    scores = {}

    # 计算每个答案的总分
    for match in matches:
        answer_id = int(match[0])
        fluency = int(match[1])
        relevance = int(match[2])
        detail = int(match[3])
        structure = int(match[4])
        expertise = int(match[5])

        total_score = fluency + relevance + detail + structure + expertise
        scores[answer_id] = total_score

    # 获取所有分数的列表
    score_values = list(scores.values())

    # 找到最大值
    max_score = max(score_values)

    # 找到所有最大值的索引（答案编号）
    max_score_indices = [i for i, score in scores.items() if score == max_score]

    # 选择最小的索引
    best_answer_index = min(max_score_indices)

    print(f"最佳答案是答案{best_answer_index}，总分为{max_score}")

    # 打印每个答案的总分
    for answer_id, total_score in scores.items():
        print(f"答案{answer_id}的总分: {total_score}")

    return best_answer_index
def rag_chain_v2(llm, question):
    # 初始化向量数据库和检索器
    vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    # retriever = vector_db.as_retriever()

    # 生成k个初步答案
    initial_answers = generate_initial_answers(llm, question)

    # 将问题和k个答案向量化
    question_vector = embeddings.embed_query(question)
    answer_vectors = embeddings.embed_documents(initial_answers)

    # 将问题向量与答案向量相加并平均化
    combined_vectors = [question_vector] + answer_vectors
    avg_vector = average_vectors(combined_vectors)

    # 使用平均化后的向量进行检索 检索出2个答案
    context_docs = vector_db.similarity_search_by_vector(embedding=avg_vector,k=5)
    #精排 大模型筛选
    document_list = json.loads(filter_relevant_strings(llm, [i.page_content for i  in context_docs ],question))

    # 过滤并提取 "filtered": true 对应的 "filtered_strings"
    filtered_strings = [item["filtered_strings"] for item in document_list if item["filtered"]]

    formatted_context = format_docs(filtered_strings)
    result = []
    for _ in range(2):
    # 构建提示生成器
        template_fina = """基于以下已知信息，请专业地回答用户的问题。
        不要乱回答，如果无法从已知信息中找到答案，请诚实地告诉用户。
        已知内容:
        {context}
        问题:
        {question}"""

        # 使用提示生成器创建提示
        prompt = template_fina.format(context=formatted_context, question=question)

        # 使用LLM生成过滤后的字符串
        response = llm.invoke(prompt)
        result.append(response)
    # 解析JSON响应


    response = llm.invoke(generate_prompt_results(question=question,answers=result))
    return result[result_filter(response)-1]
    # selected_text = random.choice(result)
    # return selected_text


# 构建RAG链

#先利用llm生成 k个答案
#利用bge 将k个答案 和 1 个问题进行向量化
#将k+1 个 答案和问题向量相加
#然后将这k+1个向量的和进行平均化
#将所得向量进行 向量召回

#精排阶段使用大模型进行进行 过滤留下nge


#最后结果进行筛选 先用一个大模型生成很多个答案 然后再利用大模型对结果进行筛选

#第一种方法 先用langchain的rag接口进行改性
#然后对langchain的rag接口进行精排
