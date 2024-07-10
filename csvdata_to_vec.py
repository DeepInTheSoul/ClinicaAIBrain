import json
import codecs
import csv
import os

from langchain_community.embeddings import QianfanEmbeddingsEndpoint
from langchain_community.vectorstores.chroma import Chroma

os.environ["QIANFAN_AK"] = "G5NhjZhLEdw53lVZKeQM3eaD"
os.environ["QIANFAN_SK"] = "Hxkgkd6iee9FMLVucKnTUNg59jw3JDWK"
# embeddings=QianfanEmbeddingsEndpoint(model='bge-large-zh')
embeddings=QianfanEmbeddingsEndpoint(model='bge-large-zh')

num = 0
with codecs.open('./file/内科.csv') as f:
    new_json=[]
    for row in csv.DictReader(f, skipinitialspace=True):
        data={}
        data['question']=row['ask']
        data['answer'] = row['answer']
        data_str = str(data)
        num  = num + 1
        new_json.append(data_str)
        if num > 1000:
            break
    print("开始编码")
    vector_db = Chroma.from_texts(new_json, embedding=embeddings, persist_directory="./chroma_db")
    vector_db.persist()
    print("医疗数据已加载进知识库")


