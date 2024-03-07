# 本地知识库大模型 langchain + LLaMa + Custom Agent

## 准备工作

1. 本例采用 13B 的 LLama-base 的模型，用户可根据需要在 `models/custom_llm.py` 中替换
1. 使用在线搜索需要在 `models/custom_search.py` 中设置你的 `RapidAPIKey = ""`，可在[这里](https://rapidapi.com/microsoft-azure-org-microsoft-cognitive-services/api/bing-web-search1)申请

代码checkout下来后，执行

```
python cli.py
```

即可实现全过程

## 代码文件结构

|-- docs  # 本地文档数据
|-- models
|   |-- chineses_text_splitter.py  # 文本切分
|   |-- custom_agent.py  # 自定义Agent，如果本地没有相关数据，可以使用在线检索
|   |-- custom_llm.py  # 自定义本地模型
|   |-- custom_search.py  # 自定义在线检索
|   |-- local_knowledge.py  # 本地向量库构建
|-- vector_store  # FAISS向量数据库工具
|-- README.md
|-- requirements.txt
|-- cli.py  # 主程序




