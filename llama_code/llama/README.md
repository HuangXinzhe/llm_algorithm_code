# llama代码
代码来源于huggingface的transformers库  
地址：https://github.com/huggingface/transformers/tree/main/src/transformers/models/llama  

LLaMA主要代码在modeling_llama.py中，包括LLaMA模型的定义、前向传播、损失函数等。
llama
├── configuration_llama.py  # LLaMA模型的配置文件
├── modeling_llama.py  # LLaMA模型的定义、前向传播、损失函数等
├── modeling_flax_llama.py  # LLaMA模型的Flax版本，用于JAX训练
├── tokenization_llama_fast.py  # LLaMA模型的fast tokenizer
├── tokenization_llama.py  # LLaMA模型的tokenizer
├── convert_llama_weights_to_hf.py  # 将LLaMA模型的权重转换为huggingface的transformers库的权重
└── README.md  # LLaMA模型的说明文档
