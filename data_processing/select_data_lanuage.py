from langdetect import detect
from langdetect import detect_langs

s1 = "汉语是世界上最优美的语言，正则表达式是一个很有用的工具"
s2 = "正規表現は非常に役に立つツールテキストを操作することです"
s3 = "あアいイうウえエおオ"
s4 = "정규 표현식은 매우 유용한 도구 텍스트를 조작하는 것입니다"
s5 = "Regular expression is a powerful tool for manipulating text."
s6 = "Regular expression 正则表达式 あアいイうウえエおオ 정규 표현식은"

print(detect(s1))
print(detect(s2))
print(detect(s3))
print(detect(s4))
print(detect(s5))
print(detect(s6))     # detect()输出探测出的语言类型
print(detect_langs(s6))    # detect_langs()输出探测出的所有语言类型及其所占的比例
