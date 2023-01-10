#不带jupyter的docker image docker.wanfangdata.com.cn/wfk8s/fasttext-faiss:base
import faiss
import sent2vec
import random
import sys

#获取faiss模型文件名
faiss_model_name = sys.argv[1]
print(f'read faiss model name:{faiss_model_name}')

#获取fasttext模型文件名
fasttext_model_name = sys.argv[2]
print(f'read fasttext model name:{fasttext_model_name}')

#获取eval样本文件名
eval_file_name = sys.argv[3]
print(f'read eval file name:{eval_file_name}')

index = faiss.read_index(faiss_model_name)
model = sent2vec.Sent2vecModel()
model.load_model(fasttext_model_name)
sentences = []
with open(eval_file_name) as f:
    for line in f:
        sentences.append(line.strip())
        
sentences = list(set(sentences))
print(f'导入faiss的句子总量为:{len(sentences)}条\n')
sentence_embeddings = model.embed_sentences(sentences)
print(f'embedding shape:{sentence_embeddings.shape}\n')
print(f'添加句子向量到faiss\n')
index.add(sentence_embeddings)
print(f'添加句子向量到faiss完成\n')
print(f'faiss中的文档总量为:{index.ntotal}条\n')

#随机抽取500条样本进行计算召回率
samples = 500
random_int_list = []
for _ in range(samples):
    random_index = random.randint(0, 101620)
    while random_index in random_int_list:
        random_index = random.randint(0, 101620)
    random_int_list.append(random_index)

#计算召回率
r1_acc = 0
r10_acc = 0
k = 10
process_count = 0
#设置当前nprobe参数
index.nprobe = 8
for random_index in random_int_list:
    process_count += 1
    if process_count % 100 == 0:
        print(f'当前处理第{process_count}条\n')
    sentence = sentences[random_index]
    #以下是随机删除一个词后的处理
    sentence_array = sentence.split(' ')
    sentence_array.pop(random.randrange(len(sentence_array)))
    sentence = ' '.join(sentence_array) 
    query = model.embed_sentence(sentence)
    D, I = index.search(query, k)
    result = I[0]
    if random_index in result:
        # 如果处于第一个
        if random_index == result[0]:
            r1_acc += 1
        r10_acc += 1

recall_1 = r1_acc / samples
recall_10 = r10_acc / samples

print(f'r1的准确率为: {recall_1}\n')
print(f'r10的准确率为: {recall_10}\n')


import json
metadata = {
        'outputs': [{
          'type': 'table',
          'format': 'csv',
          'storage': 'inline',
          'header': ['recall_1','recall_10'],
          'source': str(recall_1) + "," + str(recall_10)
        }]
    }

with open('/mlpipeline-ui-metadata.json', 'w') as f:
	json.dump(metadata, f)
