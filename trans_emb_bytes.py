import sent2vec
import sys
import struct
from datetime import datetime

#获取模型文件名
model_name = sys.argv[1]
print(f'read model name:{model_name}')
model = sent2vec.Sent2vecModel()
model.load_model(model_name)

#获取文件名
file_name = sys.argv[2]
write_file_name = sys.argv[3]
batch_verbose = int(sys.argv[4])

print(f'read faiss samples text file:{file_name},write faiss samples bytes file:{write_file_name}')
#读取faiss训练样本
process_count = 0
start_time = datetime.now()
with open(file_name,'r',encoding="utf-8") as f,open(write_file_name,'wb') as w:
   for line in f:
     process_count += 1
     # shape (1,128)
     emb = model.embed_sentence(line)[0]
     for vec in emb:
        # little endian
        pack = struct.pack('<f',vec)
        w.write(pack)
     if process_count % batch_verbose == 0:
        print(f'process line:{process_count}')

end_time = datetime.now()
use_time = end_time - start_time
print(f'process complate,total_count:{process_count},cost:{use_time.seconds}s')
