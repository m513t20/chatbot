
import numpy as np
print('hi')
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from  pathlib import Path
from src.train_model import train_model
from src.consts import questions,answers
from tqdm import tqdm
tf.get_logger().setLevel('ERROR')

bert_preprocess_model=None
bert_model=None

dataset=[]
def load_BERT():
  global bert_model
  global bert_preprocess_model
  global dataset
  global questions
  global answers
  bert = "https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/4"
  bpre = "https://tfhub.dev/tensorflow/bert_multi_cased_preprocess/3"

  bert_preprocess_model=hub.KerasLayer(bpre)
  bert_model=hub.KerasLayer(bert)

  print('tokenizing data')
  dataset=[]
  for i in tqdm(range(len(questions))):
    q_emb=bert_model(bert_preprocess_model(questions[i:i+1]))['pooled_output'].numpy()
    a_emb=bert_model(bert_preprocess_model(answers[i:i+1]))['pooled_output'].numpy()
    dataset.append([np.array(q_emb[0]),np.array(a_emb[0])])
    tf.keras.backend.clear_session(free_memory=True)


def start_chatbot(model):
  global bert_model
  global bert_preprocess_model
  global dataset
  global questions
  global answers
  model=tf.keras.models.load_model(model)
  while True:
    question=str(input("Ask question or write !q to quit\n"))
    if question=="!q":
      break
    question=[question]
    emb1=bert_model(bert_preprocess_model(question[0:1]))["pooled_output"].numpy()[0]
    p=[]
    dataset=np.array(dataset)
    for i in range(dataset.shape[0]):
      emb2=dataset[i,1]
      emb3=np.concatenate([emb1,emb2])
      p.append([i,model.predict(np.expand_dims(emb3,axis=0))[0,0]])
    p=np.array(p)
    print(p)
    answ=np.argmax(p[:,1])
    print(question[0],':',answers[answ])

if __name__ == "__main__":
  print('start')
  load_BERT()
  model=Path('.')/'model.keras'
  if not model.exists():
    train_model(model,dataset,bert_preprocess_model,bert_model)

  start_chatbot(model)