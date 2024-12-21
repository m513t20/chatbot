
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

def train_model(path,dataset,bert_preprocess_model,bert_model):
  dataset=np.array(dataset)
  X,Y=[],[]
  for i in range(dataset.shape[0]):
    for j in range(dataset.shape[0]):
      X.append(np.concatenate([dataset[i,0,:],dataset[j,1,:]],axis=0))
      if i==j:
        Y.append(1)
      else:
        Y.append(0)
  X=np.array(X)
  Y=np.array(Y)

  model=tf.keras.models.Sequential()
  model.add(tf.keras.layers.InputLayer(input_shape=(1536,)))
  model.add(tf.keras.layers.Dense(100,activation='selu'))
  model.add(tf.keras.layers.Dense(1,activation='sigmoid'))
  es=tf.keras.callbacks.EarlyStopping(monitor='auc',mode='MAX',patience=10,restore_best_weights=True)

  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),loss='binary_crossentropy',metrics=[tf.keras.metrics.AUC(curve='pr',name='auc')])
  model.fit(X,Y,epochs=3000,class_weight={0:1,1:np.sqrt(Y.shape[0]-1)})
  model.save(path)