import tensorflow as tf
import json
tf.enable_eager_execution()

d = tf.data.TFRecordDataset('generated_features/train_feature_file.fea')

num_samples = 0

l = []
for ele in d:
    num_samples += 1
    l.append(tf.train.Example.FromString(ele.numpy()))
    # print(tf.train.Example.FromString(ele.numpy()))
print(num_samples)  # 88786
# json.dump(l, open('train_feature_file.json', 'w', encoding='utf-8'))
