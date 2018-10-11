import tensorflow as tf
import tensorflow_hub as hub
import os


# os.environ['TFHUB_CACHE_DIR'] = "tf-hub_modules/text_encoder/1fb57c3ffe1a38479233ee9853ddd7a8ac8a8c47"
embed = hub.Module("tf-hub_modules/text_encoder/1fb57c3ffe1a38479233ee9853ddd7a8ac8a8c47/1fb57c3ffe1a38479233ee9853ddd7a8ac8a8c47")
embeddings = embed([
"The quick brown fox jumps over the lazy dog.",
"I am a sentence for which I would like to get its embedding"])
sess = tf.Session()
sess.run(tf.initialize_all_variables())
sess.run(tf.initialize_all_tables())
print(sess.run(embeddings))