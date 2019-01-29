import tensorflow as tf
import os


tf.reset_default_graph()

v1 = tf.get_variable('v1', shape=[3], initializer=tf.zeros_initializer)
v2 = tf.get_variable('v2', shape=[5], initializer=tf.zeros_initializer)

inc_v1 = v1.assign(v1 + 1)
des_v2 = v2.assign(v2 - 1)

init = tf.global_variables_initializer()

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    inc_v1.op.run()
    des_v2.op.run()
    path = saver.save(sess, os.getcwd() + '/model.pb')
    print('Save to:{}'.format(str(path)))

