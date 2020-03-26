{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import tensorflow as tf\n",
    "import numpy as np\n",
    "import os, sys"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "ename": "AttributeError",
     "evalue": "module 'tensorflow' has no attribute 'reset_default_graph'",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mAttributeError\u001b[0m                            Traceback (most recent call last)",
      "\u001b[0;32m<ipython-input-4-324b4f1c2a7e>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m     14\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     15\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 16\u001b[0;31m \u001b[0mtf\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mreset_default_graph\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     17\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     18\u001b[0m x = tf.placeholder(tf.float32, shape =[None,2],\n",
      "\u001b[0;31mAttributeError\u001b[0m: module 'tensorflow' has no attribute 'reset_default_graph'"
     ]
    }
   ],
   "source": [
    "DATA_SIZE = 100\n",
    "SAVE_PATH = '/users/Shayanriyaz/OneDrive/Documents/CyberSens/AIMO Project/Data/.save'\n",
    "EPOCHS = 1000\n",
    "LEARNING_RATE = 0.01\n",
    "MODEL_Name = 'test'\n",
    "\n",
    "if not os.path.exists(SAVE_PATH):\n",
    "    os.mkdir(SAVE_PATH)\n",
    "    \n",
    "data = (np.random.rand(DATA_SIZE,2),\n",
    "       np.random.rand(DATA_SIZE,1))\n",
    "test = (np.random.rand(DATA_SIZE // 8,2),\n",
    "       np.random.rand(DATA_SIZE // 8,1))\n",
    "\n",
    "\n",
    "tf.reset_default_graph()\n",
    "\n",
    "x = tf.placeholder(tf.float32, shape =[None,2],\n",
    "    name= 'inputs')\n",
    "y = tf.placeholder(tf.float32,shape = [None,1],\n",
    "                  name = 'targets')\n",
    "\n",
    "net = tf.layers.dense(x,16,activation = tf.nn.relu)\n",
    "net = tf.layers.dense(net,16,activation = tf.nn.relu)\n",
    "pred = tf.layer.dense(net,1,activatipn = tf.nn.sigmoid,\n",
    "                     name = 'prediction')\n",
    "\n",
    "\n",
    "\n",
    "loss = tf.reduce_mean(tf.squared_difference(y,pred),name = 'loss')\n",
    "train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)\n",
    "\n",
    "checkpoint = tf.train.latest_checkpoint(SAVE_PATH)\n",
    "should_train = checkpoint = None\n",
    "\n",
    "with tf.Session as sess:\n",
    "    sess.run(tf.global_variables_initializer())\n",
    "    if should_train:\n",
    "        print(\"Training\")\n",
    "        saver = tf.train.Saver()\n",
    "        for epoch in range(EPOCHS):\n",
    "            _, curr_loss = sess.run([train_step,loss],\n",
    "                                   feed_dict = {x:data[0],y:data[1]})\n",
    "            print('EPOCH = {}, LOSS = {:0.4f}'.format(epoch,curr_loss))\n",
    "            path = saver.save(sess,SAVE_PATH + '/' + MODEL_NAME + '.ckpt')\n",
    "            print(\"saved at {}\".format(path))\n",
    "    else:\n",
    "            print(\"Restoring\")\n",
    "            graph = tf.get_default_graph()\n",
    "            saver = tf.train.import_meta_graph(checkpoint + '.meta')\n",
    "            saver.restore(sess, checkpoint)\n",
    "                \n",
    "            loss = graph.get_tensor_by_name('loss:0')\n",
    "            test_loss = sess.run(loss,feed_dict = {'input:0': test[0],'targets:0': test[1]})\n",
    "                \n",
    "            print(ses.run(pred, feed_dict = {'inputs:0': np.random.rand(10,2)}))\n",
    "            print(\"TEST LOSS = {0:0.4f}\".format(test_loss))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
