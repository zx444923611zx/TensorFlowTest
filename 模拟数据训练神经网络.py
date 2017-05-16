#同numpy生成模拟数据训练神经网络
import tensorflow as tf
from numpy.random import RandomState

#训练数据的batch大小
batch_size = 8

#神经网络参数
w1 = tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
w2 = tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))

#输入数据占位
x = tf.placeholder(tf.float32,shape=(None,2),name='x-input')
y_ = tf.placeholder(tf.float32,shape=(None,1),name='y-input')

#神经网络传播过程
a = tf.matmul(x,w1)
y = tf.matmul(a,w2)

#定义损耗函数和反向传播算法
cross_entropy = -tf.reduce_mean(y_*tf.log(tf.clip_by_value(y,1e-10,1.0))) #y的输出值为1 ，所有不要按行求和
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy) #训练的模型,不同的优化器训练出来的不一样


#通过随机数生成一个模拟数据集
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size,2)
#定义规则来给出样本的标签x1+x2<1被认为是正样例，其他的为负样例，0表示为负样例，1表示正样例
Y = [[int(x1+x2<1)] for (x1,x2) in X]

#变量初始化
with tf.Session() as sess: #Session()大写和括号
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    # sess.close()

    #训练前权值w1,w2的初始化值
    print(w1)
    print(sess.run(w2))

    #按batch进行训练模型
    train_i = 5000
    for i in range(train_i):
        start = (i*batch_size) % dataset_size
        end = min(start+batch_size,dataset_size)
        # end = (i*batch_size) % 128 + batch_size
        sess.run(train_step,feed_dict={x:X[start:end],y_:Y[start:end]})
        if i%1000==0:
            total_cross_entropy = sess.run(cross_entropy,feed_dict={x:X,y_:Y})
            print("After %d training steps,cross emtropy on all data is %g"%(i,total_cross_entropy))

    #输出训练后的权值
    print(sess.run(w1))
    print(sess.run(w2))

