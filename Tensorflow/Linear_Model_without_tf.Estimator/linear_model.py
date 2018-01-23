import tensorflow as tf
"""
Declaration of the complete paramenters used for the model
"""
W = tf.Variable([.3], dtype=tf.float32, name="Weight")
b = tf.Variable([-.3], dtype=tf.float32, name="Bias")
x = tf.placeholder(tf.float32, name="x")
y = tf.placeholder(tf.float32, name="y")
x_train = [1., 2., 3., 4.]
y_train = [0., -1., -2., -3.]
"""
Mathemetical Expressions required for the calculation
"""
linear_model = W*x + b
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
"""
Optimization Using Gradient Descent Optimizer funtion that increases the value of the variable 
inorder to fulfill the requirement given as the argument like 'Minimize'
"""
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
for i in range(1000):
    sess.run(train, {x: x_train, y: y_train})
final_W, final_b, final_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
print("W: %s, b: %s, loss: %s"%(final_W, final_b, final_loss))
#Tensorboard Command
writer = tf.summary.FileWriter("D:\machine_learning\Linear_Model_without_tf.Estimator\graph", sess.graph)