import numpy as np
import tensorflow as tf
#
f_columns = [tf.feature_column.numeric_column("x", shape=[1])]
estimator = tf.estimator.LinearRegressor(feature_columns = f_columns, model_dir="D:/machine_learning/Linear_Model_with_tf.Estimator/output")
#Declaration of the input arrays
x_train = np.array([1., 2., 3., 4.])
y_train = np.array([0., -1., -2., -3.])
x_eval = np.array([2., 5., 8., 1.])
y_eval = np.array([-1.01, -4.1, -7., 0.])
#input functions declaration
in_fn = tf.estimator.inputs.numpy_input_fn({"x": x_train}, y_train, batch_size=4, num_epochs=None, shuffle=True)
in_train_fn = tf.estimator.inputs.numpy_input_fn({"x": x_train}, y_train, batch_size=4, num_epochs=1000, shuffle=False)
in_eval_fn = tf.estimator.inputs.numpy_input_fn({"x": x_eval}, y_eval, batch_size=4, num_epochs=1000, shuffle=False)
#train API
estimator.train(input_fn=in_fn, steps=1000)
train_metrics = estimator.evaluate(input_fn=in_train_fn)
eval_metrics = estimator.evaluate(input_fn=in_eval_fn)
print("train_metrics: %r"%train_metrics)
print("eval_metrics: %r"%eval_metrics)
"""
Output:
train_metrics: {'global_step': 1000, 'average_loss': 3.0425802e-08, 'loss': 1.2170321e-07}
eval_metrics: {'global_step': 1000, 'average_loss': 0.0025418026, 'loss': 0.01016721}
"""