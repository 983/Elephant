import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

if 0:
    path = "points_elephant.npy"

    # superfluous parameters will turn out to be zero
    n_parameters = 10*4
else:
    path = "points_fancy_elephant.npy"

    # must be multiple of 4
    n_parameters = 100*4

assert(n_parameters % 4 == 0)

points = np.load(path)

parameters = np.random.randn(n_parameters//4, 4)
parameters = tf.Variable(parameters)

ax = parameters[:, 0:1]
bx = parameters[:, 1:2]
ay = parameters[:, 2:3]
by = parameters[:, 3:4]

# weird shapes so (k*t) broadcasts to shape (n_parameters/4, len(points))
t = np.linspace(0, 2*np.pi, len(points), endpoint=False).reshape(1, -1)
k = np.arange(1, n_parameters//4 + 1).reshape(-1, 1)

c = np.cos(k*t)
s = np.sin(k*t)

# sum over parameter axis
x = tf.reduce_sum(ax*c + bx*s, axis=0)
y = tf.reduce_sum(ay*c + by*s, axis=0)

approximated_points = tf.stack([x, y], axis=1)

difference = points - approximated_points

loss = tf.reduce_mean(tf.square(difference))

optimizer = tf.train.AdamOptimizer(0.1)
train_op = optimizer.minimize(loss)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

n_iterations = 200
for iteration in range(n_iterations):
    train_loss, _ = sess.run([loss, train_op])
    print("iteration %d/%d, loss: %f"%(iteration, n_iterations, train_loss))

print("number of non-zero parameters:")
print(np.sum(np.abs(sess.run(parameters)) > 0.01))

points2 = sess.run(approximated_points)

# append first point to plot closed curve
points2 = np.concatenate([points2, points2[:1]])

if 1:
    plt.plot(points2[:, 0], points2[:, 1])
    # eye
    if len(points2) == 129: plt.plot([0.1], [1], 'bo')
    plt.show()
