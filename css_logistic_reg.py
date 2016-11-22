from __future__ import  print_function
import tensorflow as tf
import input_parser as ip
import gensim

input_file = "./data/data.xlsx"

#tf parameters
learning_rate = 0.01
training_epochs = 25
batch_size = 100
display_step = 1

# word2vec_file = "./word2vec_models/default_200/my200.mdl"
word2vec_file = "./word2vec_models/updated_200/my_data_model.mdl"

model = gensim.models.Word2Vec.load(word2vec_file)
num_features = model.layer1_size

input_data = ip.DataParser(input_file)
num_classes = input_data.num_classes()

X = tf.placeholder(tf.float32, [None, num_features]) # input_vector: shape of word2vec model vector
Y = tf.placeholder(tf.float32, [None, num_classes])  # output_vector: shape of num_classes

W = tf.Variable(tf.zeros([num_features, num_classes])) #weights
b = tf.Variable(tf.zeros([num_classes])) #bias

#model
pred = tf.nn.softmax(tf.matmul(X, W) + b)

#cost
cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(pred), reduction_indices=1))

#optimier
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.initialize_all_variables()


with tf.Session() as sess:
    sess.run(init)

    #training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(input_data.num_examples()/batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = input_data.next_batch(batch_size, model)

            _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs,
                                                          Y: batch_ys})

            avg_cost += c/total_batch

        if (epoch+1)%display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    test_x, test_y = input_data.get_vectorize_test_data_mean(model)
    print("Accuracy:", accuracy.eval({X: test_x, Y: test_y}))

    test_x, test_y = input_data.get_vectorize_train_data_mean(model, 0, -1)
    print("Bias Accuracy:", accuracy.eval({X: test_x, Y: test_y}))