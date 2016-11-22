from __future__ import  print_function
import tensorflow as tf
import input_parser as ip
import gensim

input_file = "../data/data.xlsx"

#tf parameters
learning_rate = 0.01
training_epochs = 50
batch_size = 100
display_step = 1

# word2vec_file = "../word2vec_models/default_200/my200.mdl"
word2vec_file = "../word2vec_models/updated_200/my_data_model.mdl"

model = gensim.models.Word2Vec.load(word2vec_file)
num_features = model.layer1_size

input_data = ip.DataParser(input_file)
num_classes = input_data.num_classes()

n_hidden_1 = 170 # 1st layer number of features
n_hidden_2 = 170
n_hidden_3 = 170

# tf Graph input
x = tf.placeholder("float", [None, num_features])
y = tf.placeholder("float", [None, num_classes])


# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)

    # Output layer with linear activation
    out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
    return out_layer

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([num_features, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    'out': tf.Variable(tf.random_normal([n_hidden_3, num_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)


# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(input_data.num_examples() / batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = input_data.next_batch(batch_size, model)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
                                                          y: batch_ys})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_cost))
    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    test_x, test_y = input_data.get_vectorize_test_data_mean(model)
    print("Accuracy:", accuracy.eval({x: test_x, y: test_y}))

    test_x, test_y = input_data.get_vectorize_train_data_mean(model, 0, -1)
    print("Bias Accuracy:", accuracy.eval({x: test_x, y: test_y}))