from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

from argparse import ArgumentParser

import socket
my_name = socket.gethostname()

from trains import Task
task = Task.init(project_name='TensorFlow 2 example',
                 task_name='TensorFlow 2 quickstart for experts - {}'.format(my_name))


# Build the tf.keras model using the Keras model subclassing API
class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(32, 3, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)


def prepare_data(batch_size):
    # Load and prepare the MNIST dataset.
    mnist = tf.keras.datasets.fashion_mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Add a channels dimension
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    # Use tf.data to batch and shuffle the dataset
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(batch_size)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)
    return train_ds, test_ds


def train_model(model, trains_task_parameters, train_ds, test_ds):
    print('Starting model training with learning_rate={} and batch_size={}'.
          format(trains_task_parameters.learning_rate, trains_task_parameters.batch_size))

    # Choose an optimizer and loss function for training
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=trains_task_parameters.learning_rate)

    # Select metrics to measure the loss and the accuracy of the model.
    # These metrics accumulate the values over epochs and then print the overall result.
    train_loss = tf.keras.metrics.Mean(name='train_loss', dtype=tf.float32)
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss', dtype=tf.float32)
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    # Use tf.GradientTape to train the model
    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images)
            loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, predictions)

    # Test the model
    @tf.function
    def test_step(images, labels):
        predictions = model(images)
        t_loss = loss_object(labels, predictions)

        test_loss(t_loss)
        test_accuracy(labels, predictions)

    # Set up summary writers to write the summaries to disk in a different logs directory
    train_log_dir = '/tmp/logs/gradient_tape/train'
    test_log_dir = '/tmp/logs/gradient_tape/test'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    # Set up checkpoints manager
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
    manager = tf.train.CheckpointManager(ckpt, '/tmp/tf_ckpts', max_to_keep=3)
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    # Start training
    iteration = 0
    for epoch in range(trains_task_parameters.epochs):
        for images, labels in train_ds:
            iteration += 1
            train_step(images, labels)
            with train_summary_writer.as_default():
                tf.summary.scalar('loss/train_loss', train_loss.result(), step=epoch)
                tf.summary.scalar('accuracy/train_accuracy', train_accuracy.result(), step=epoch)
                if iteration % 1000 == 0:
                    tf.summary.image('image', images, step=iteration, max_outputs=8)

        ckpt.step.assign_add(1)
        if int(ckpt.step) % 1 == 0:
            save_path = manager.save()
            print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))

        for test_images, test_labels in test_ds:
            test_step(test_images, test_labels)
            with test_summary_writer.as_default():
                tf.summary.scalar('loss/test_loss', test_loss.result(), step=epoch)
                tf.summary.scalar('accuracy/test_accuracy', test_accuracy.result(), step=epoch)

        template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        print(template.format(epoch+1,
                              train_loss.result(),
                              train_accuracy.result()*100,
                              test_loss.result(),
                              test_accuracy.result()*100))

        # Reset the metrics for the next epoch
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()


if __name__ == "__main__":
    tf.keras.backend.set_floatx('float64')
    parser = ArgumentParser()
    parser.add_argument("--learning_rate", type=float, default=0.001, help='Learning rate for the train process"')    
    parser.add_argument("--batch_size", type=int, default=32, help='Batch size for the train process"')    
    parser.add_argument("--epochs", type=int, default=5, help='Number of epochs for the train process"')  
    task_parameters = parser.parse_args()

    train_dataset, test_dataset = prepare_data(task_parameters.batch_size)

    # Create an instance of the model
    task_model = MyModel()

    train_model(task_model, task_parameters, train_dataset, test_dataset)
