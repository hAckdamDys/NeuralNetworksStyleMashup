# importujemy biblioteke pomagajaca nam w operacjach tensorowych
import tensorflow as tf

# importujemy przykladowe dane wejsciowe
from tensorflow.examples.tutorials.mnist import input_data

# importujemy biblioteke do rysowania wykresow i wstepnie ja konfigurujemy
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (5.0, 4.0)
plt.rcParams['image.cmap'] = 'Greys'

# na koniec importujemy i konfigurujemy standardowa biblioteke do obliczen numerycznych
import numpy as np
np.set_printoptions(suppress=True)
np.set_printoptions(precision=2)

# wsparcia dla przetwarzania obrazow
from PIL import Image

# i wysokopoziomowa biblioteke keras (ktora uzywa tensorflow w roli swojego backendu)
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input, decode_predictions

# zbior jest automatycznie podzielony na train, validation i test
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


def intro():
    # DONE: obejrzyjmy przykladowy element zbioru uczacego - cyfre 3 lub 6
    example_image = mnist.train.images[3]
    # koniecznie przeksztalcenie wektora ponownie w obrazek
    example_image_reshaped = example_image.reshape((28, 28))
    example_label = mnist.train.labels[3]
    print(example_label)
    plt.imshow(example_image_reshaped)
    plt.show()


def exercise_one():
    x = tf.placeholder(tf.float32, [None, 784])  # miejsce na obraz cyfry
    y_ = tf.placeholder(tf.float32, [None, 10])  # miejsce na klase cyfry

    W = tf.Variable(tf.zeros([784, 10]))  # wagi funkcji liniowej
    b = tf.Variable(tf.zeros([10]))  # bias funkcji liniowej
    y = tf.matmul(x, W) + b  # wynik funkcji f
    # DONE: zastap None, zaimplementuj funkcje f
    # wykorzystaj przygotowane wyzej zmienne oraz tf.matmul i tf.nn.softmax

    # cross_entropy = tf.reduce_mean(-tf.reduce_sum(None, reduction_indices=[1]))

    #wiekszy refactor:
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))


    # DONE: zastap None, dokoncz implementacje funkcji L
    # wykorzystaj przygotowane wyzej zmienne oraz tf.log

    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    for i in range(1000):
        # by przyspieszyc trenowanie uzywamy 100 elementowy batchy zbioru uczacego na raz
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
    # DONE: jaki wynik udalo sie uzyskac?
    # 0.9201

def exercise_two():
    x = tf.placeholder(tf.float32, [None, 784])  # miejsce na obraz cyfry
    y_ = tf.placeholder(tf.float32, [None, 10])  # miejsce na klase cyfry

    W1 = tf.Variable(tf.zeros([784, 100]))
    b1 = tf.Variable(tf.zeros([100]))
    h1 = tf.nn.relu(tf.matmul(x, W1) + b1)

    W2 = tf.Variable(tf.zeros([100, 10]))
    b2 = tf.Variable(tf.zeros([10]))
    y = tf.matmul(h1, W2) + b2
    # DONE: zastap None, zaimplementuj aktywacje obu warstw wzorujac sie na exercise_one
    # dla pierwszej warstwy skorzystaj z tf.nn.relu, dla drugiej (wyjsciowej) nadal z tf.nn.softmax

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
    # DONE: zastap None, wykorzystaj powyzej implementacje z exercise_one
    # w pozostalych przypadkach uzyj implementacji z poprzednich cwiczen
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    init = tf.global_variables_initializer()
    sess = tf.InteractiveSession()
    sess.run(init)



    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        if i % 50 == 0:
            train_acc = accuracy.eval({x: batch_xs, y_: batch_ys})
            print('step: %d, acc: %6.3f' % (i, train_acc) )

    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
    # DONE: jaki wynik udalo sie tym razem uzyskac? Czy zaskoczyl cie?
    # 0.1135

def exercise_three():
    # DONE: rozwiaz problem z poczatkowymi aktywacjami funkcji ReLU
    # zmien poczatkowe wartosci wag na losowe z zakresu -0.1 do 0.1, a biasy na po prostu 0.1
    # skorzystaj z funkcji tf.truncated_normal (dla wag) i tf.constant (dla biasow)
    x = tf.placeholder(tf.float32, [None, 784])  # miejsce na obraz cyfry
    y_ = tf.placeholder(tf.float32, [None, 10])  # miejsce na klase cyfry


    W1 = tf.Variable(tf.random_uniform([784, 300], -0.1, 0.1))
    b1 = tf.Variable(tf.zeros([300]))
    h1 = tf.nn.relu(tf.matmul(x, W1) + b1)

    W2 = tf.Variable(tf.random_uniform([300, 300], -0.1, 0.1))
    b2 = tf.Variable(tf.zeros([300]))
    h2 = tf.nn.relu(tf.matmul(h1, W2) + b2)

    # W2 = tf.Variable(tf.zeros([100, 10]))
    W3 = tf.Variable(tf.random_uniform([300, 10], -0.1, 0.1))

    b3 = tf.Variable(tf.zeros([10]))
    y = tf.matmul(h2, W3) + b3



    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))

    # train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)

    train_step = tf.train.MomentumOptimizer(0.05,0.9).minimize(cross_entropy)


    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # testVal = tf.Variable(tf.random_uniform([784, 100], -0.1, 0.1))

    init = tf.global_variables_initializer()
    sess = tf.InteractiveSession()
    sess.run(init)
    # print(sess.run(testVal))



    for i in range(2000):
        batch_xs, batch_ys = mnist.train.next_batch(1000)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        if i % 50 == 0:
            train_acc = accuracy.eval({x: batch_xs, y_: batch_ys})
            print('step: %d, acc: %6.4f' % (i, train_acc) )


    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
    # DONE: raz jeszcze sprawdz jaka wartosc przyjmie accuracy
    # 0.9787

def exercise_four():
    # DONE: uruchom przyklad pobierajacy gotowa siec, wytrenowana na zbiorze imagenet
    # pobranie gotowego modelu zlozonej sieci konwolucyjnej
    model = VGG16(weights='imagenet', include_top=True)
    # DONE: podejrzyj z ilu i jakich warstw sie sklada
    layers = dict([(layer.name, layer.output) for layer in model.layers])
    for name, layer in layers.items():
        print(name, layer)
    # DONE: podejrzyj ile parametrow musialo zostac wytrenowanych
    print(model.count_params())

    # otworzmy przykladowe zdjecie i dostosujemy jego rozmiar i zakres wartosci do wejscia sieci
    image_path = 'cat_disguise.jpg'
    image = Image.open(image_path)
    image = image.resize([224,224])
    # DONE: zastap None dobierajac wlasciwy rozmiar wejsciowy obrazu
    # (poznasz go analizujac wypisana wczesniej strukture sieci)
    x = np.asarray(image, dtype='float32')
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # i sprawdzimy jaki wynik przewidzi siec
    preds = model.predict(x)
    print('Predicted:', decode_predictions(preds, top=3)[0])

    # DONE: pobaw sie z innymi zdjeciami z Internetu - jak radzi sobie siec? kiedy sie myli?


def main():
    # DONE: tu wybieraj wykonywane cwiczenie
    # intro()
    # exercise_one()
    # exercise_two()
    # exercise_three()
    exercise_four()


if __name__ == '__main__':
    main()
