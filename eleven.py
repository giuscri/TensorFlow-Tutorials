import tensorflow as tf
import inception
import numpy as np
import matplotlib.pyplot as plt
import os
from sys import argv, stderr
from PIL import Image
from PIL import ImageFilter

"""
exercises:
    Try using some of your own images.
        > ✔

    Try other arguments for adversary_example().
    Try another target-class, noise-limit and
    required score. What is the result?
        > I've tested few classes: it doesn't seem
        to change how good this trick is. Changing
        required score doesn't seem to speed gradient
        descent too much. Reducing the noise limit
        makes harder to reach the needed score, as
        expected.

    Do you think it is possible to generate
    adversarial noise that would cause
    mis-classification for any desired
    target-class? How would you prove your theory?
        > - 

    Try another formula for calculating the
    step-size in find_adversary_noise(). Can you
    make the optimization converge faster?
        > Fastest I got was `20 / max(1e-10, np.abs(g).max())`

    Try blurring the noisy input-image right
    before it is input into the neural network.
    Does it remove the adversarial noise and cause
    correct classification again?
        > No, seems there's no relation.

    Try lowering the bit-depth of the noisy input
    image instead of blurring it. Does it remove
    the adversarial noise and result in correct
    classification? For example if you only allow
    16 or 32 colour-levels for Red, Green and
    Blue, whereas normally there are 255 levels.
        > This seems to improve the result.

    Do you think your noise-removal also works for
    hand-written digits in the MNIST data-set, or
    for strange geometric shapes? These are
    sometimes called 'fooling images', do an
    internet search.
        > Yes.

    Can you find adversarial noise that would work
    for all images, so you don't have to find
    adversarial noise specifically for each image?
    How would you do this?
        > Don't know. Feels too good to be true. 

    Can you implement the optimization in
    find_adversary_noise() directly in TensorFlow
    instead of using NumPy? You would need to make
    the noise a variable in the TensorFlow graph
    as well, so it can be optimized by TensorFlow.
        > -

    Explain to a friend what Adversarial Examples
    are and how the program finds them. 
        > Instead of training a model to recognize a
        class of images, you can use gradient descent
        (or anything that can minimize a function
        iteratively I guess) to minimize the loss
        function between a target output and what's
        classified by the model, by changing the
        image. This is interesting as the resulting
        image that _fools_ the classifier it's
        indistinguishable from the original image by a
        human.
"""

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def minmax_scale(x):
    return (x - x.min()) / (x.max() - x.min())

def show_images(img, noisy_img, noise, clsname_source, clsname_target,
                    sourcescore, original_sourcescore, targetscore):
    """Shows img, noisy_img, noise in a single figure."""

    figure, axes = plt.subplots(1, 3)

    a = axes.flat[0]
    a.imshow(img / 255.0)
    a.set_xlabel('Original image:\n{} ({:.2%})'.format(
        clsname_source, original_sourcescore
    ))

    a = axes.flat[1]
    a.imshow(noisy_img / 255.0)
    a.set_xlabel('Image + Noise:\n{} ({:.2%})'.format(
        clsname_target, targetscore
    ))

    a = axes.flat[2]
    a.imshow(minmax_scale(noise))
    a.set_xlabel('Amplified noise')

    for a in axes.flat:
        a.set_xticks([])
        a.set_yticks([])

    figure.show()

def adversary_noise(model, session, image_path, cls_target,
                        score=0.99, iterations=100, noiselim=3.0, blur=False, depth=255):
    """
    Computes adversary noise.
   
    This allows to fool Inception `model`
    in classifying image at `img_path`
    as being in class `cls_target` with
    a score of `score`.

    It basically applies gradient descent
    on the loss function computed between
    the neural network output and one-hot
    encoded `cls_target` in respect to
    the image loaded from `img_path`.
    """

    feed_dict = model._create_feed_dict(image_path=image_path)
    predicted, image = session.run([
        model.y_pred,
        model.resized_image
    ], feed_dict=feed_dict)

    if blur:
        image = image.squeeze()
        image = image.astype('uint8')
        image = Image.fromarray(image)
        image = image.filter(ImageFilter.GaussianBlur(radius=3))
        image = np.array(image)
        image = image.reshape((1, 299, 299, 3))

    assert depth > 0, "minimum depth is 1"
    if depth < 255:
        image = image.squeeze()
        image = np.ceil(image / (255 / depth)) * 255 / depth
        image = image.reshape((1, 299, 299, 3))
        # Reduce range of values per channel
        # by playing with ceils.

    predicted = np.squeeze(predicted)
    original_sourcescore = predicted.max()
    cls_source = np.argmax(predicted)

    clsname_source = model.name_lookup.cls_to_name(
                        cls_source, only_first_name=True)
    clsname_target = model.name_lookup.cls_to_name(
                        cls_target, only_first_name=True)

    with model.graph.as_default():
    # You need to take tensors from the Inception graph.

        cls_target_pholder = tf.placeholder(dtype=tf.int32)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=model.y_logits, labels=[cls_target_pholder])

        gradient = tf.gradients(loss, model.resized_image)
        # You want to apply gradient descent on
        # the loss between nn output and one-hot
        # encoded target.  In order to modify
        # `image` you'll leverage the noise that
        # gets added to the img.

    noise = 0
    for _ in range(iterations):
        noisy_image = image + noise
        noisy_image = np.clip(a=noisy_image, a_min=0.0, a_max=255.0)
        feed_dict = {
            model.tensor_name_resized_image: noisy_image,
            cls_target_pholder: cls_target
        }
        predicted, g = session.run([model.y_pred, gradient],
                                    feed_dict=feed_dict)

        predicted = np.squeeze(predicted)
        sourcescore = predicted[cls_source]
        targetscore = predicted[cls_target]

        g = np.array(g).squeeze()

        stepsize = 7 / max(1e-10, np.abs(g).max())

        if targetscore >= score: break

        noise = noise - stepsize * g
        noise = np.clip(a=noise, a_min=-noiselim, a_max=noiselim)

    return (
        image.squeeze(),
        noisy_image.squeeze(),
        noise,
        clsname_source,
        clsname_target,
        sourcescore,
        original_sourcescore,
        targetscore,
    )

if __name__ == '__main__':
    if len(argv) < 2:
        image_path = './images/parrot_cropped1.jpg'
    else:
        image_path = argv[1]

    print(f"using {image_path}", file=stderr)

    model = inception.Inception()

    with tf.Session(graph=model.graph) as session:
        n = adversary_noise(model, session, image_path, 400, 0.90, depth=10)
        show_images(*n)

    model.close()
