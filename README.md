## A small and simple neural network in Python

This is a basic handcrafted neural network implementation. It's not much but the motive was to prove myself that I understood backpropagation.

### Network
We use only dense (linear) layers and sigmoid activation functions. The code was hardcoded to classify mnist but you can easily modify it for other purposes.
We also take advange of vectorized operations with numpy, the version with loops is many times slower on my cpu.

Default network shape is (784,40,10), minibatch size is 20 and learning rate is 5.0. They are kind of arbitrary, I didn't take time to optimize them. With the default hyperparameters network achieves 95% accuracy in a few epochs.

### Data
I downloaded mnist data from: http://yann.lecun.com/exdb/mnist/. The data is also in this repository unmodified.

### Code
`init_net` initializes the network with given shape. `forward` takes the input and predicts the label. `backward` implements the backpropagation algorithm which is the heart of the neural network.

Gradients are accumulated through a minibatch. `optimize` applies the accumulated gradients to weights and biases. `zero_grad` zeroes the gradients between minibatches.

### Backpropagation
I like to think of a neural network as interconnected gears. I made the following picture which helped me a lot. Note that  superscripts here denotes the layer numbers. They are in reverse order starting from 0 which is the output layer.

Here we can easily see how weights in a layer can affect the final gear i.e. the cost. If you rotate W<sup>0</sup> gear one cycle, how much is the cost gear rotated? That ratio is basically the gradient of W<sup>0</sup> gear.

![backprop](https://drive.google.com/uc?export=view&id=1nIFRo3lRl86xfE9aW26FUmvJ52YbPQ-2)

With backpropagation, all we want is to know how much each weight affects the final cost. Once we know that, we can easily modify weights to decrease the cost. The naive approach is to calculate &part;C/&part;w directly for each weight. In that case we modify a weight just a bit, send the input again, check how much the cost has changed. There is a big problem here. For each weight, we have to send the input again and calculate the whole forward pass. That's obviously not practical.
Backpropagation solves this problem in a way that we calculate the derivatives only with a single forward pass. 

#### Gradient of W<sup>0</sup>
Here is how it's work. We start from the final gear (cost) and ask ourselves which gear directly affects it? As you can tell from the image, the answer is A<sup>0</sup>, so we take derivative &part;C/&part;a<sup>0</sup> and note it somewhere. Remember that the derivative only tells how much A<sup>0</sup> affects C. Next we ask which gear directly affects A<sup>0</sup>, it's Z<sup>0</sup>. We calculate &part;z<sup>0</sup>/&part;a<sup>0</sup>. Next we ask which gear directly affects Z<sup>0</sup>, there are actually two, W<sup>0</sup> and A<sup>-1</sup>. Let's continue with W<sup>0</sup> first and calculate &part;z<sup>0</sup>/&part;w<sup>0</sup>.

Now we have three pieces of information. How much W<sup>0</sup> affects Z<sup>0</sup>, how much Z<sup>0</sup> affects A<sup>0</sup> and how much A<sup>0</sup> affects the cost. We multiply these three derivatives and what we get is how much W<sup>0</sup> affects the final cost, i.e. W<sup>0</sup>'s gradient. Check the image, this is the equation in the second group.

#### The Checkpoints
Here's the crucial part, since we already calculated how much Z<sup>0</sup> affects the cost, we can consider it as a checkpoint. So instead of calculating how much W<sup>1</sup> affects the cost directly, we can calculate how much it affects the Z<sup>0</sup>. As we already know how much Z<sup>0</sup> affects cost. Finally, we can multiply these two derivatives and recover the total effects i.e. how much W<sup>1</sup> affects the final cost.

#### Gradient of W<sup>-1</sup>
So let's calculate the gradient of W<sup>1</sup>. Which gear directly affects Z<sup>0</sup>? They are W<sup>0</sup> and A<sup>-1</sup>, but we already calculated W<sup>0</sup>, and we're trying to get to W<sup>1</sup>. Hence, we continue with A<sup>-1</sup> and calculate &part;z<sup>0</sup>/&part;a<sup>-1</sup>. Next, which gear directly affects A<sup>-1</sup>? It's Z<sup>-1</sup>. We calculate &part;a<sup>-1</sup>/&part;z<sup>-1</sup>. Next, which gear directly affects the Z<sup>-1</sup>? 

Both A<sup>-2</sup> and W<sup>-1</sup>. We're interested in W<sup>-1</sup> so we calculate &part;z<sup>-1</sup>/&part;w<sup>-1</sup>. Let's phrase it. We know how much W<sup>-1</sup> affects Z<sup>-1</sup> and we know how much Z<sup>-1</sup> affects A<sup>-1</sup> and we know how much A<sup>-1</sup> affects Z<sup>0</sup>. We multiply these three derivatives and learn how much W<sup>-1</sup> affects Z<sup>0</sup>. Once that's done, we can multiply this with gradient of Z<sup>0</sup> to learn how much W<sup>-1</sup> affects the final cost, which is the gradient of W<sup>1</sup>. This is the &part;C/&part;W<sup>1</sup> equation in the third group of equations.

#### Other Gradients
As you may guess, now Z<sup>-1</sup> is our new checkpoint, so we repeat the same process for the rest of the layers. For instance, to calculate gradient of W<sup>-2</sup>, we follow the exact steps in the previous section, only the indices change. Once we calculated the gradients, the job of backpropagation is done. Optimizer applies those gradients to actual weights and biases.

#### Biases

Note that we never mentioned biases in this writing or in the picture. That's because it's very easy to calculate and while we're calculating the gradients of weights, we implicitly calculate the gradients of biases as well. Imagine another gear connected to Z<sup>0</sup> and it's named B<sup>0</sup>, just like the weight. How much B<sup>0</sup> affects Z<sup>0</sup> is a constant. &part;z<sup>0</sup>/&part;b<sup>0</sup> is actually 1. So how much B<sup>0</sup> affects the cost is the same as how much Z<sup>0</sup> affects it, therefore gradient of B<sup>0</sup> is the gradient of Z<sup>0</sup>, which we already calculated. You can see it in the `backward` function with the assignments `dc_db = dc_dz`.

### Sources

The main source I used was Grant Sanderson's awesome neural network series, mostly the [fourth video](https://www.youtube.com/watch?v=tIeHLnjs5U8&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=4).
Another helpful source was Casper Hansen's neural network [tutorial](https://mlfromscratch.com/neural-networks-explained).
