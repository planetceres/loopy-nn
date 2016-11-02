## Looping Neural Networks

This is a Tensorflow implementation of looping neural networks, inspired by feedback processes observed in neuroscience. This follows the architecture presented in "Loopy Neural Nets: Imitating Feedback Loop in the Human Brain" by [Caswell, Wang, and Shen](http://cs231n.stanford.edu/reports2016/110_Report.pdf) at Stanford. 

The looping CNN construction is fairly simple. Batched inputs enter the network as raw pixel values, and convolve through layers of filters. These layers together comprise a "loop", which can be "unrolled" n times. The defaults, consistent with the paper, for the MNIST dataset are 16, 8, and 1 layers with 3 unrolls.

After each unroll, the raw pixel inputs re-enter the network, and pass through again. Weights are reused from one unroll to the next, and only after the final unroll is a maxpooling layer and relu activation applied. 

This uses the development branch of Tensorflow, currently version [0.11.0rc0](https://www.tensorflow.org/)
