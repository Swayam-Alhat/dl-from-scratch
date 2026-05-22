# Learnings

## Chain rule

Let's consider a example,

Car travels 2x (twice) fast as bicycle

Bicycle travels 4x (4 times) fast as walking man

Then, how much fast car is as compared to walking man?

Let's consider values,

If, walking man speed is 1km/hr , then we know, bicycle is 4x times fast as walking man,

so, speed of bicycle is $4 \times 1 = 4$. So, speed of bicycle is 4km/hr

And we know, car is 2x (twice) as fast as car. so speed of car is $2 \times 4 = 8$. So, car speed is 8km/hr

So, to know how much fast car is as compared to man, we will compare both speeds

car speed = 8km/hr

walking man speed = 1km/hr

So, we get know, car is 8 times fast as walking main.

And if we multiply speed of car with bicycle i.e $8 \times 4 = 8$.

so, we get 8. Means if we know,

how x is depend on y

and, how y is depend on z,

we can know, how x is depend on z -> by multiplying `how x depends on y` with `how y depends on z`

The same concept is chain rule

**_That's why we take product NOT add in chain rule_**

## How learning happens

After forward pass & loss calculation, backpropagation begins

backpropagation calculates gradient (also called derivative) of all weights with the respect to loss function. Means it calculate how each weights in network impacts loss function's output.

we know, derivative means how function's output changes when we slightly change input value. Let's say function is f(x), so we calculate derivative of f(x) with the respect to x.

Undertand about derivatives, partial derivatives and chain rule in detail

This same applies here in ANN. we know, loss function outputs loss. So, we take derivative of loss function with the respect to each weights in network to know how much each weight affects loss (i.e loss function's output)

After calculating derivatives of each weight, we use gradient update rule to update weights in such a way, at next iteration, loss function output is minimized.

The gradient update rule formula is designed in such a way, that weights are updated in such a way, that loss function output is decreased

loss decrease means accuracy of model increases

## backprop

backprop in simple words, is process of calculating derivative of loss function with the respect to each weights and bias of network that produced this loss function.

example, for typical neural net,

we know loss function outputs loss value. If this value is huge, then accuracy is low. If loss is very small, then accuracy of network is high. So, we want loss function to output minimum value. Thats why we have to update its input (prediction value) so that loss function outputs minimum value

so, we use derivative concept. We take derivative of loss function w.r.t its input (prediction) i.e dL/dp

After calculating derivative of loss function w.r.t p (prediction), we can update p so loss function outputs minimum value,

But,  
Prediction (p) is not just a single independent value. It was produced by an activation function (like tanh, ReLU, sigmoid etc). ( we know, neural network have weights and bias. each neuron in network calculates weighted sum of its inputs and weights and apply activation. Their output is again passed to next layer as inputs. Again, neurons in next layer , calculates weighted sum of this inputs (which are output of previous neuron) , apply activation function, same for next neurons in next layer)

So, Basically, Our end output known as loss was not only produced by loss function, but by each weights and bias of neural network.

So, basically loss is produced by all the weights and bias in network. So we have to calculate derivative loss function w.r.t each weights and bias.

And for that we use chain rule,

i.e if z is produced y & y is produced by x. so, if we calculate

derivative of z w.r.t y, i.e (dz/dy) we get to know, how y impacts z

same way, calculate derivative of y w.r.t x, (dy/dx) to get how x impacts y

But to know, how x impacts z,

we have to calculate derivative of z w.r.t x i.e (dz/dx)

so, to calculate dz/dx, we use chain rule, which says

$dz/dx = dy/dx \times dz/dy$

means, `impact of x on z` = `impact of x on y` $\times$ `impact of y on z`.

So,

As loss function is depend on each weights and bias in network, we have take derivative of loss function w.r.t each weights and bias

so, we will calculate in this way,  
first, calculate dL/dp,  
dL/dp tells us how p impacts L (loss function)  
so, we can use dL/dp to update p, so L outputs minimum value

next, we know, prediction p is function's output. we have to take derivative of that function w.r.t its input.

prediction p was produced by sigmoid(weighted sum), weighted sum is numerical value produced by calculating weighted sum of inputs of previous layer.

so, take derivative of p (sigmoid function) w.r.t weighted sum. i.e dp/dws  
This tells us how weighted sum impacts p

But, we want to know how weighted sum impacts L (loss function). Because we have to reduce it

So, we calculate dL/dws = dL/dp $\times$ dp/dws,  
`impact of ws on L` = `impact of p on L` $\times$ `impact of ws on p`

## Reset gradient of weights to zero after backward pass

After forward pass, loss is calculated & gradient of loss w.r.t each weight is calculate. So, further, we can update those weights.

We know, we have to iterate :

1. forward pass
2. backward pass

But after each backward pass, we should reset the gradients to 0. Because weights are updated. Means now, gradients are loss is w.r.t old weights

So, we should reset it to zero

If we don't reset it, after first backward pass, each weight has gradient value. weights are updated. So, at next forward pass & backward pass, gradients are added to previous value because backward is function for single iteration. And we add grads.

```python
def __add__(self, other):

        other = other if isinstance(other , Value) else Value(other)
        out = Value(self.data + other.data , (self, other) , '+')

        # define _backward for Value objects which are result of operation between 2 objects
        # This function calculates derivative of out w.r.t its inputs (self, other)
        def _backward():
            # we calculate grad for chidren nodes of out object using chain rule
            # derivative of f(a,b) = a + b is always 1
            # here we use chain rule to calculate grad of childs i.e
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        # set backward function
        out._backward = _backward
        return out
```

This \_backward is for each operation

Actual backward is occuring this way,

```python
# actual backward function for final output(loss function)
    def backward(self):

        # build a sequential graph expression
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # apply _backward to calculate grad using chain rule in reverse order
        # set grad of final output (loss) w.r.t loss 1.0
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
```

**So, after single backward pass, we should set grads of loss w.r.t each weights to zero**

## Mini bacth gradient descent

In real world, we have large dataset, so we cannot forward pass entire dataset at once. So, we select small part of training data (batch) & perform forward pass, backward pass & weight updates ...

## Shrink learning rate as we get close to minimum loss

At begining of training, learning rate is high & as we get closer to minimum loss, we shrink learning rate so to converge slowly...

This is called **learning rate decay**
