# Derivatives explained

## Functions

Before diving into derivatives, let's understand **function**.

**_Function is any expression which takes input and generates output._**

Example, $f(x) = x^2$

so, this function takes $x$ as input, and generates output. like, $f(2) = 2^2 = 4$

## Derivatives

**derivative is nothing but slope of function at any point** $x$

Plotting function $f(x) = x^2$ on graph

```python
import numpy as np
import matplotlib.pyplot as plt

# function
def f(x):
    return x**2

Xs = np.linspace(1,5,100)
Ys = f(Xs)

# plot function
plt.plot(Xs, Ys)

```

Output :

![Output](./figures/output.png)

Derivatives gives us value of slope of function for that $x$ value. Meaning, how function's output changes when we slightly nudge $x$

Let's understand by considering example $x = 2$ for function $f(x) = x^2$

So, we know, when $x = 2$, $f(x) = 4$

Let's say, we increase $x$ by $a$ $(a = 0.001)$ , so, function becomes $f(x + a)$

So, $$f(x + a) = f(2 + 0.001) = f(2.001) = 2.001^2 = 4.0040$$

So, if we want to know _slope_ of $f(x)$ for $x = 2$, we use

$$ slope = \frac{f(x + a) - f(x)}{a} $$

$$ slope = \frac{4.0040 - 4}{0.001} $$

$$ slope = \frac{0.004}{0.001} $$

$$ slope = 4 $$

Let's understand how $slope = 4$ gives us information about how function's output changes when we slightly nudge input $x$

from above example,

for $x = 2$,  
derivative or slope of $f(x) = 4$

And when we slightly nudge $x$ towards positive side by a $(a = 0.001)$

i.e

$$x = x + a$$

$$ x = 2 + 0.001$$

for $x = x + a$  
function $f(x + a) = (x + a)^2$ outputs : $4.0040$

if we consider the difference between $f(x + a)$ & $f(x)$

$$= 4.0040 - 4$$

$$ = 0.004$$

This tells us that, for $x = 2$, when we slightly nudge $x$, function's output change (increases) by $0.004$

difference is $0.004$ which is equals to $slope \times a$

$$4 \times 0.001 = 0.004$$

where $slope = 4$ & $a = 0.001$

So, we get to know that slope or derivative of function tells -

If input $x$ is slightly nudged by $a$, output of function changes by $slope \times a$

- If slope or derivative is positive, then function's output increases by $slope \times a$
- If slope or derivative is negative, then function's output decreases by $slope \times a$

## Differentiation

Differentiating a function gives us a equation which is used to find slope or derivative of function $f(x)$ for any value $x$

This slope gives us information about how function's output changes when we slightly increase $x$

Example, if we differentiate function $f(x) = x^2$,

By differentiation rule, $x^n = nx^{n-1}$

we get, $2x$

Now, if we put $x = 2$ in equation $2x$, we get $4$ which is known as slope or derivative of function $f(x)$ for $x=2$

> [!Note]
> If function is linear (i.e forms a straight line when plotted on graph), then we use $$slope (m) = \frac{y_{2} - y_{1}}{x_{2} - x_{1}} = \frac{Rise}{Run}$$  
> For non-linear function (i.e function that forms a curve or parabola), we use calculus derivatives method, because for curve, slope is different at different $x$ value.  
> Differentiating a non-linear function gives us a equation that can be used to find slope of function at any point $x$  
> Example, for function $f(x) = x^2$ ,
>
> - At $x = 2$, derivative or slope is $2x = 2 \times 2 = 4$
> - At $x = 3$ , derivative or slope is $2x = 2 \times 3 = 6$
> - At $x = 4$, derivative or slope is $2x = 2 \times 4 = 8$

## How derivatives are used in Neural nets to reduce loss function

As we undertand how slope or derivative helps us to know how any function's output changes when we slightly nudge its input variables

The same concept is used to reduce a loss function

**To reduce any function - means to find a input value $(x)$ which makes function $f(x)$ to output minimun possible value**

Let's understand how slope or derivative is used to reduce any function

Let's consider same above function $f(x) = x^2$

As we know, slope or derivative of $f(x)$ at $x=2$ is $4$

$slope = 4$

when $x = 2$,  
$f(x) = 2^2 = 4$  
$f(x) = 4$

If we nudge $x$ by tiny amount $a = 0.001$,  
$f(x + a) = (2 + 0.001)^2$  
$f(x + a) = 4.0040$

If we calculate difference between outputs of both $f(x + a)$ & $f(x)$ ,

$f(x+a) - f(x) = 4.0040 - 4 = 0.004$

Meaning, $f(x)$ increased by $0.004$ when we nudge $x$ by tiny amount $a$

And,  
$0.004 = slope \times a$  
$0.004 = 4 \times 0.001$

So, here we get to know that,

**When slope is positive, $f(x)$ output increases as $x$ increases**

So, we decrease $x$ to make $f(x)$ produce lower value

> [!Note]  
> Here, we should increase $x$ by tiny amount to avoid sudden change in slope value. because slope of function changes as $x$ changes.  
> Changing $x$ by small amount changes function's output by small amount which is necessary when we want to find $x$ that produces minimum output of $f(x)$

$x_{new} = x_{old} - slope$

This formula works well to reduce any function.

It works for both cases

- when slope of function is positive, function's output increases as $x$ increases. So, the above formula decreases $x$ so function's output decreases

- when slope of function is negative, function's output decreases as $x$ increases. So, the above formula increases $x$ so function's output decreases

But, above formula $x_{new} = x_{old} - slope$ updates $x$ by huge amount (i.e by $slope$ of function).

So, we use small value like $0.001$ to make changes in $x$ by small amount

$$x_{new} = x_{old} - (0.001 \times slope)$$

In above formula, multiplying $slope$ by very small value like $0.001$ results in a small value. This small value is used to update (either increase or decrease) $x$

The formula,

$$x_{new} = x_{old} - (0.001 \times slope)$$

> [!Note]  
> Above formula is also known as gradient update rule. It is used to update weights in ANNs.In formula, 0.001 is known as learning rate. It can be any small value like 0.01, 0.1 etc

Now, let's use it to check, $f(x) = x^2$ is reduced.

$x_{old} = 2$  
$slope$ of $f(x)$ = $4$

$$x_{new} = x_{old} - (0.001 \times slope)$$

$$x_{new} = 2 - (0.001 \times 4)$$

$$x_{new} = 2 - 0.004$$

$$x_{new} = 1.996$$

Now, use $x_{new} = 1.996$ in $f(x)$

$f(x) = x^2$  
$f(x) = (1.996)^2$  
$f(x) = 3.9840$

So, we can notice that $f(x)$ decreased as we decreased $x$.

So, we repeatedly update $x$ using above formula, until the change in $f(x)$ is very small  
At that point, we get optimal value of $x$ that make $f(x)$ to produce minimum value

> [!Note]  
> Technically, the stopping condition is when gradient becomes very small (close to zero), not just when change in f(x) becomes small.  
> When gradient is close to zero, it means the function has flattened out at that point — meaning we are near a minimum. The change in f(x) also becomes small as a consequence, but gradient is the actual signal we watch.  
> This makes sense from the update rule itself:  
> $$x_{new} = x_{old} - (lr \times slope)$$
> When $slope \approx 0$,  
>  $x_{new} \approx x_{old}$  
> meaning $x$ barely changes

**Same way, we reduce loss function in ANNs**

> [!Note]  
> Partial derivatives is way to calculate derivatives of function who has multiple inputs. example, $f(x,y) = x^2 + 2xy$. This function has 2 inputs, so we take partial derivative of $f(x,y)$ with respect to $x$ while keeping $y$ as constant. Similarly, we take partial derivative of $f(x,y)$ with the respect to $y$ while keeping $x$ as constant

But, as loss function is produced by multiple weights, we take partial derivative of loss function with the respect to each weights. This partial derivatives together is called gradient.

So, formula to update weights in ANNs,

$$w_{new} = w_{old} - (lr \times gradient)$$

where,  
$lr$ is learning rate (any small value like 0.001, 0.01, 0.1)  
$gradient$ is partial derivative of loss function with the respect to $w$

> [!Note]  
> Read [Gradient-descent.md](./Gradient-descent.md) file to know about gradient descent
