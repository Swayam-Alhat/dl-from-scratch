# Gradient Descent

> [!Note]
> Before reading this, understand what is function, Its derivatives & partial derivatives. Also understand why we calculate derivative of any function

## Goal

We want to minimize the Cost function C.  
Let's assume that cost function C is simple function with 2 variables (v1, v2) & we want to minimize it.  
Means, we want to find values of variables (v1, v2) such that
C outputs the minimum possible value.

## Step 1 - Formula for Change in C

We know, C is a function of two variables v1 and v2.
When we change v1 and v2 slightly, C also changes.
The formula for that change is:

ΔC = (∂C/∂v1) × Δv1 + (∂C/∂v2) × Δv2

_This above formula is called total differential formula_  
_It is used to calculate the change in a function of multiple variables_

Where:

- ∂C/∂v1 = partial derivative of C w.r.t v1 (how C changes when v1 changes slightly)

- ∂C/∂v2 = partial derivative of C w.r.t v2 (how C changes when v2 changes slightly)

- Δv1, Δv2 = small changes in v1 and v2

> [!Note]
> Partial derivatives tells us how a function changes when we slightly change
> one variable while keeping others constant.  
> When we bundle all partial derivatives into a single vector, that is called the Gradient.

## Step 2 - Rewriting using Vectors (Dot Product)

We can bundle partial derivatives into one vector called gradient:

∇C = (∂C/∂v1 , ∂C/∂v2)

And bundle the changes into one vector:

Δv = (Δv1 , Δv2)

Now the same formula can be written cleanly as dot product:

ΔC = ∇C · Δv

Same formula, just cleaner notation.

## Step 3 - How to make ΔC negative?

We want ΔC to be negative. (Means C value decreased. That is our goal.)

So lets choose Δv smartly:

Δv = -lr × ∇C

Where lr = learning rate (a small positive number)

## Step 4 - Proof that ΔC will always be negative

Substitute Δv in the formula:

ΔC = ∇C · (-lr × ∇C)
ΔC = -lr × (∇C · ∇C)
ΔC = -lr × ‖∇C‖²

Now:

- lr is always positive
- ‖∇C‖² is always positive (square of anything is positive)
- So, -lr × ‖∇C‖² is always NEGATIVE ✓

This means, if we choose Δv = -lr × ∇C,
then ΔC will ALWAYS be negative.
Means C will ALWAYS decrease. That is exactly what we want!

## Step 5 - The Update Rule

So what we do is, update current value of v using:

new_v = old_v - lr × ∇C

If we try to undertand it, we will know,

∇C tells us how C changes when v changes slightly.

Case 1: Gradient is positive

- Means, as v increases, C increases
- So we should DECREASE v to bring C down
- The minus sign does exactly that:
  new_v = old_v - lr × (+gradient) → v decreases ✓

Case 2: Gradient is negative

- Means, as v increases, C decreases
- So we should INCREASE v to bring C down
- The minus sign flips it:
  new_v = old_v - lr × (-gradient) → v increases ✓

In both cases, the minus sign automatically pushes v
in the correct direction to decrease C.

**_We keep repeating this update again and again.
Each time, C decreases a little.
Eventually we reach the minimum of C._**

## Step 6 - Role of Learning Rate

- Small lr → tiny steps → slow but safe
- Large lr → big steps → fast but might overshoot the minimum

## Summary

So, in same way, we can minimize the cost function used in ANNs. Meaning we can find the weights which make cost function to output minimum possible value.

This is how, weights and bias are updated to get optimal weights and bias that outputs accurate prediction

The entire gradient descent algorithm in one line:

new_weight = old_weight - lr × ∇C

new_bias = old_bias - lr × ∇C

This one formula, applied repeatedly, minimizes the cost function C.
