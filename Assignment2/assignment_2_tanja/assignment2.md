# Delivarable-1

No. 24; Updated March 2011
Click here to download and print a PDF version of this document.
Parents are usually the first to recognize that their child has a problem with emotions or behavior. Still, the decision to seek professional help can be difficult and painful for a parent. The first step is to gently try to talk to the child. An honest open talk about feelings can often help. Parents may choose to consult with the child's physicians, teachers, members of the clergy, or other adults who know the child well. These steps may resolve the problems for the child and family.
Following are a few signs which may indicate that a child and adolescent psychiatric evaluation will be useful.
- Marked fall in school performance
- Poor grades in school despite trying very hard
- Severe worry or anxiety, as shown by regular refusal to go to school, go to sleep or take part in activities that are normal for the child's age
- Frequent physical

# Deliverable-2

It is $32*4096 = 131072$ tokens in each batch. And 99087 which corresponds to 75.60% are ignored because of the padding. This means that a large amount of the data is padding and therefore it does not contribute to the training of the model and this means computational ressources are wasted and the training takes lot more time.

 
# Deliverable-3

see python file appended


# Deliverable-4

The total parameter count of the model is `8,053,329,920`

# Deliverable-5

=== Base version ===

MFU:    min 30.86% | mean 32.36% | max 31.98%
TP:     min 304.35 | mean 319.18 | max 327.15

=== `--fused-optimizer` ===

MFU:    min 32.24% | mean 35.84% | max 36.26%
TP:     min 332.51 | mean 354.75 | max 368.54

Here the optimizer does its updates in a single CUDA kernel (meaning it is fused) as opposed to several CUDA kernels. This reduces the overhead from kernel launches and causes throughput to increase.

=== `--compile` ===

MFU:    min 35.11% | mean 36.52% | max 37.82%
TP:     min 347.20 | mean 367.24 | max 377.78

Pytorch is doing just-in-time compilation to the model code, which includes optimizing its execution graph. This might fuse some seperate python operations into a single CUDA kernel, which means we can get a similar speedup as in the --fused-optimizer example.

=== `--fused-optimizer --compile` ===

MFU:    min 33.15% | mean 41.56% | max 43.41%
TFLOPs: min 385.45 | mean 412.12 | max 431.49

Since the first option enables fusing for the optimizer and the second one for the model, using both together yields another increase in throughput.

=== `--sequence-length 4096` ===

MFU:    min 36.48% | mean 39.66% | max 42.66%
TP:     min 358.73 | mean 392.37 | max 406.16

The throughput increases for larger sequence length. This is because kernels are able to operate on more data, therefore there are less kernel launches and therefore less overhead.


# Deliverable-6

In `trace.jpeg` you find the trace of one Feedforward pass. 
In the lowest row, there are CPU functions for each `matmul` call which issue CUDA API calls. 
The actual CUDA kernels are highlighted in the color of the corresponding CPU activity in the uppermost row.

The `matmul` has operands with shapes `[1, 4096, 4096] * [4096, 14336]`. Therefore, one operation takes $2 * 4096^2 * 14336 \approx 481$ GFLOP. At a peak performance of 989 TFLOP/s, this would take 486µs.

The operands have $4096*4096 + 4096*14336$ floats and the result $4096*14336$. Using BF16 (2 bytes/float), the total memory that needs to be moved to/from the GPU is about 268GB. At 4 TB/s peak memory bandwidth, memory movement would take 6.7µs which is a tiny portion of the computation time. 

For the three `matmul` kernels, we read off execution times of 606, 598, 622µs. So there is a significant overhead that is probably not caused by memory bounds.
