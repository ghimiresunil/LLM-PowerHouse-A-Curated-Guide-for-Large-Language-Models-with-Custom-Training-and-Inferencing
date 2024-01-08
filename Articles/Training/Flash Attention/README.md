# Flash Attention

In the LLM world, we often speak of context length (basically the length of the input sequence) of a model and how we can extend it. Now, think of self-attention with a quadratic complexity with respect to the context length. Really hard to scale! Here’s where “Flash Attention” comes in, that is an IO-aware, exact computation of attention but at a memory complexity that is sub-quadratic/linear! But how does it work?

Before that a quick overview of preliminaries, i.e GPU memory hierarchy (from top to bottom):
-  SRAM : Fastest memory (19TB/s with 20MB capacity, where the computations happen) 
- HBM : Slower than SRAM (1.5TB/s with 40GB capacity)
- DRAM : Main memory, very slow (12.8 GB/s > 1TB)

3 main operations need to be done for attention computation. 
- Equation 01: S = QK_Transpose
- Equation 02: P = softmax(S)
- Equation 03: O = PV

Where, Q = Query, K = Key, V = Value and O being the final output

Let’s see how Self-Attention (SA) uses this memory hierarchy :
Loading matrices Q,K,V in HBM.
- Calculates S from (eq 1) and writes it back to HBM.
- Loads S from HBM again computes P (eq 2) and writes P back to HBM.
- Loads P and V from HBM, computes O (eq 3) and writes back to HBM.

See this pattern? With SA, there is a significant overhead due to the back and forth (excessive reads and writes) between the HBM and SRAM.

Let’s see how Flash Attention (FA) uses two concepts to avoid too many reads and writes to the HBM. 

- Tiling: In FA, Q,K,V are all divided into blocks and perform computation of (eq 1) & (eq 2) i.e “softmax reduction” incrementally without having access to the full input so that we can use the SRAM to its full capacity. 
- Recomputation: Store an intermediate result (softmax normalization factor) during backward pass computation.

The "simplified" flow should be as follows: 
- Split softmax into numerator and denominator (since they are both distributive in nature, this partial computation becomes possible). We have 3 partial output values stored in HBM. O_i that is the partial output, L_i, the partial numerator of softmax and M_i the partial denominator.
- In a two-loop structure (outer loop going over blocks of K,V and inner loop going over blocks of Q), use K_i, V_i and Q_j to update the O_j, L_j and M_j in the HBM
- Recompute O_j, L_j and M_j in the backward pass.