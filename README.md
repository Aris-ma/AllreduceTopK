# ARC-TopK

Communication remains a central bottleneck in large-scale distributed machine learning, and gradient sparsification has emerged as a promising strategy to alleviate this challenge. However, existing gradient compressors face notable limitations: \RandK\ discards structural information and performs poorly in practice, while \TopK\ preserves informative entries but loses the contraction property and requires costly \texttt{All-Gather} operations. In this paper, we propose \arctopK, an \texttt{All-Reduce}-Compatible Top-$K$ compressor that aligns sparsity patterns across nodes using a lightweight sketch of the gradient, enabling index-free \texttt{All-Reduce} while preserving globally significant information. \arctopK\ is provably contractive and, when combined with momentum error feedback (EF21M), achieves linear speedup and sharper convergence rates than the original EF21M under standard assumptions. Empirically, \arctopK\ matches the accuracy of \TopK\ while reducing wall-clock training time by up to 60.7\%, offering an efficient and scalable solution that combines the robustness of \RandK\ with the strong performance of \TopK.


# Installation


# Reproduce Experiments