# NanoGPT Practice / NanoGPT 实践 🚀

This directory contains my practice code for building a Generative Pre-trained Transformer (GPT) from scratch.
这个目录包含了从零开始构建 GPT 模型的实践代码。

## 📺 Reference / 参考资料

The implementation here is entirely based on Andrej Karpathy's excellent tutorial video:
这里的代码实现完全基于 Andrej Karpathy 的经典教程视频：
**[Let's build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY)**

## 🔍 Differences from the Official nanoGPT / 与官方 nanoGPT 项目的区别

While this code follows the educational video step-by-step, the official `nanoGPT` repository incorporates several practical optimizations for better performance and efficiency.
虽然这里的代码是跟着教学视频一步步写出来的，但官方的 `nanoGPT` 仓库包含了一些更贴近实战的性能优化。

Key differences (some of which were explicitly mentioned by Karpathy in the video) include:
主要区别（其中一些 Karpathy 已在视频末尾提及）包括：

- **Merged QKV Projections / QKV 权重合并:** 
  Instead of separate linear layers for Query, Key, and Value, the official implementation merges them into a single linear layer (`c_attn`) for faster computation.
  官方实现没有为 Query、Key 和 Value 分别使用独立的线性层，而是将它们合并成了一个单一的线性层（`c_attn`）以加速计算。

- **Weight Tying / 参数共享:** 
  The official repo shares the weights between the token embedding table and the final language modeling head (`lm_head`), which significantly reduces the total parameter count.
  官方仓库在 Token 词嵌入表和最终的语言建模头（`lm_head`）之间共享了权重，这大幅减少了模型的总参数量。

- **Weight Decay Separation / 权重衰减分离:** 
  As highlighted in the video, the official code carefully separates parameters that should be weight-decayed (e.g., 2D matrix weights) from those that shouldn't (e.g., biases and LayerNorm weights) to ensure better training dynamics.
  正如视频中强调的那样，官方代码仔细区分了需要进行权重衰减的参数（如二维矩阵权重）和不需要衰减的参数（如偏置项和 LayerNorm 权重），以确保更好的训练动态。
