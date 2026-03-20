# NanoGPT Practice / NanoGPT 实践 🚀

This directory contains my practice code for building a Generative Pre-trained Transformer (GPT) from scratch.
这个目录包含了从零开始构建 GPT 模型的实践代码。

## 📝 Terminology Note: The Overloaded "Block" / 术语说明：“Block”的歧义

If you are reading the code in this `nanogpt_practice` folder, you will notice the heavy use of the word `block` and `block_size`. This is kept intentionally to faithfully reflect Andrej Karpathy's excellent tutorial. There is no need to rewrite this learning footprint, but it is crucial to understand the context switch when you move to the main `femtovllm` engine.
如果您正在阅读 `nanogpt_practice` 文件夹中的代码，您会注意到 `block` 和 `block_size` 这两个词被频繁使用。这里刻意保留了原样，以忠实记录跟随 Andrej Karpathy 优秀教程的学习足迹。我们完全没必要去修改这段学习过程的代码，但在您前往主引擎 `femtovllm` 时，理解这里的语境切换至关重要。

Here is a quick mapping of what "block" means here versus the rest of the repository:
以下是“block”在当前目录与整个代码库其他部分中的含义对比：

| Context / 语境 | Term / 术语 | Meaning / 实际含义 |
| :--- | :--- | :--- |
| **nanoGPT (Here)** | `block_size` | Sequence length or context window ($$T$$). / 序列长度或上下文窗口（$$T$$）。 |
| **nanoGPT (Here)** | `Block` | A single Transformer Decoder Layer. / 单个 Transformer 解码器层。 |
| **femtovllm (Engine)** | `block_size` | Number of tokens in a physical KV Cache block (PagedAttention). / 物理 KV Cache 块中包含的 Token 数量。 |
| **csrc (CUDA)** | `kBlockSize` | Number of threads in a CUDA thread block. / CUDA 线程块中的线程数。 |

**Key Takeaway / 核心提示：**
In this reference folder, "block" is about the **model architecture and sequence context** (as taught in the tutorial). In the main engine and CUDA code, "block" strictly refers to **memory management and hardware execution**.
在这个参考文件夹中，“block” 指的是**模型架构与序列上下文**（如教程中所授）。而在主引擎和 CUDA 代码中，“block” 严格指代**显存管理与硬件执行**。

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
