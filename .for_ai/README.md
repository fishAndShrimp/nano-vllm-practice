# .for_ai / AI Supplementary Context (AI 补充上下文)

This folder represents a personal experiment in "AI-Native" scaffolding.
这是一个关于“AI 原生”脚手架的个人探索与实践。

It captures supplementary context—beyond just code—that is ignored by Git but explicitly injected into the LLM (e.g., via `repomix`) to enhance understanding.
它捕捉了代码之外的补充上下文——这些信息会被 Git 忽略，但会被显式注入到 LLM（如通过 `repomix`）以增强其理解能力。

**The Concept / 核心理念：**

Traditional repos track "Source Code"; this folder tracks **"Contextual Knowledge"**. It can hold system info, local preferences, or any implicit details that make the AI smarter without polluting the project history.
传统的仓库追踪“源代码”；这个文件夹追踪**“上下文知识”**。它可以包含系统信息、本地偏好，或任何能让 AI 变聪明且不污染项目历史的隐性细节。

**Examples / 抛砖引玉：**

*   **Conda Channels**: Including `.condarc` allows the LLM to use the correct mirrors/channels in `environment.yml`.
    **Conda 源**：包含 `.condarc` 后，LLM 可以在生成 `environment.yml` 时使用正确的镜像源。

*   **CUDA Compatibility**: Providing `nvcc_version.txt` ensures the LLM recommends the strictly matching PyTorch CUDA version.
    **CUDA 兼容性**：提供 `nvcc_version.txt` 能确保 LLM 推荐完全匹配的 PyTorch CUDA 版本。

This is just a humble attempt to make the repository more "readable" for AI models.
这只是一个微小的尝试，旨在让代码仓库对 AI 模型更加“可读”。
