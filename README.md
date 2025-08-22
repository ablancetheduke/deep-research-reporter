# Deep-Research-Reporter

一个最小可运行的管线系统，用于从题目生成**结构化的高质量分析报告**，基于大语言模型（LLM）。  
该 MVP 聚焦 **Closed-Book 模式**（不依赖外部检索），适合初赛，重点在结构、覆盖度、推理与字数控制。后续可扩展为 **Open-Book**（支持检索、事实校验、情景预测）。

---

## 特性
- **题意解析 → 大纲规划 → 分节写作 → 自检修订 → 压缩排版**
- **字数控制**：控制在目标字数 ±10%
- **固定报告框架**：Title、Abstract、Key Takeaways、Body、Risks、Conclusion
- **自检机制**：基于 LLM 的编辑回路，提升逻辑清晰度与可读性
- **CLI 一键生成**：命令行直接生成 Markdown 格式的报告
- **多LLM提供商支持**：OpenAI、Gemini、DeepSeek、ChatGLM
- **MCP集成**：支持Model Context Protocol，可作为工具集成到AI客户端

---

## 安装
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
````

配置 LLM API Key（默认使用 OpenAI）：

```bash
export OPENAI_API_KEY=sk-xxxx
# 或者在项目根目录创建 .env 文件，内容如下：
# OPENAI_API_KEY=sk-xxxx
```

---

## 使用方法

### 基本用法

在命令行中运行：

```bash
python -m drr.cli \
  --topic "Assess the near-term outlook of grid-scale energy storage" \
  --words 1000
```

输出将直接打印在终端，可重定向保存到文件：

```bash
python -m drr.cli --topic "..." --words 1200 > report.md
```

### 多提供商支持

```bash
# 使用 OpenAI
python -m drr.cli -t "AI发展趋势" -w 1000 -p openai -m gpt-4o-mini

# 使用 Gemini
python -m drr.cli -t "量子计算应用" -w 800 -p gemini -m gemini-1.5-flash

# 使用 DeepSeek
python -m drr.cli -t "区块链技术" -w 1200 -p deepseek -m deepseek-chat
```

### MCP 服务器模式

启动 MCP 服务器：

```bash
python start_mcp_server.py
```

使用 MCP 客户端示例：

```bash
python examples/mcp_client_example.py
```

---

## 项目结构

```
deep-research-reporter/
├─ README.md
├─ requirements.txt
├─ start_mcp_server.py
├─ src/
│  └─ drr/
│     ├─ __init__.py
│     ├─ llm.py
│     ├─ prompts.py
│     ├─ pipeline.py
│     ├─ cli.py
│     └─ mcp_server.py
├─ examples/
│  ├─ sample_topic.txt
│  ├─ mcp_config.json
│  └─ mcp_client_example.py
└─ docs/
   └─ MCP_INTEGRATION.md
```

---

## 路线图

* [x] 多模型支持（OpenAI, Gemini, DeepSeek, ChatGLM）
* [x] MCP 集成支持
* [ ] `retrieval.py` —— 支持 Open-Book 检索增强
* [ ] `factcheck.py` —— 主张-证据事实校验
* [ ] `scenarios.py` —— 情景预测与趋势分析（复赛用）
* [ ] 更多 MCP 工具扩展

---

## 许可

MIT License

```
