# MCP (Model Context Protocol) 集成指南

本文档介绍如何在 Deep Research Reporter 中集成和使用 MCP (Model Context Protocol)。

## 概述

MCP 是一个标准化的协议，允许 AI 助手与外部工具和服务进行交互。通过集成 MCP，您可以将 Deep Research Reporter 作为工具提供给支持 MCP 的 AI 客户端。

## 功能特性

- **多提供商支持**: 支持 OpenAI、Gemini、DeepSeek、ChatGLM 和 MCP
- **工具化接口**: 提供标准化的工具调用接口
- **异步处理**: 支持异步请求处理
- **健康检查**: 内置服务器健康状态监控
- **模型管理**: 支持列出和管理不同提供商的模型

## 安装和设置

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 环境变量配置

创建 `.env` 文件并配置以下环境变量：

```env
# MCP 服务器配置
MCP_SERVER_URL=http://localhost:3000
MCP_API_KEY=your-api-key-here  # 可选

# 其他 LLM 提供商配置
OPENAI_API_KEY=your-openai-key
GEMINI_API_KEY=your-gemini-key
DEEPSEEK_API_KEY=your-deepseek-key
CHATGLM_API_KEY=your-chatglm-key
```

## 使用方法

### 1. 启动 MCP 服务器

```bash
# 启动 MCP 服务器
python -m drr.mcp_server
```

服务器将在 `http://localhost:3000` 上运行。

### 2. 使用 CLI 与 MCP 交互

```bash
# 使用 MCP 提供商生成报告
python -m drr.cli -t "人工智能发展趋势" -w 1000 -p mcp -m mcp-server -o report.md
```

### 3. 使用 MCP 客户端示例

```bash
# 运行客户端示例
python examples/mcp_client_example.py
```

## API 参考

### 可用工具

#### 1. generate_report

生成结构化分析报告。

**参数:**
- `topic` (string, 必需): 研究主题
- `word_limit` (integer, 可选): 目标字数，默认 1000
- `provider` (string, 可选): LLM 提供商，默认 "openai"
- `model` (string, 可选): 模型名称，默认 "gpt-4o-mini"

**示例:**
```json
{
  "name": "generate_report",
  "arguments": {
    "topic": "量子计算的发展现状",
    "word_limit": 800,
    "provider": "gemini",
    "model": "gemini-1.5-flash"
  }
}
```

#### 2. list_models

列出指定提供商的可用模型。

**参数:**
- `provider` (string, 可选): 提供商名称，默认 "all"

**示例:**
```json
{
  "name": "list_models",
  "arguments": {
    "provider": "openai"
  }
}
```

#### 3. health_check

检查服务器健康状态。

**参数:** 无

**示例:**
```json
{
  "name": "health_check",
  "arguments": {}
}
```

### MCP 协议方法

#### initialize

初始化 MCP 连接。

#### tools/list

列出所有可用工具。

#### tools/call

调用指定工具。

## 集成到其他 MCP 客户端

### 1. 配置 MCP 客户端

在您的 MCP 客户端配置文件中添加：

```json
{
  "mcpServers": {
    "deep-research-reporter": {
      "command": "python",
      "args": ["-m", "drr.mcp_server"],
      "env": {
        "MCP_SERVER_URL": "http://localhost:3000",
        "OPENAI_API_KEY": "your-openai-key"
      }
    }
  }
}
```

### 2. 在 AI 客户端中使用

```python
# 示例：在支持 MCP 的 AI 客户端中调用
response = await mcp_client.call_tool("generate_report", {
    "topic": "区块链技术在供应链管理中的应用",
    "word_limit": 1200,
    "provider": "openai"
})
```

## 错误处理

### 常见错误代码

- `-32601`: 方法未找到
- `-32602`: 无效参数
- `-32603`: 内部错误
- `-32700`: 解析错误

### 错误响应格式

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "error": {
    "code": -32603,
    "message": "Internal error: Tool execution failed"
  }
}
```

## 开发和扩展

### 添加新工具

1. 在 `MCPServer` 类的 `__init__` 方法中注册新工具
2. 实现工具方法
3. 在 `_handle_list_tools` 方法中添加工具描述

### 自定义 MCP 服务器

您可以创建自定义的 MCP 服务器实现：

```python
from drr.mcp_server import MCPServer

class CustomMCPServer(MCPServer):
    def __init__(self):
        super().__init__()
        # 添加自定义工具
        self.tools["custom_tool"] = self._custom_tool
    
    def _custom_tool(self, arguments):
        # 实现自定义工具逻辑
        return "Custom tool result"
```

## 故障排除

### 1. 服务器无法启动

- 检查端口 3000 是否被占用
- 确认所有依赖已正确安装
- 检查环境变量配置

### 2. 工具调用失败

- 验证 API 密钥配置
- 检查网络连接
- 查看服务器日志

### 3. 性能问题

- 调整 `max_tokens` 参数
- 使用更快的模型
- 优化网络连接

## 贡献

欢迎提交 Issue 和 Pull Request 来改进 MCP 集成功能。

## 许可证

本项目采用 MIT 许可证。
