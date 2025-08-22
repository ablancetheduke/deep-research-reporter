#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MCP Client Example for Deep Research Reporter

This example demonstrates how to use the MCP client to interact with
the Deep Research Reporter MCP server.
"""

import asyncio
import json
import aiohttp
from typing import Dict, Any


class MCPClient:
    """Simple MCP client for testing."""
    
    def __init__(self, server_url: str = "http://localhost:3001"):
        self.server_url = server_url
        self.request_id = 1
    
    async def call(self, method: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make an MCP call."""
        payload = {
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": method,
            "params": params or {}
        }
        
        self.request_id += 1
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.server_url,
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                return await response.json()
    
    async def initialize(self) -> Dict[str, Any]:
        """Initialize the MCP connection."""
        return await self.call("initialize")
    
    async def list_tools(self) -> Dict[str, Any]:
        """List available tools."""
        return await self.call("tools/list")
    
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a specific tool."""
        return await self.call("tools/call", {
            "name": name,
            "arguments": arguments
        })


async def main():
    """Main example function."""
    client = MCPClient()
    
    print("=== MCP Client Example ===\n")
    
    # 1. 初始化连接
    print("1. 初始化MCP连接...")
    init_result = await client.initialize()
    print(f"初始化结果: {json.dumps(init_result, indent=2, ensure_ascii=False)}\n")
    
    # 2. 列出可用工具
    print("2. 列出可用工具...")
    tools_result = await client.list_tools()
    print(f"可用工具: {json.dumps(tools_result, indent=2, ensure_ascii=False)}\n")
    
    # 3. 健康检查
    print("3. 健康检查...")
    health_result = await client.call_tool("health_check", {})
    print(f"健康检查结果: {json.dumps(health_result, indent=2, ensure_ascii=False)}\n")
    
    # 4. 列出模型
    print("4. 列出可用模型...")
    models_result = await client.call_tool("list_models", {"provider": "all"})
    print(f"模型列表: {json.dumps(models_result, indent=2, ensure_ascii=False)}\n")
    
    # 5. 生成报告（示例）
    print("5. 生成研究报告示例...")
    report_result = await client.call_tool("generate_report", {
        "topic": "人工智能在医疗保健中的应用",
        "word_limit": 500,
        "provider": "openai",
        "model": "gpt-4o-mini"
    })
    
    if "result" in report_result:
        print("报告生成成功！")
        print("报告内容:")
        print(report_result["result"]["content"])
    else:
        print(f"报告生成失败: {json.dumps(report_result, indent=2, ensure_ascii=False)}")


if __name__ == "__main__":
    asyncio.run(main())
