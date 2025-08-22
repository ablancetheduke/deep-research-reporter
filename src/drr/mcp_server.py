# src/drr/mcp_server.py
# -*- coding: utf-8 -*-

"""
MCP (Model Context Protocol) Server implementation for Deep Research Reporter.

This module provides a server that implements the MCP protocol, allowing
the Deep Research Reporter to be used as a tool in MCP-compatible clients.
"""

import json
import asyncio
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from aiohttp import web

from .pipeline import generate_report_v2
from .llm import LLM

logger = logging.getLogger(__name__)


@dataclass
class MCPRequest:
    """Represents an MCP request."""
    jsonrpc: str
    id: int
    method: str
    params: Dict[str, Any]


@dataclass
class MCPResponse:
    """Represents an MCP response."""
    jsonrpc: str = "2.0"
    id: Optional[int] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        response = {"jsonrpc": self.jsonrpc}
        if self.id is not None:
            response["id"] = self.id
        if self.result is not None:
            response["result"] = self.result
        if self.error is not None:
            response["error"] = self.error
        return response


class MCPServer:
    """MCP Server implementation for Deep Research Reporter."""

    def __init__(self):
        self.tools = {
            "generate_report": self._generate_report,
            "list_models": self._list_models,
            "health_check": self._health_check,
        }

    async def handle_request(self, request_data: Dict[str, Any]) -> MCPResponse:
        """Handle an incoming MCP request."""
        try:
            request = MCPRequest(
                jsonrpc=request_data.get("jsonrpc", "2.0"),
                id=request_data.get("id"),
                method=request_data.get("method", ""),
                params=request_data.get("params", {})
            )

            if request.method == "initialize":
                return self._handle_initialize(request)
            elif request.method == "tools/list":
                return self._handle_list_tools(request)
            elif request.method == "tools/call":
                return await self._handle_tool_call(request)
            else:
                return MCPResponse(
                    id=request.id,
                    error={"code": -32601, "message": f"Method not found: {request.method}"}
                )

        except Exception as e:
            logger.error(f"Error handling request: {e}")
            return MCPResponse(
                id=request_data.get("id"),
                error={"code": -32603, "message": f"Internal error: {str(e)}"}
            )

    def _handle_initialize(self, request: MCPRequest) -> MCPResponse:
        """Handle initialization request."""
        return MCPResponse(
            id=request.id,
            result={
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {}
                },
                "serverInfo": {
                    "name": "deep-research-reporter",
                    "version": "1.0.0"
                }
            }
        )

    def _handle_list_tools(self, request: MCPRequest) -> MCPResponse:
        """Handle tools listing request."""
        tools = [
            {
                "name": "generate_report",
                "description": "Generate a structured analysis report on a given topic",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "topic": {
                            "type": "string",
                            "description": "The topic to research and report on"
                        },
                        "word_limit": {
                            "type": "integer",
                            "description": "Target word count for the report",
                            "default": 1000
                        },
                        "provider": {
                            "type": "string",
                            "enum": ["openai", "gemini", "deepseek", "chatglm", "mcp"],
                            "description": "LLM provider to use",
                            "default": "openai"
                        },
                        "model": {
                            "type": "string",
                            "description": "Model name for the chosen provider",
                            "default": "gpt-4o-mini"
                        }
                    },
                    "required": ["topic"]
                }
            },
            {
                "name": "list_models",
                "description": "List available models for each provider",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "provider": {
                            "type": "string",
                            "enum": ["openai", "gemini", "deepseek", "chatglm", "mcp"],
                            "description": "Provider to list models for"
                        }
                    }
                }
            },
            {
                "name": "health_check",
                "description": "Check the health status of the server",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            }
        ]

        return MCPResponse(
            id=request.id,
            result={"tools": tools}
        )

    async def _handle_tool_call(self, request: MCPRequest) -> MCPResponse:
        """Handle tool call request."""
        tool_name = request.params.get("name")
        arguments = request.params.get("arguments", {})

        if tool_name not in self.tools:
            return MCPResponse(
                id=request.id,
                error={"code": -32601, "message": f"Tool not found: {tool_name}"}
            )

        try:
            # 对于异步工具调用
            if asyncio.iscoroutinefunction(self.tools[tool_name]):
                result = await self.tools[tool_name](arguments)
            else:
                result = self.tools[tool_name](arguments)

            return MCPResponse(
                id=request.id,
                result={"content": result}
            )

        except Exception as e:
            logger.error(f"Error in tool {tool_name}: {e}")
            return MCPResponse(
                id=request.id,
                error={"code": -32603, "message": f"Tool execution error: {str(e)}"}
            )

    def _generate_report(self, arguments: Dict[str, Any]) -> str:
        """Generate a research report."""
        topic = arguments.get("topic")
        if not topic:
            raise ValueError("Topic is required")

        word_limit = arguments.get("word_limit", 1000)
        provider = arguments.get("provider", "openai")
        model = arguments.get("model", "gpt-4o-mini")

        return generate_report_v2(
            topic=topic,
            word_limit=word_limit,
            provider=provider,
            model=model
        )

    def _list_models(self, arguments: Dict[str, Any]) -> str:
        """List available models for a provider."""
        provider = arguments.get("provider", "all")
        
        models = {
            "openai": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
            "gemini": ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-pro"],
            "deepseek": ["deepseek-chat", "deepseek-coder"],
            "chatglm": ["glm-4", "glm-3-turbo"],
            "mcp": ["mcp-server"]
        }

        if provider == "all":
            return json.dumps(models, indent=2, ensure_ascii=False)
        elif provider in models:
            return json.dumps({provider: models[provider]}, indent=2, ensure_ascii=False)
        else:
            return json.dumps({"error": f"Unknown provider: {provider}"}, ensure_ascii=False)

    def _health_check(self, arguments: Dict[str, Any]) -> str:
        """Check server health."""
        return json.dumps({
            "status": "healthy",
            "version": "1.0.0",
            "tools_available": list(self.tools.keys())
        }, ensure_ascii=False)

async def handle_root(request):
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>MCP Server Test</title>
    </head>
    <body>
        <h2>MCP Server Test Page</h2>
        <button onclick="sendHealthCheck()">Health Check</button>
        <pre id="output"></pre>

        <script>
        async function sendHealthCheck() {
            const response = await fetch("/", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    jsonrpc: "2.0",
                    id: 1,
                    method: "tools/call",
                    params: {name: "health_check",
                arguments: {}
                    }
                })
            });

            const result = await response.json();
            document.getElementById("output").textContent = JSON.stringify(result, null, 2);
        }
        </script>
    </body>
    </html>
    """
    return web.Response(text=html, content_type='text/html')



async def main():
    """Main server function."""
    server = MCPServer()
    
    # 简单的HTTP服务器实现
    import aiohttp
    from aiohttp import web

    async def handle_mcp(request):
        try:
            data = await request.json()
            response = await server.handle_request(data)
            return web.json_response(
                response.to_dict(),
                headers={
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Methods': 'POST, OPTIONS',
                    'Access-Control-Allow-Headers': 'Content-Type',
                }
            )
        except Exception as e:
            return web.json_response(
                {
                    "jsonrpc": "2.0",
                    "error": {"code": -32603, "message": str(e)}
                },
                headers={
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Methods': 'POST, OPTIONS',
                    'Access-Control-Allow-Headers': 'Content-Type',
                }
            )

    app = web.Application()
    app.router.add_post("/", handle_mcp)
    
    # 添加CORS支持
    async def handle_options(request):
        return web.Response(
            headers={
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'POST, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type',
            }
        )
    
    app.router.add_options("/", handle_options)
    app.router.add_get("/", handle_root)
    
    runner = web.AppRunner(app)
    await runner.setup()
    
    site = web.TCPSite(runner, "localhost", 3001)
    await site.start()
    
    print("MCP Server running on http://localhost:3001")
    print("Press Ctrl+C to stop")
    
    try:
        await asyncio.Future()  # 保持运行
    except KeyboardInterrupt:
        await runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())


# HTML 页面
async def handle_root(request):
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>MCP Server Test</title>
    </head>
    <body>
        <h2>MCP Server Test Page</h2>
        <button onclick="sendHealthCheck()">Health Check</button>
        <pre id="output"></pre>

        <script>
        async function sendHealthCheck() {
            const response = await fetch("/", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    jsonrpc: "2.0",
                    id: 1,
                    method: "health_check",
                    params: {}
                })
            });

            const result = await response.json();
            document.getElementById("output").textContent = JSON.stringify(result, null, 2);
        }
        </script>
    </body>
    </html>
    """
    return web.Response(text=html, content_type='text/html')



