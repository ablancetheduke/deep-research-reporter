#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MCP Server 启动脚本

这个脚本用于启动 Deep Research Reporter 的 MCP 服务器。
"""

import asyncio
import sys
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from drr.mcp_server import main

if __name__ == "__main__":
    print("启动 Deep Research Reporter MCP 服务器...")
    print("服务器将在 http://localhost:3000 上运行")
    print("按 Ctrl+C 停止服务器")
    print("-" * 50)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n服务器已停止")
    except Exception as e:
        print(f"启动失败: {e}")
        sys.exit(1)
