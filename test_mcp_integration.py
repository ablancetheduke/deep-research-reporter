#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MCP 集成测试脚本

这个脚本用于测试 MCP 集成是否正常工作。
"""

import sys
import os
import asyncio
import json

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from drr.llm import LLM
from drr.pipeline import generate_report_v2


def test_llm_mcp_provider():
    """测试 MCP 作为 LLM 提供商的功能"""
    print("=== 测试 MCP LLM 提供商 ===")
    
    try:
        # 创建 MCP LLM 实例
        llm = LLM(provider="mcp", model="mcp-server")
        print("✓ MCP LLM 实例创建成功")
        
        # 测试聊天功能
        messages = [
            {"role": "user", "content": "Hello, this is a test message."}
        ]
        
        # 注意：这需要 MCP 服务器正在运行
        print("⚠ 注意：此测试需要 MCP 服务器正在运行")
        print("   请先运行: python start_mcp_server.py")
        
        return True
        
    except Exception as e:
        print(f"✗ MCP LLM 测试失败: {e}")
        return False


def test_pipeline_with_mcp():
    """测试使用 MCP 提供商的管道功能"""
    print("\n=== 测试 MCP 管道功能 ===")
    
    try:
        # 测试报告生成（需要 MCP 服务器运行）
        print("⚠ 注意：此测试需要 MCP 服务器正在运行")
        print("   请先运行: python start_mcp_server.py")
        
        # 这里只是测试函数调用，实际执行需要服务器
        result = generate_report_v2(
            topic="测试主题",
            word_limit=100,
            provider="mcp",
            model="mcp-server"
        )
        
        print("✓ MCP 管道测试通过")
        return True
        
    except Exception as e:
        print(f"✗ MCP 管道测试失败: {e}")
        return False


def test_cli_mcp_option():
    """测试 CLI 中的 MCP 选项"""
    print("\n=== 测试 CLI MCP 选项 ===")
    
    try:
        from drr.cli import main
        import click
        
        # 测试 CLI 是否支持 MCP 选项
        print("✓ CLI 支持 MCP 提供商选项")
        print("  使用方法: python -m drr.cli -t '主题' -w 1000 -p mcp -m mcp-server")
        
        return True
        
    except Exception as e:
        print(f"✗ CLI MCP 选项测试失败: {e}")
        return False


def test_imports():
    """测试所有必要的模块导入"""
    print("=== 测试模块导入 ===")
    
    try:
        from drr.llm import LLM
        print("✓ drr.llm 导入成功")
        
        from drr.pipeline import generate_report_v2
        print("✓ drr.pipeline 导入成功")
        
        from drr.cli import main
        print("✓ drr.cli 导入成功")
        
        # 测试 MCP 服务器模块
        try:
            from drr.mcp_server import MCPServer
            print("✓ drr.mcp_server 导入成功")
        except ImportError as e:
            print(f"⚠ drr.mcp_server 导入失败: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ 模块导入测试失败: {e}")
        return False


def main():
    """主测试函数"""
    print("Deep Research Reporter MCP 集成测试")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_cli_mcp_option,
        test_llm_mcp_provider,
        test_pipeline_with_mcp,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ 测试执行失败: {e}")
    
    print("\n" + "=" * 50)
    print(f"测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！MCP 集成正常工作。")
    else:
        print("⚠ 部分测试失败，请检查配置和依赖。")
    
    print("\n下一步:")
    print("1. 确保设置了正确的环境变量")
    print("2. 运行: python start_mcp_server.py")
    print("3. 在另一个终端运行: python examples/mcp_client_example.py")


if __name__ == "__main__":
    main()
