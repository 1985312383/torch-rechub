#!/usr/bin/env python3
"""
代码格式化脚本 - Google Python风格
=================================

这个脚本使用YAPF和isort来格式化Python代码。

使用方法:
    python config/format_code.py                # 格式化所有代码
    python config/format_code.py --check       # 只检查格式
    python config/format_code.py --verbose     # 详细输出
"""

import subprocess
import sys
import argparse
from pathlib import Path

def run_command(cmd, description, verbose=False):
    """运行命令并返回结果"""
    if verbose:
        print(f"运行命令: {cmd}")
    
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=True, 
            text=True
        )
        
        if result.returncode == 0:
            print(f"✅ {description}完成")
            if verbose and result.stdout:
                print(f"输出: {result.stdout}")
            return True
        else:
            print(f"❌ {description}失败")
            if result.stderr:
                print(f"错误: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ {description}异常: {e}")
        return False

def check_tools():
    """检查必要工具是否安装"""
    tools = ["yapf", "isort"]
    for tool in tools:
        if not run_command(f"{tool} --version", f"检查{tool}", False):
            print(f"请安装 {tool}: pip install {tool}")
            return False
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Google Python风格代码格式化工具"
    )
    
    parser.add_argument(
        "--check", 
        action="store_true",
        help="只检查格式，不修改文件"
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="显示详细输出"
    )
    
    parser.add_argument(
        "directories",
        nargs="*",
        default=["torch_rechub/", "examples/", "tests/"],
        help="要处理的目录"
    )
    
    args = parser.parse_args()
    
    print("🚀 代码格式化工具启动")
    print("=" * 40)
    
    # 检查工具
    if not check_tools():
        sys.exit(1)
    
    # 检查目录
    existing_dirs = []
    for d in args.directories:
        if Path(d).exists():
            existing_dirs.append(d)
        else:
            print(f"⚠️ 目录 {d} 不存在")
    
    if not existing_dirs:
        print("❌ 没有找到有效目录")
        sys.exit(1)
    
    dirs_str = " ".join(existing_dirs)
    print(f"📂 处理目录: {', '.join(existing_dirs)}")
    
    if args.check:
        print("\n🔍 检查模式 - 验证代码格式")
        success = True
        
        success &= run_command(
            f"yapf --style=config/.style.yapf --diff --recursive {dirs_str}",
            "YAPF格式检查",
            args.verbose
        )
        
        success &= run_command(
            f"isort --check-only --diff {dirs_str}",
            "isort导入检查", 
            args.verbose
        )
        
        if success:
            print("\n🎉 所有格式检查通过!")
            sys.exit(0)
        else:
            print("\n❌ 格式检查失败，请运行格式化:")
            print("python config/format_code.py")
            sys.exit(1)
    
    else:
        print("\n🔧 格式化模式 - 修改代码格式")
        
        # YAPF格式化
        if not run_command(
            f"yapf --style=config/.style.yapf --in-place --recursive {dirs_str}",
            "YAPF代码格式化",
            args.verbose
        ):
            sys.exit(1)
        
        # isort导入排序
        if not run_command(
            f"isort {dirs_str}",
            "isort导入排序",
            args.verbose
        ):
            sys.exit(1)
        
        print("\n🎉 代码格式化完成!")

if __name__ == "__main__":
    main()