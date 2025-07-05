#!/usr/bin/env python3
"""
论文对比实验快速启动脚本
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """运行命令并显示进度"""
    print(f"\n{'='*50}")
    print(f"🚀 {description}")
    print(f"{'='*50}")
    print(f"执行命令: {cmd}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ 执行成功")
        print(result.stdout)
    else:
        print("❌ 执行失败")
        print("错误输出:")
        print(result.stderr)
    
    return result.returncode == 0

def main():
    """主函数"""
    print("M²-MOEP 论文对比实验快速启动")
    print("目标论文: Non-autoregressive Conditional Diffusion Models for Time Series Prediction")
    print()
    
    # 步骤1: 检查数据集
    print("步骤1: 检查数据集可用性")
    if not run_command("python check_datasets.py", "检查数据集"):
        print("❌ 数据集检查失败，请检查数据集目录")
        return
    
    # 询问用户是否继续
    print("\n" + "="*50)
    response = input("是否继续运行论文对比实验? (y/n): ").strip().lower()
    
    if response != 'y':
        print("实验取消")
        return
    
    # 询问实验类型
    print("\n选择实验类型:")
    print("1. 快速测试 (10 epochs)")
    print("2. 标准实验 (50 epochs)")
    print("3. 深度实验 (100 epochs)")
    print("4. 只运行多变量实验")
    print("5. 自定义")
    
    choice = input("请选择 (1-5): ").strip()
    
    if choice == '1':
        cmd = "python paper_comparison_experiment.py --epochs 10"
        desc = "快速测试实验"
    elif choice == '2':
        cmd = "python paper_comparison_experiment.py --epochs 50"
        desc = "标准对比实验"
    elif choice == '3':
        cmd = "python paper_comparison_experiment.py --epochs 100"
        desc = "深度对比实验"
    elif choice == '4':
        cmd = "python paper_comparison_experiment.py --modes multivariate"
        desc = "多变量对比实验"
    elif choice == '5':
        epochs = input("输入训练轮数 (默认50): ").strip() or "50"
        modes = input("输入模式 (multivariate/univariate/both, 默认both): ").strip()
        datasets = input("输入数据集 (用空格分隔，默认全部): ").strip()
        
        cmd = f"python paper_comparison_experiment.py --epochs {epochs}"
        if modes == "multivariate":
            cmd += " --modes multivariate"
        elif modes == "univariate":
            cmd += " --modes univariate"
        if datasets:
            cmd += f" --datasets {datasets}"
        
        desc = "自定义对比实验"
    else:
        print("无效选择")
        return
    
    # 步骤2: 运行实验
    print(f"\n步骤2: 运行实验")
    if run_command(cmd, desc):
        print("\n🎉 实验完成！")
        print("生成的文件:")
        print("- paper_comparison_results_*.json (详细结果)")
        print("- paper_comparison_report_*.txt (对比报告)")
        print("- comparison_table_multivariate.tex (LaTeX表格)")
        print("- paper_comparison_*.log (实验日志)")
    else:
        print("\n❌ 实验失败，请检查日志文件")

if __name__ == "__main__":
    main() 