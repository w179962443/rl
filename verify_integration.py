"""
验证脚本 - 检查项目集成是否成功
"""

import os
import sys


def check_file_exists(filepath, description):
    """检查文件是否存在"""
    exists = os.path.exists(filepath)
    status = "✓" if exists else "✗"
    print(f"{status} {description}: {filepath}")
    return exists


def check_directory_exists(dirpath, description):
    """检查目录是否存在"""
    exists = os.path.isdir(dirpath)
    status = "✓" if exists else "✗"
    print(f"{status} {description}: {dirpath}")
    return exists


def main():
    """主验证函数"""
    print("=" * 70)
    print("项目集成验证")
    print("=" * 70)
    
    all_checks = []
    
    # 1. 检查Flappy Bird环境文件
    print("\n1. Flappy Bird 环境文件:")
    all_checks.append(check_file_exists("envs/flappybird_env.py", "Flappy Bird 环境"))
    all_checks.append(check_file_exists("envs/__init__.py", "环境包初始化"))
    
    # 2. 检查Flappy Bird实验目录
    print("\n2. Flappy Bird 实验目录:")
    all_checks.append(check_directory_exists("experiments/flappybird", "实验目录"))
    all_checks.append(check_directory_exists("experiments/flappybird/models", "模型目录"))
    all_checks.append(check_directory_exists("experiments/flappybird/logs", "日志目录"))
    
    # 3. 检查Flappy Bird脚本
    print("\n3. Flappy Bird 脚本:")
    all_checks.append(check_file_exists("experiments/flappybird/__init__.py", "模块初始化"))
    all_checks.append(check_file_exists("experiments/flappybird/train.py", "训练脚本"))
    all_checks.append(check_file_exists("experiments/flappybird/test.py", "测试脚本"))
    all_checks.append(check_file_exists("experiments/flappybird/README.md", "说明文档"))
    
    # 4. 检查快速启动脚本
    print("\n4. 快速启动脚本:")
    all_checks.append(check_file_exists("train_flappybird.bat", "训练批处理"))
    all_checks.append(check_file_exists("test_flappybird.bat", "测试批处理"))
    
    # 5. 检查主脚本
    print("\n5. 主脚本:")
    all_checks.append(check_file_exists("train.py", "主训练脚本"))
    all_checks.append(check_file_exists("test.py", "主测试脚本"))
    
    # 6. 检查Agent文件
    print("\n6. Agent 文件:")
    all_checks.append(check_file_exists("agents/dqn_agent.py", "DQN Agent"))
    
    # 7. 检查文档
    print("\n7. 文档文件:")
    all_checks.append(check_file_exists("README.md", "项目README"))
    all_checks.append(check_file_exists("FLAPPYBIRD_INTEGRATION.md", "集成文档"))
    all_checks.append(check_file_exists("requirements.txt", "依赖文件"))
    
    # 8. 检查导入
    print("\n8. Python 导入检查:")
    try:
        sys.path.insert(0, os.getcwd())
        
        # 检查envs包
        from envs import FlappyBirdEnv
        print("✓ FlappyBirdEnv 导入成功")
        all_checks.append(True)
        
        # 检查agents包
        from agents import DQNAgent
        print("✓ DQNAgent 导入成功")
        all_checks.append(True)
        
        # 检查experiments.flappybird
        from experiments.flappybird import train_flappybird, test_flappybird
        print("✓ Flappy Bird 训练/测试函数导入成功")
        all_checks.append(True)
        
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        all_checks.append(False)
    
    # 9. 检查DQN Agent的新方法
    print("\n9. DQN Agent 方法检查:")
    try:
        from agents import DQNAgent
        import numpy as np
        
        # 创建测试agent
        config = {
            "gamma": 0.99,
            "learning_rate": 0.001,
            "batch_size": 32,
            "memory_size": 1000,
            "target_update_freq": 10,
            "hidden_sizes": [64],
            "epsilon_start": 1.0,
            "epsilon_end": 0.01,
            "epsilon_decay": 0.995,
        }
        agent = DQNAgent(state_size=4, action_size=2, config=config)
        
        # 检查属性
        has_epsilon = hasattr(agent, 'epsilon')
        has_epsilon_min = hasattr(agent, 'epsilon_min')
        has_epsilon_decay = hasattr(agent, 'epsilon_decay')
        has_config = hasattr(agent, 'config')
        
        print(f"{'✓' if has_epsilon else '✗'} epsilon 属性")
        print(f"{'✓' if has_epsilon_min else '✗'} epsilon_min 属性")
        print(f"{'✓' if has_epsilon_decay else '✗'} epsilon_decay 属性")
        print(f"{'✓' if has_config else '✗'} config 属性")
        
        all_checks.extend([has_epsilon, has_epsilon_min, has_epsilon_decay, has_config])
        
        # 检查方法
        has_train = hasattr(agent, 'train') and callable(getattr(agent, 'train'))
        has_update_target = hasattr(agent, 'update_target_network') and callable(getattr(agent, 'update_target_network'))
        
        print(f"{'✓' if has_train else '✗'} train() 方法")
        print(f"{'✓' if has_update_target else '✗'} update_target_network() 方法")
        
        all_checks.extend([has_train, has_update_target])
        
    except Exception as e:
        print(f"✗ Agent 方法检查失败: {e}")
        all_checks.append(False)
    
    # 总结
    print("\n" + "=" * 70)
    passed = sum(all_checks)
    total = len(all_checks)
    success_rate = (passed / total * 100) if total > 0 else 0
    
    print(f"验证结果: {passed}/{total} 通过 ({success_rate:.1f}%)")
    
    if passed == total:
        print("✓ 所有检查通过！项目集成成功！")
        return 0
    else:
        print(f"✗ {total - passed} 项检查失败，请检查相关文件")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
