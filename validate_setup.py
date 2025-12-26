"""
Project validation and setup script.
"""
import os
import sys


def check_structure():
    """Check if all required directories and files exist."""
    print("\n" + "="*60)
    print("Checking project structure...")
    print("="*60 + "\n")
    
    required_files = [
        'README.md',
        'QUICKSTART.md',
        'DEVELOPMENT.md',
        'requirements.txt',
        'train.py',
        'test.py',
        'evaluate.py',
        'demo.py',
        'config.py',
        'agents/__init__.py',
        'agents/base_agent.py',
        'agents/dqn_agent.py',
        'agents/qlearning_agent.py',
        'utils/__init__.py',
        'utils/logger.py',
        'utils/plotter.py',
        'utils/visualize_qtable.py',
        'experiments/cartpole/README.md',
        'experiments/pong/README.md',
        'experiments/frozenlake/README.md',
    ]
    
    missing = []
    for filepath in required_files:
        full_path = os.path.join('.', filepath)
        if not os.path.exists(full_path):
            missing.append(filepath)
            print(f"✗ {filepath}")
        else:
            print(f"✓ {filepath}")
    
    print("\n" + "="*60)
    if missing:
        print(f"Missing {len(missing)} file(s)!")
        return False
    else:
        print("All required files exist! ✓")
        return True


def check_dependencies():
    """Check if all required packages are installed."""
    print("\n" + "="*60)
    print("Checking dependencies...")
    print("="*60 + "\n")
    
    required_packages = [
        'torch',
        'gymnasium',
        'numpy',
        'matplotlib',
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - NOT INSTALLED")
            missing.append(package)
    
    print("\n" + "="*60)
    if missing:
        print(f"\nMissing {len(missing)} package(s)!")
        print("\nTo install missing packages, run:")
        print(f"  pip install {' '.join(missing)}")
        print("\nOr install all dependencies:")
        print("  pip install -r requirements.txt")
        return False
    else:
        print("All dependencies installed! ✓")
        return True


def test_imports():
    """Test if core modules can be imported."""
    print("\n" + "="*60)
    print("Testing imports...")
    print("="*60 + "\n")
    
    try:
        from agents import BaseAgent, DQNAgent, QLearningAgent
        print("✓ agents module")
        
        from utils import Logger, Plotter
        print("✓ utils module")
        
        print("\n" + "="*60)
        print("All imports successful! ✓")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("\n" + "="*60)
        return False


def create_directories():
    """Create necessary directories."""
    print("\n" + "="*60)
    print("Creating directories...")
    print("="*60 + "\n")
    
    directories = [
        'models',
        'results/logs',
        'results/plots',
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ {directory}")
    
    print("\n" + "="*60)
    print("Directories created! ✓")


def main():
    """Run all validation checks."""
    print("\n\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*10 + "RL Game Package Setup Validation" + " "*16 + "║")
    print("╚" + "="*58 + "╝")
    
    # Check structure
    structure_ok = check_structure()
    
    # Create directories
    create_directories()
    
    # Check dependencies
    deps_ok = check_dependencies()
    
    # Test imports
    imports_ok = test_imports()
    
    # Summary
    print("\n\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*20 + "Summary" + " "*31 + "║")
    print("║" + "="*58 + "║")
    
    checks = [
        ("Project structure", structure_ok),
        ("Dependencies", deps_ok),
        ("Module imports", imports_ok),
    ]
    
    for check_name, status in checks:
        status_str = "✓ PASS" if status else "✗ FAIL"
        print(f"║ {check_name:.<40} {status_str:>15} ║")
    
    all_ok = all(status for _, status in checks)
    
    print("║" + "="*58 + "║")
    if all_ok:
        print("║" + " "*15 + "✓ Setup successful! Ready to train!" + " "*5 + "║")
        print("║" + "="*58 + "║")
        print("\nNext steps:")
        print("  1. Read QUICKSTART.md for quick start guide")
        print("  2. Run: python demo.py (for a quick test)")
        print("  3. Train: python train.py --game cartpole --episodes 500")
        print("  4. Test: python test.py --game cartpole --model models/cartpole_best.pth")
        print("\n")
        return 0
    else:
        print("║" + " "*10 + "✗ Setup incomplete! Please fix errors." + " "*8 + "║")
        print("║" + "="*58 + "║")
        print("\n")
        return 1


if __name__ == '__main__':
    sys.exit(main())
