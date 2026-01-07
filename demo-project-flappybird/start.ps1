# Flappy Bird AI - PowerShell 启动脚本
# 编码: UTF-8

# 设置错误处理
$ErrorActionPreference = "Continue"

# 设置控制台
$Host.UI.RawUI.WindowTitle = "Flappy Bird AI - 启动脚本"
try {
    $Host.UI.RawUI.BackgroundColor = "Black"
    $Host.UI.RawUI.ForegroundColor = "Green"
} catch {
    # 忽略颜色设置错误
}
Clear-Host

# 函数：显示菜单
function Show-Menu {
    Clear-Host
    Write-Host "╔════════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
    Write-Host "║           Flappy Bird - 强化学习AI 启动脚本                   ║" -ForegroundColor Cyan
    Write-Host "╚════════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  [1] 安装环境" -ForegroundColor Yellow
    Write-Host "  [2] 训练AI" -ForegroundColor Yellow
    Write-Host "  [3] 测试AI（自动模式）" -ForegroundColor Yellow
    Write-Host "  [4] 测试AI（交互模式）" -ForegroundColor Yellow
    Write-Host "  [5] 手动玩游戏" -ForegroundColor Yellow
    Write-Host "  [0] 退出" -ForegroundColor Red
    Write-Host ""
    Write-Host "================================================================" -ForegroundColor Gray
}

# 函数：检查conda是否安装
function Test-CondaInstalled {
    $conda = Get-Command conda -ErrorAction SilentlyContinue
    return $null -ne $conda
}

# 函数：检查conda环境是否存在
function Test-CondaEnvExists {
    param([string]$EnvName)
    
    $envList = conda env list 2>$null | Out-String
    return $envList -match $EnvName
}

# 函数：安装环境
function Install-Environment {
    Clear-Host
    Write-Host "════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host "  正在检查和安装环境..." -ForegroundColor Cyan
    Write-Host "════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host ""
    
    # 检查conda
    if (-not (Test-CondaInstalled)) {
        Write-Host "[错误] 未检测到 Conda！" -ForegroundColor Red
        Write-Host ""
        Write-Host "请先安装 Anaconda 或 Miniconda:" -ForegroundColor Yellow
        Write-Host "https://www.anaconda.com/download" -ForegroundColor Cyan
        Write-Host "或" -ForegroundColor Yellow
        Write-Host "https://docs.conda.io/en/latest/miniconda.html" -ForegroundColor Cyan
        Write-Host ""
        pause
        return
    }
    
    Write-Host "[✓] Conda 已安装" -ForegroundColor Green
    Write-Host ""
    
    # 检查环境是否存在
    if (Test-CondaEnvExists -EnvName "flappy-bird") {
        Write-Host "[!] 检测到 flappy-bird 环境已存在" -ForegroundColor Yellow
        Write-Host ""
        $recreate = Read-Host "是否要删除并重新创建? (y/n)"
        
        if ($recreate -eq "y" -or $recreate -eq "Y") {
            Write-Host ""
            Write-Host "[→] 正在删除旧环境..." -ForegroundColor Yellow
            conda deactivate 2>$null
            conda env remove -n flappy-bird -y
            Write-Host "[✓] 旧环境已删除" -ForegroundColor Green
        }
        else {
            Write-Host ""
            Write-Host "[→] 跳过环境创建，准备安装依赖包..." -ForegroundColor Yellow
            Install-Packages
            return
        }
    }
    
    # 创建环境
    Write-Host ""
    Write-Host "[→] 正在创建 flappy-bird conda 环境 (Python 3.10)..." -ForegroundColor Yellow
    conda create -n flappy-bird python=3.10 -y
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[错误] 环境创建失败！" -ForegroundColor Red
        pause
        return
    }
    
    Write-Host "[✓] 环境创建成功" -ForegroundColor Green
    
    Install-Packages
}

# 函数：安装依赖包
function Install-Packages {
    Write-Host ""
    Write-Host "[→] 正在激活环境..." -ForegroundColor Yellow
    
    # 激活conda环境
    $condaPath = (Get-Command conda).Source
    $condaRoot = Split-Path (Split-Path $condaPath)
    $activateScript = Join-Path $condaRoot "Scripts\activate.bat"
    
    Write-Host "[✓] 环境已激活" -ForegroundColor Green
    
    Write-Host ""
    Write-Host "[→] 正在安装依赖包..." -ForegroundColor Yellow
    Write-Host ""
    
    # 使用conda run安装包
    conda run -n flappy-bird pip install pygame torch numpy matplotlib
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "[错误] 依赖包安装失败！" -ForegroundColor Red
        pause
        return
    }
    
    Write-Host ""
    Write-Host "════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host "[✓] 环境安装完成！" -ForegroundColor Green
    Write-Host "════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "环境名称: flappy-bird" -ForegroundColor White
    Write-Host "Python版本: 3.10" -ForegroundColor White
    Write-Host "已安装: pygame, torch, numpy, matplotlib" -ForegroundColor White
    Write-Host ""
    pause
}

# 函数：训练AI
function Start-Training {
    Clear-Host
    Write-Host "════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host "  启动训练..." -ForegroundColor Cyan
    Write-Host "════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host ""
    
    if (-not (Test-CondaEnvExists -EnvName "flappy-bird")) {
        Write-Host "[错误] 未找到 flappy-bird 环境！" -ForegroundColor Red
        Write-Host "请先运行 [1] 安装环境" -ForegroundColor Yellow
        Write-Host ""
        pause
        return
    }
    
    Write-Host "[→] 正在激活 flappy-bird 环境..." -ForegroundColor Yellow
    Write-Host "[→] 正在启动训练..." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "----------------------------------------------------------------" -ForegroundColor Gray
    
    conda run -n flappy-bird python train.py
    
    Write-Host "----------------------------------------------------------------" -ForegroundColor Gray
    Write-Host ""
    Write-Host "训练结束！" -ForegroundColor Green
    Write-Host ""
    pause
}

# 函数：测试AI（自动模式）
function Start-Test {
    Clear-Host
    Write-Host "════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host "  启动测试（自动模式）..." -ForegroundColor Cyan
    Write-Host "════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host ""
    
    if (-not (Test-CondaEnvExists -EnvName "flappy-bird")) {
        Write-Host "[错误] 未找到 flappy-bird 环境！" -ForegroundColor Red
        Write-Host "请先运行 [1] 安装环境" -ForegroundColor Yellow
        Write-Host ""
        pause
        return
    }
    
    if (-not (Test-Path "models\best_model.pth")) {
        Write-Host "[错误] 未找到训练好的模型！" -ForegroundColor Red
        Write-Host "请先运行 [2] 训练AI" -ForegroundColor Yellow
        Write-Host ""
        pause
        return
    }
    
    Write-Host "[→] 正在激活 flappy-bird 环境..." -ForegroundColor Yellow
    Write-Host "[→] 正在加载模型并测试..." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "----------------------------------------------------------------" -ForegroundColor Gray
    
    conda run -n flappy-bird python test.py --episodes 10
    
    Write-Host "----------------------------------------------------------------" -ForegroundColor Gray
    Write-Host ""
    Write-Host "测试结束！" -ForegroundColor Green
    Write-Host ""
    pause
}

# 函数：测试AI（交互模式）
function Start-InteractiveTest {
    Clear-Host
    Write-Host "════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host "  启动测试（交互模式）..." -ForegroundColor Cyan
    Write-Host "════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host ""
    
    if (-not (Test-CondaEnvExists -EnvName "flappy-bird")) {
        Write-Host "[错误] 未找到 flappy-bird 环境！" -ForegroundColor Red
        Write-Host "请先运行 [1] 安装环境" -ForegroundColor Yellow
        Write-Host ""
        pause
        return
    }
    
    if (-not (Test-Path "models\best_model.pth")) {
        Write-Host "[错误] 未找到训练好的模型！" -ForegroundColor Red
        Write-Host "请先运行 [2] 训练AI" -ForegroundColor Yellow
        Write-Host ""
        pause
        return
    }
    
    Write-Host "[→] 正在激活 flappy-bird 环境..." -ForegroundColor Yellow
    Write-Host "[→] 正在启动交互模式..." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "控制说明:" -ForegroundColor Yellow
    Write-Host "  SPACE - 跳跃（人类模式）" -ForegroundColor White
    Write-Host "  A     - 切换 AI/人类 模式" -ForegroundColor White
    Write-Host "  R     - 重新开始" -ForegroundColor White
    Write-Host "  ESC   - 退出" -ForegroundColor White
    Write-Host ""
    Write-Host "----------------------------------------------------------------" -ForegroundColor Gray
    
    conda run -n flappy-bird python test.py --interactive
    
    Write-Host "----------------------------------------------------------------" -ForegroundColor Gray
    Write-Host ""
    pause
}

# 函数：手动玩游戏
function Start-Game {
    Clear-Host
    Write-Host "════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host "  手动玩游戏..." -ForegroundColor Cyan
    Write-Host "════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host ""
    
    if (-not (Test-CondaEnvExists -EnvName "flappy-bird")) {
        Write-Host "[错误] 未找到 flappy-bird 环境！" -ForegroundColor Red
        Write-Host "请先运行 [1] 安装环境" -ForegroundColor Yellow
        Write-Host ""
        pause
        return
    }
    
    Write-Host "[→] 正在激活 flappy-bird 环境..." -ForegroundColor Yellow
    Write-Host "[→] 正在启动游戏..." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "按 SPACE 键跳跃，关闭窗口退出" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "----------------------------------------------------------------" -ForegroundColor Gray
    
    conda run -n flappy-bird python game.py
    
    Write-Host "----------------------------------------------------------------" -ForegroundColor Gray
    Write-Host ""
    pause
}

# 主循环
do {
    Show-Menu
    $choice = Read-Host "请选择操作 (0-5)"
    
    switch ($choice) {
        "1" { Install-Environment }
        "2" { Start-Training }
        "3" { Start-Test }
        "4" { Start-InteractiveTest }
        "5" { Start-Game }
        "0" { 
            Clear-Host
            Write-Host ""
            Write-Host "感谢使用 Flappy Bird AI！" -ForegroundColor Cyan
            Write-Host ""
            Start-Sleep -Seconds 2
            exit 0
        }
        default { 
            Write-Host "无效的选择，请重试！" -ForegroundColor Red
            Start-Sleep -Seconds 1
        }
    }
} while ($true)

# 如果脚本意外退出，等待用户按键
Write-Host ""
Write-Host "按任意键退出..." -ForegroundColor Yellow
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
