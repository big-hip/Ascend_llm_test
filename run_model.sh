#!/bin/bash
# =========================================================
# 脚本名称: run_model.sh
# 功能描述: 自动配置环境并运行模型文件夹内的 capture_pd_graphs.py
# 使用方法: ./run_model.sh <模型文件夹名称>
# =========================================================

# 开启严格模式
set -e
set -u

# =========================================================
# 1. 配置区域
# =========================================================

# 获取脚本所在的目录 (即 Ascend_llm_test)
CURRENT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# [重要] 找到项目根目录
# [修改点] 动态设置项目根目录
# 逻辑：脚本在 Ascend_llm_test 下，依赖库在上一级 Project_finally 下
# 所以 PROJECT_ROOT 就是 CURRENT_DIR 的父目录
PROJECT_ROOT="$(dirname "$CURRENT_DIR")"

# 目标 Python 脚本的文件名 (它位于模型文件夹内)
TARGET_SCRIPT_NAME="capture_pd_graphs.py"

# --- 本地依赖库目录名称定义 ---
DIR_TRANSFORM="transform_tool_v4"
DIR_DIST_INF="Ascend_IR"
DIR_PERF_EVAL="Performance_Eval"
DIR_COSTMODEL="Costmodel"

# =========================================================
# 2. 样式定义
# =========================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info()  { echo -e "${BLUE}[INFO]${NC} $1"; }
log_succ()  { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_err()   { echo -e "${RED}[ERROR]${NC} $1"; }

# =========================================================
# 3. 参数校验与路径解析
# =========================================================

# 检查参数
if [ -z "${1:-}" ]; then
    echo -e "${YELLOW}Usage:${NC} $0 <model_folder_name>"
    echo -e "${YELLOW}Example:${NC} $0 decapoda-research-llama-7B-hf"
    exit 1
fi

MODEL_NAME="$1"
CURRENT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# 模型文件夹的绝对路径
MODEL_DIR="$CURRENT_DIR/$MODEL_NAME"
# 目标 Python 脚本的绝对路径
TARGET_SCRIPT_PATH="$MODEL_DIR/$TARGET_SCRIPT_NAME"

# 打印任务信息
echo "========================================================="
echo -e "任务启动: 模型 [${YELLOW}$MODEL_NAME${NC}]"
echo -e "脚本位置: $CURRENT_DIR"
echo -e "项目根目录: $PROJECT_ROOT"
echo -e "Python脚本路径: $TARGET_SCRIPT_PATH"
echo "========================================================="

# 校验模型路径
if [ ! -d "$MODEL_DIR" ]; then
    log_err "Model directory not found: '$MODEL_DIR'"
    exit 1
fi

# 校验 Python 脚本是否存在于模型文件夹内
if [ ! -f "$TARGET_SCRIPT_PATH" ]; then
    log_err "Target script '$TARGET_SCRIPT_NAME' not found inside '$MODEL_DIR'!"
    exit 1
fi

# =========================================================
# 4. 环境清理与 Transformers 安装
# =========================================================
log_info "[1/5] Checking Transformers environment..."

if [[ "$MODEL_NAME" == *"deepseek"* ]] || [[ "$MODEL_NAME" == *"Qwen"* ]]; then
    log_info "Detected newer model ($MODEL_NAME), installing latest transformers..."
    # 请确认此版本号是否正确，如果不确定建议改为 pip install -U transformers
    pip install transformers==4.57.3 -i https://pypi.tuna.tsinghua.edu.cn/simple
else
    log_info "Installing standard transformers..."
    pip install transformers==4.35.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
fi

log_info "[2/5] Cleaning up old local packages..."
pip uninstall -y Dist_IR Performance_Eval IR_transform Costmodel > /dev/null 2>&1 || true
log_succ "Cleanup done."

# =========================================================
# 5. 本地依赖库安装函数
# =========================================================
install_local_pkg() {
    local pkg_path=$1
    local full_path="$PROJECT_ROOT/$pkg_path"
    
    if [ -z "$pkg_path" ]; then
        log_err "Package directory variable is empty!"
        exit 1
    fi

    local pkg_name=$(basename "$full_path")
    echo -n ">> Installing $pkg_name ... "

    if [ ! -d "$full_path" ]; then
        echo ""
        log_err "Directory '$full_path' does not exist!"
        log_err "Expected path: $PROJECT_ROOT/$pkg_path"
        exit 1
    fi

    if [ ! -f "$full_path/setup.py" ]; then
        echo ""
        log_err "setup.py not found in '$full_path'!"
        exit 1
    fi

    cd "$full_path" || exit 1
    
    if python setup.py install > /dev/null 2>&1; then
        echo -e "${GREEN}Success${NC}"
    else
        echo -e "${RED}Failed${NC}"
        log_warn "Retrying with verbose output..."
        python setup.py install
        exit 1
    fi
}

# =========================================================
# 6. 执行本地库安装
# =========================================================
log_info "[3/5] Installing local dependencies..."

install_local_pkg "$DIR_TRANSFORM"
install_local_pkg "$DIR_DIST_INF"
install_local_pkg "$DIR_PERF_EVAL"
install_local_pkg "$DIR_COSTMODEL"

# =========================================================
# 7. 运行主程序
# =========================================================
# 切换到模型目录执行，确保相对路径读取文件正常
cd "$MODEL_DIR" || exit 1

# 设置环境变量 (传递给 Python 脚本)
export TARGET_MODEL_DIR="$MODEL_DIR" 
export TARGET_MODEL_NAME="$MODEL_NAME"

# 日志文件保存在脚本同级目录下
LOG_FILE="$CURRENT_DIR/log_${MODEL_NAME}.txt"

# --- [修改] 加载 envs.sh ---
# 检查 **模型目录 ($MODEL_DIR)** 下是否有 envs.sh，如果有则加载
if [ -f "envs.sh" ]; then
    log_info "Sourcing environment variables from $MODEL_DIR/envs.sh..."
    source "envs.sh"
else
    # 这是一个正常的警告，有些模型可能不需要 envs.sh
    log_warn "envs.sh not found in model directory, skipping source."
fi
# -------------------------

log_info "[4/5] Executing: $TARGET_SCRIPT_NAME inside $MODEL_DIR"
log_info "Logs will be saved to: $LOG_FILE"

echo "---------------------------------------------------------"
# 运行 Python 脚本
# 注意：此时我们已经在 MODEL_DIR 里面了，所以直接运行当前目录下的脚本
python "$TARGET_SCRIPT_NAME" --model_dir "$MODEL_DIR" 2>&1 | tee "$LOG_FILE"
PY_EXIT_CODE=${PIPESTATUS[0]}
echo "---------------------------------------------------------"

# 切回脚本目录 (可选，好习惯)
cd "$CURRENT_DIR"

if [ $PY_EXIT_CODE -eq 0 ]; then
    log_succ "[5/5] Task finished successfully."
else
    log_err "[5/5] Task failed with exit code $PY_EXIT_CODE."
    exit $PY_EXIT_CODE
fi