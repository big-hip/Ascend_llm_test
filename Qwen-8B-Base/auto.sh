#!/bin/bash

# =========================================================
# 0. 参数解析
# =========================================================
#要执行的文件
TARGET_SCRIPT="capture_pd_graphs.py"

echo ">> 准备运行的目标脚本: $TARGET_SCRIPT"

# =========================================================
# 1. 动态路径推导
# =========================================================
CURRENT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$CURRENT_DIR/../.." && pwd)

echo ">> 脚本位置: $CURRENT_DIR"
echo ">> 项目根目录: $PROJECT_ROOT"

# =========================================================
# 1.1 [新增] 配置依赖库目录名称 (在此处集中修改)
# =========================================================
DIR_TRANSFORM="transform_tool_v4"
DIR_DIST_INF="Distributed_Parallelism_Inference_2"
DIR_COSTMODEL="Costmodel"
DIR_PERF_EVAL="Performance_Eval"

# =========================================================
# 2. 基础环境清理与安装
# =========================================================
echo ">> [1/7] Installing Transformers..."

#对于Qwen 和deepseek 使用这个Transformers环境
pip install -U transformers -i https://pypi.tuna.tsinghua.edu.cn/simple
# #对于 其他模型使用这个 Transformers环境
# pip install transformers==4.35.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
echo ">> [2/7] Cleaning up old packages..."
pip uninstall -y Dist_IR
pip uninstall -y Performance_Eval
pip uninstall -y IR_transform

# =========================================================
# 3. 本地依赖库安装 (使用变量路径)
# =========================================================

install_local_pkg() {
    local pkg_path=$1
    local pkg_name=$(basename "$pkg_path")
    echo ">> Installing local package: $pkg_name ..."
    
    if [ -d "$pkg_path" ]; then
        cd "$pkg_path" || exit 1
        python setup.py install > /dev/null 2>&1
        if [ $? -eq 0 ]; then
            echo "   Success."
        else
            echo "   Failed! Re-running without silence to show error:"
            python setup.py install
            exit 1
        fi
    else
        echo "Error: Directory $pkg_path not found!"
        exit 1
    fi
}

echo ">> [3/7] Installing $DIR_TRANSFORM..."
install_local_pkg "$PROJECT_ROOT/$DIR_TRANSFORM"

echo ">> [4/7] Installing $DIR_DIST_INF..."
install_local_pkg "$PROJECT_ROOT/$DIR_DIST_INF"

echo ">> [5/7] Installing $DIR_COSTMODEL..."
install_local_pkg "$PROJECT_ROOT/$DIR_COSTMODEL"

echo ">> [6/7] Installing $DIR_PERF_EVAL..."
install_local_pkg "$PROJECT_ROOT/$DIR_PERF_EVAL"

# =========================================================
# 4. 运行主程序
# =========================================================

# 回到工作目录
cd "$CURRENT_DIR" || exit 1

# 加载环境变量
if [ -f "envs.sh" ]; then
    echo ">> Sourcing envs.sh..."
    source envs.sh
fi

# 检查目标 Python 脚本是否存在
if [ ! -f "$CURRENT_DIR/$TARGET_SCRIPT" ]; then
    echo "Error: Target script '$CURRENT_DIR/$TARGET_SCRIPT' does not exist!"
    exit 1
fi

echo ">> Starting Execution: $TARGET_SCRIPT"
echo ">> Logs will be saved to: log_${TARGET_SCRIPT%.*}.txt"

# 运行指定的 Python 脚本
python "$CURRENT_DIR/$TARGET_SCRIPT" 2>&1 | tee "log_${TARGET_SCRIPT%.*}.txt"

echo ">> Done."