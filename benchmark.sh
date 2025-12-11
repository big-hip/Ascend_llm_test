#!/bin/bash

# ================= 配置区 =================
TARGET_ENV="torch_npu"

# 【重要修改】使用相对路径。
# 既然脚本都在 Ascend_llm_test 下，直接用当前目录 ./ 即可
TARGET_SCRIPT="/data/new_root/Project_finally/Performance_Eval/Performance_Eval/Cost_model/benchmark/operator_benchmark.py" 
# =========================================

# 1. 初始化 Conda
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"

# 2. 激活环境
if ! conda activate "$TARGET_ENV" 2>/dev/null; then
    echo "[Error] 无法激活环境: $TARGET_ENV"
    exit 1
fi

echo ">> 已加载环境: $TARGET_ENV"
echo ">> 正在执行: $TARGET_SCRIPT"

# 3. 执行 Python 脚本 (不带额外传参)
echo "------------------------------------------"
python "$TARGET_SCRIPT"
EXIT_CODE=$?
echo "------------------------------------------"

# 4. 根据结果输出日志
if [ $EXIT_CODE -eq 0 ]; then
    echo ">> Benchmark 任务执行成功。"
else
    echo ">> Benchmark 任务执行失败 (Exit Code: $EXIT_CODE)。"
fi

# 5. 返回状态码给外层脚本
exit $EXIT_CODE