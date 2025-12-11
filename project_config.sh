#!/bin/bash

# =========================================================
# 全局依赖库名称配置
# 修改此处，所有 source 这个文件的脚本都会生效
# =========================================================

# 使用 export 确保变量能被子 Shell 继承（虽然 source 不需要 export，但为了稳健性）
export DIR_TRANSFORM="transform_tool_v4"
export DIR_DIST_INF="Ascend_IR"
export DIR_PERF_EVAL="Performance_Eval"
export DIR_COSTMODEL="Costmodel"
# 你甚至可以把 transformers 的版本逻辑也放在这里
# export TRANSFORMERS_VERSION="latest" # or "4.35.2"