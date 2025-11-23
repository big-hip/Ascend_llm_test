cd /user/tangchengxiang/transform_tool_v4 ||exit
python set_up.py install
cd /user/tangchengxiang/Ascend_llm_test/Qwen-8B-Base || exit
source envs.sh
cd /user/tangchengxiang/Ascend_IR  || exit
python set_up.py install

# cd  /user/tangchengxiang/Performance_Eval
# python setup.py install
cd /user/tangchengxiang/Ascend_llm_test/Qwen-8B-Base || exit
python /user/tangchengxiang/Ascend_llm_test/Qwen-8B-Base/train_aten_IR_2.py  2>&1 | tee full_debug.log