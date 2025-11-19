cd /user/tangchengxiang/transform_tool_v4 ||exit
python set_up.py install
cd /user/tangchengxiang/Ascend_llm_test/decapoda-research-llama-7B-hf || exit
source envs.sh
cd /user/tangchengxiang/Ascend_IR  || exit
python set_up.py install
cd /user/tangchengxiang/Ascend_llm_test/decapoda-research-llama-7B-hf || exit
python /user/tangchengxiang/Ascend_llm_test/decapoda-research-llama-7B-hf/train_aten_IR.py