cd /user/tangchengxiang/transform_tool_v4 ||exit
python set_up.py install
cd /user/tangchengxiang/Ascend_llm_test/Transformer || exit
source envs.sh
cd /user/tangchengxiang/Ascend_IR  || exit
python set_up.py install
cd /user/tangchengxiang/Ascend_llm_test/Transformer || exit
python /user/tangchengxiang/Ascend_llm_test/Transformer/MOEtransformer.py