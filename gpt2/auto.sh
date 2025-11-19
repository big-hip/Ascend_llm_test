cd /user/tangchengxiang/transform_tool_v4 ||exit
python set_up.py install
cd /user/tangchengxiang/Ascend_llm_test/gpt2 || exit
source envs.sh
cd /user/tangchengxiang/Ascend_IR  || exit
python set_up.py install

cd  /user/tangchengxiang/Performance_Eval
python setup.py install
cd /user/tangchengxiang/Ascend_llm_test/gpt2 || exit
python /user/tangchengxiang/Ascend_llm_test/gpt2/train_aten_IR.py