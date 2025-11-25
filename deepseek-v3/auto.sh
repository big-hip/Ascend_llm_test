cd /user/tangchengxiang/transform_tool_v4 ||exit
python set_up.py install
cd /user/tangchengxiang/Ascend_llm_test/deepseek-v3 || exit
source envs.sh
cd /user/tangchengxiang/Ascend_IR  || exit
python set_up.py install

cd  /user/tangchengxiang/Performance_Eval
python setup.py install
cd /user/tangchengxiang/Ascend_llm_test/deepseek-v3 || exit
python /user/tangchengxiang/Ascend_llm_test/deepseek-v3/train_aten_IR.py 2>&1 |tee log.txt