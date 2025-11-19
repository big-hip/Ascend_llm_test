# export dist_config='{
#   "device_nums": 42,
#   "parallel_degree": 
#   {
#     "D": 1,
#     "P": 1,
#     "T": 2,
#     "S": 1,
#     "E": 1
#   },
#   "parallel_config": 
#   {
#     "DP_config": [1.0],
#     "PP_config": [1.0],
#     "Zero_config": {"stage":0,"ratio":[1.0]},
    
#     "SP_config": 
#     {
#       "partition": [1.0]
#     },

#     "EP_config":[{"ep_mode":"tp_ep","expert_map":{"N-N-0":[0,1],"N-N-1":[2,3]},"combine_after":false}],

#     "TP_config": 
#     {
#       "sp_then_tp":{
#           "parallel_degree":2,
#           "sp_ratio":[0.1,0.9]},
#       "embedding":{
#           "parallel_degree": 2,
#           "ratio": [0.4,0.6]},
#       "encoder": { 
#         "0": {
#           "feed_forward": {
#             "parallel_degree": 2,
#             "ratio": [0.4,0.6]},
#           "attn": {
#             "num_head":8,
#             "hidden":512,
#             "parallel_degree": 2,
#             "ratio": [0.25,0.75]}}},
#       "decoder":{
#         "0":{
#           "feed_forward":{
#             "parallel_degree": 2,
#             "ratio": [0.3,0.7]},
#           "self_attn":{
#             "num_head":8,
#             "hidden":512,
#             "parallel_degree": 2,
#             "ratio": [0.75,0.25]},
#           "cross_attn":{
#             "num_head":8,
#             "hidden":512,
#             "parallel_degree": 2,
#             "ratio": [0.75,0.25]}}}}
#   }
# }'

    
# # 打印环境变量以验证
# echo "dist_config 已设置为："
# echo $dist_config


export dist_config='{
  "device_nums": 42,
  "parallel_degree": 
  {
    "D": 4,
    "P": 5,
    "T": 2,
    "S": 1,
    "E": 1
  },
  "parallel_config": 
  {
    "DP_config": [0.1, 0.2, 0.3, 0.4],
    "PP_config": [0.1,0.1,0.4,0.2,0.2],
    "Zero_config": {"stage":0,"ratio":[1.0]},
    
    "SP_config": 
    {
      "partition": [1.0]
    },

    "EP_config":[{"ep_mode":"tp_ep","expert_map":{"N-N-0":[0,1],"N-N-1":[2,3]},"combine_after":false}],

    "TP_config": 
    {
      "sp_then_tp":{
          "parallel_degree":2,
          "sp_ratio":[0.1,0.9]},
      "embedding":{
          "parallel_degree": 2,
          "ratio": [0.4,0.6]},
      "encoder": { 
        "0": {
          "feed_forward": {
            "parallel_degree": 2,
            "ratio": [0.4,0.6]},
          "attn": {
            "num_head":8,
            "hidden":512,
            "parallel_degree": 2,
            "ratio": [0.25,0.75]}}},
      "decoder":{
        "0":{
          "feed_forward":{
            "parallel_degree": 2,
            "ratio": [0.3,0.7]},
          "self_attn":{
            "num_head":8,
            "hidden":512,
            "parallel_degree": 2,
            "ratio": [0.75,0.25]},
          "cross_attn":{
            "num_head":8,
            "hidden":512,
            "parallel_degree": 2,
            "ratio": [0.75,0.25]}}}}
  }
}'

    
# 打印环境变量以验证
echo "dist_config 已设置为："
echo $dist_config

