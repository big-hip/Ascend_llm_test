export dist_config='{
  "prefill":{
  "device_nums": 8,
  "parallel_degree": 
  {
    "D": 2,
    "P": 2,
    "T": 2,
    "S": 1,
    "E": 1
  },
  "parallel_config": 
  {
    "DP_config": [0.5,0.5],
    "PP_config": [0.5,0.5],
    "Zero_config": {"stage":0,"ratio":[1.0]},
    "SP_config": 
    {
      "partition": [1.0]
    },

    "EP_config":[{"ep_mode":"tp_ep","expert_map":{"N-N-0":[0,1],"N-N-1":[2,3]},"combine_after":false}],

    "TP_config": 
    {
      "model": "gpt2",
      "sp_then_tp":{
          "parallel_degree":2,
          "sp_ratio":[0.1,0.9]},
      "embedding":{
          "parallel_degree": 2,
          "ratio": [0.4,0.6]},
      "decoder":{
        "0":{
          "mlp":{
            "parallel_degree": 2,
            "ratio": [0.5,0.5],
            "hidden":128},
          "attn":{
            "num_head":4,
            "hidden":128,
            "parallel_degree": 2,
            "ratio": [0.75,0.25]}}}}
  }},
  "decode":{
  "device_nums": 8,
  "parallel_degree": 
  {
    "D": 2,
    "P": 2,
    "T": 2,
    "S": 1,
    "E": 1
  },
  "parallel_config": 
  {
    "DP_config": [0.5,0.5],
    "PP_config": [0.5,0.5],
    "Zero_config": {"stage":0,"ratio":[1.0]},
    "SP_config": 
    {
      "partition": [1.0]
    },

    "EP_config":[{"ep_mode":"tp_ep","expert_map":{"N-N-0":[0,1],"N-N-1":[2,3]},"combine_after":false}],

    "TP_config": 
    {
      "model": "gpt2",
      "sp_then_tp":{
          "parallel_degree":2,
          "sp_ratio":[0.1,0.9]},
      "embedding":{
          "parallel_degree": 2,
          "ratio": [0.4,0.6]},
      "decoder":{
        "0":{
          "mlp":{
            "parallel_degree": 2,
            "ratio": [0.5,0.5],
            "hidden":768},
          "attn":{
            "num_head":4,
            "hidden":768,
            "parallel_degree": 2,
            "ratio": [0.75,0.25]}}}}
  }}

}'

    
# 打印环境变量以验证
echo "dist_config 已设置为："
echo $dist_config
