# export dist_config='{
#   "prefill":{
#       "device_nums": 8,
#       "parallel_degree": 
#       {
#         "D": 2,
#         "P": 1,
#         "T": 4,
#         "S": 1,
#         "E": 1
#       },
#       "parallel_config": 
#       {
#         "DP_config": [0.5,0.5],
#         "PP_config": [1],
#         "Zero_config": {"stage":0,"ratio":[1.0]},
        
#         "SP_config": 
#         {
#           "partition": [1.0]
#         },

#         "EP_config":[{"ep_mode":"none_ep","expert_map":{"N-N-0":[0,1],"N-N-1":[2,3]},"combine_after":false}],

#         "TP_config": 
#         {
#           "embedding":{
#               "parallel_degree": 4,
#               "ratio": [0.25,0.25,0.25,0.25]},
#           "decoder":{
#             "all":{
#               "mlp":{
#                 "parallel_degree": 4,
#                 "ratio": [0.25,0.25,0.25,0.25],
#                 "hidden":4096},
#               "attn":{
#                 "num_head":32,
#                 "hidden":4096,
#                 "parallel_degree": 4,
#                 "ratio": [0.25,0.25,0.25,0.25]}}}}
#       }
#       },
#   "decode":{
#       "device_nums": 8,
#       "parallel_degree": 
#       {
#         "D": 2,
#         "P": 1,
#         "T": 4,
#         "S": 1,
#         "E": 1
#       },
#       "parallel_config": 
#       {
#         "DP_config": [0.5,0.5],
#         "PP_config": [1],
#         "Zero_config": {"stage":0,"ratio":[1.0]},
        
#         "SP_config": 
#         {
#           "partition": [1.0]
#         },

#         "EP_config":[{"ep_mode":"none_ep","expert_map":{"N-N-0":[0,1],"N-N-1":[2,3]},"combine_after":false}],

#         "TP_config": 
#         {
    
#           "embedding":{
#               "parallel_degree": 4,
#               "ratio": [0.25,0.25,0.25,0.25]},
#           "decoder":{
#             "all":{
#               "mlp":{
#                 "parallel_degree": 4,
#                 "ratio": [0.25,0.25,0.25,0.25],
#                 "hidden":4096},
#               "attn":{
#                 "num_head":32,
#                 "hidden":4096,
#                 "parallel_degree": 4,
#                 "ratio": [0.25,0.25,0.25,0.25]}}}}
#       }
#       }
# }'

    
# # 打印环境变量以验证
# echo "dist_config 已设置为："
# echo $dist_config


export dist_config='{
  "prefill":{
      "device_nums": 8,
      "parallel_degree": 
      {
        "D": 4,
        "P": 1,
        "T": 2,
        "S": 1,
        "E": 1
      },
      "parallel_config": 
      {
        "DP_config": [0.25,0.25,0.25,0.25],
        "PP_config": [1],
        "Zero_config": {"stage":0,"ratio":[1.0]},
        
        "SP_config": 
        {
          "partition": [1.0]
        },

        "EP_config":[{"ep_mode":"none_ep","expert_map":{"N-N-0":[0,1],"N-N-1":[2,3]},"combine_after":false}],

        "TP_config": 
        {
          "embedding":{
              "parallel_degree": 2,
              "ratio": [0.5,0.5]},
          "decoder":{
            "all":{
              "mlp":{
                "parallel_degree": 2,
                "ratio": [0.5,0,5],
                "hidden":4096},
              "attn":{
                "num_head":32,
                "hidden":4096,
                "parallel_degree": 2,
                "ratio": [0.5,0.5]}}}}
      }
      },
  "decode":{
      "device_nums": 8,
      "parallel_degree": 
      {
        "D": 4,
        "P": 1,
        "T": 2,
        "S": 1,
        "E": 1
      },
      "parallel_config": 
      {
        "DP_config": [0.25,0.25,0.25,0.25],
        "PP_config": [1],
        "Zero_config": {"stage":0,"ratio":[1.0]},
        
        "SP_config": 
        {
          "partition": [1.0]
        },

        "EP_config":[{"ep_mode":"none_ep","expert_map":{"N-N-0":[0,1],"N-N-1":[2,3]},"combine_after":false}],

        "TP_config": 
        {
    
          "embedding":{
              "parallel_degree": 2,
              "ratio": [0.5,0.5]},
          "decoder":{
            "all":{
              "mlp":{
                "parallel_degree": 2,
                "ratio": [0.5,0.5],
                "hidden":4096},
              "attn":{
                "num_head":32,
                "hidden":4096,
                "parallel_degree": 2,
                "ratio": [0.5,0.5]}}}}
      }
      }
}'

    
# 打印环境变量以验证
echo "dist_config 已设置为："
echo $dist_config


# export dist_config='{
#   "prefill":{
#       "device_nums": 8,
#       "parallel_degree": 
#       {
#         "D": 8,
#         "P": 1,
#         "T": 1,
#         "S": 1,
#         "E": 1
#       },
#       "parallel_config": 
#       {
#         "DP_config": [0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125],
#         "PP_config": [1],
#         "Zero_config": {"stage":0,"ratio":[1.0]},
        
#         "SP_config": 
#         {
#           "partition": [1.0]
#         },

#         "EP_config":[{"ep_mode":"none_ep","expert_map":{"N-N-0":[0,1],"N-N-1":[2,3]},"combine_after":false}],

#         "TP_config": 
#         {
#           "embedding":{
#               "parallel_degree": 1,
#               "ratio": [1.0]},
#           "decoder":{
#             "all":{
#               "mlp":{
#                 "parallel_degree": 1,
#                 "ratio": [1.0],
#                 "hidden":4096},
#               "attn":{
#                 "num_head":32,
#                 "hidden":4096,
#                 "parallel_degree": 1,
#                 "ratio": [1.0]}}}}
#       }
#       },
#   "decode":{
#       "device_nums": 8,
#       "parallel_degree": 
#       {
#         "D": 8,
#         "P": 1,
#         "T": 1,
#         "S": 1,
#         "E": 1
#       },
#       "parallel_config": 
#       {
#         "DP_config": [0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125],
#         "PP_config": [1],
#         "Zero_config": {"stage":0,"ratio":[1.0]},
        
#         "SP_config": 
#         {
#           "partition": [1.0]
#         },

#         "EP_config":[{"ep_mode":"none_ep","expert_map":{"N-N-0":[0,1],"N-N-1":[2,3]},"combine_after":false}],

#         "TP_config": 
#         {
    
#           "embedding":{
#               "parallel_degree": 1,
#               "ratio": [1.0]},
#           "decoder":{
#             "all":{
#               "mlp":{
#                 "parallel_degree": 1,
#                 "ratio": [1.0],
#                 "hidden":4096},
#               "attn":{
#                 "num_head":32,
#                 "hidden":4096,
#                 "parallel_degree": 1,
#                 "ratio": [1.0]}}}}
#       }
#       }
# }'

    
# # 打印环境变量以验证
# echo "dist_config 已设置为："
# echo $dist_config


# export dist_config='{
#   "prefill":{
#       "device_nums": 8,
#       "parallel_degree": 
#       {
#         "D": 1,
#         "P": 1,
#         "T": 1,
#         "S": 1, 
#         "E": 1
#       },
#       "parallel_config": 
#       {
#         "DP_config": [1.0],
#         "PP_config": [1],
#         "Zero_config": {"stage":0,"ratio":[1.0]},
        
#         "SP_config": 
#         {
#           "partition": [1.0]
#         },

#         "EP_config":[{"ep_mode":"none_ep","expert_map":{"N-N-0":[0,1],"N-N-1":[2,3]},"combine_after":false}],

#         "TP_config": 
#         {
#           "embedding":{
#               "parallel_degree": 8,
#               "ratio": [0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125]},
#           "decoder":{
#             "all":{
#               "mlp":{
#                 "parallel_degree": 8,
#                 "ratio": [0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125],
#                 "hidden":4096},
#               "attn":{
#                 "num_head":32,
#                 "hidden":4096,
#                 "parallel_degree": 8,
#                 "ratio": [0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125]}}}}
#       }
#       },
#   "decode":{
#       "device_nums": 8,
#       "parallel_degree": 
#       {
#         "D": 1,
#         "P": 1,
#         "T": 1,
#         "S": 1,
#         "E": 1
#       },
#       "parallel_config": 
#       {
#         "DP_config": [1.0],
#         "PP_config": [1],
#         "Zero_config": {"stage":0,"ratio":[1.0]},
        
#         "SP_config": 
#         {
#           "partition": [1.0]
#         },

#         "EP_config":[{"ep_mode":"none_ep","expert_map":{"N-N-0":[0,1],"N-N-1":[2,3]},"combine_after":false}],

#         "TP_config": 
#         {
    
#           "embedding":{
#               "parallel_degree": 8,
#               "ratio": [0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125]},
#           "decoder":{
#             "all":{
#               "mlp":{
#                 "parallel_degree": 8,
#                 "ratio": [0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125],
#                 "hidden":4096},
#               "attn":{
#                 "num_head":32,
#                 "hidden":4096,
#                 "parallel_degree": 8,
#                 "ratio": [0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125]}}}}
#       }
#       }
# }'

    
# # 打印环境变量以验证
# echo "dist_config 已设置为："
# echo $dist_config


# export dist_config='{
#   "prefill":{
#       "device_nums": 8,
#       "parallel_degree": 
#       {
#         "D": 2,
#         "P": 1,
#         "T": 2,
#         "S": 1,
#         "E": 1
#       },
#       "parallel_config": 
#       {
#         "DP_config": [0.5,0.5],
#         "PP_config": [1.0],
#         "Zero_config": {"stage":0,"ratio":[1.0]},
        
#         "SP_config": 
#         {
#           "partition": [1.0],
#           "hidden": 4096,      
#           "num_head": 32,
#           "model_name":"Qwen"       
#         },

#         "EP_config":[{"ep_mode":"none_ep","expert_map":{"N-N-0":[0,1],"N-N-1":[2,3]},"combine_after":false}],

#         "TP_config": 
#         {
#           "embedding":{
#               "parallel_degree": 2,
#               "ratio": [0.5,0.5]},
#           "decoder":{
#             "all":{
#               "mlp":{
#                 "parallel_degree": 2,
#                 "ratio": [0.5,0.5],
#                 "hidden":4096},
#               "attn":{
#                 "num_head":32,
#                 "hidden":4096,
#                 "parallel_degree": 2,
#                 "ratio": [0.5,0.5]}}}}
#       }
#       },
#   "decode":{
#       "device_nums": 4,
#       "parallel_degree": 
#       {
#         "D": 2,
#         "P": 1,
#         "T": 2,
#         "S": 1,
#         "E": 1
#       },
#       "parallel_config": 
#       {
#         "DP_config": [0.5,0.5],
#         "PP_config": [1.0],
#         "Zero_config": {"stage":0,"ratio":[1.0]},
        
#         "SP_config": 
#         {
#           "partition": [1.0],
#           "hidden": 4096,      
#           "num_head": 32,
#           "inference_stage":"decode",
#           "model_name":"Qwen"       
#         },

#         "EP_config":[{"ep_mode":"none_ep","expert_map":{"N-N-0":[0,1],"N-N-1":[2,3]},"combine_after":false}],

#         "TP_config": 
#         {
    
#           "embedding":{
#               "parallel_degree": 2,
#               "ratio": [0.5,0.5]},
#           "decoder":{
#             "all":{
#               "mlp":{
#                 "parallel_degree": 2,
#                 "ratio": [0.5,0.5],
#                 "hidden":4096},
#               "attn":{
#                 "num_head":32,
#                 "hidden":4096,
#                 "parallel_degree": 2,
#                 "ratio": [0.5,0.5]}}}}
#       }
#       }
# }'

    
# # 打印环境变量以验证
# echo "dist_config 已设置为："
# echo $dist_config


# export dist_config='{
#   "prefill":{
#       "device_nums": 8,
#       "parallel_degree": 
#       {
#         "D": 1,
#         "P": 1,
#         "T": 4,
#         "S": 1,
#         "E": 1
#       },
#       "parallel_config": 
#       {
#         "DP_config": [1.0],
#         "PP_config": [1],
#         "Zero_config": {"stage":0,"ratio":[1.0]},
        
#         "SP_config": 
#         {
#           "partition": [1.0],
#           "hidden": 4096,      
#           "num_head": 32,
#           "model_name":"Qwen"       
#         },

#         "EP_config":[{"ep_mode":"none_ep","expert_map":{"N-N-0":[0,1],"N-N-1":[2,3]},"combine_after":false}],

#         "TP_config": 
#         {
#           "embedding":{
#               "parallel_degree": 4,
#               "ratio": [0.25,0.25,0.25,0.25]},
#           "decoder":{
#             "all":{
#               "mlp":{
#                 "parallel_degree": 4,
#                 "ratio": [0.25,0.25,0.25,0.25],
#                 "hidden":4096},
#               "attn":{
#                 "num_head":32,
#                 "hidden":4096,
#                 "parallel_degree": 4,
#                 "ratio": [0.25,0.25,0.25,0.25]}}}}
#       }
#       },
#   "decode":{
#       "device_nums": 8,
#       "parallel_degree": 
#       {
#         "D": 1,
#         "P": 1,
#         "T": 4,
#         "S": 1,
#         "E": 1
#       },
#       "parallel_config": 
#       {
#         "DP_config": [1.0],
#         "PP_config": [1],
#         "Zero_config": {"stage":0,"ratio":[1.0]},
        
#         "SP_config": 
#         {
#           "partition": [1.0],
#           "hidden": 4096,      
#           "num_head": 32,
#           "inference_stage":"decode",
#           "model_name":"Qwen"       
#         },

#         "EP_config":[{"ep_mode":"none_ep","expert_map":{"N-N-0":[0,1],"N-N-1":[2,3]},"combine_after":false}],

#         "TP_config": 
#         {
    
#           "embedding":{
#               "parallel_degree": 4,
#               "ratio": [0.25,0.25,0.25,0.25]},
#           "decoder":{
#             "all":{
#               "mlp":{
#                 "parallel_degree": 4,
#                 "ratio": [0.25,0.25,0.25,0.25],
#                 "hidden":4096},
#               "attn":{
#                 "num_head":32,
#                 "hidden":4096,
#                 "parallel_degree": 4,
#                 "ratio": [0.25,0.25,0.25,0.25]}}}}
#       }
#       }
# }'

    
# # 打印环境变量以验证
# echo "dist_config 已设置为："
# echo $dist_config