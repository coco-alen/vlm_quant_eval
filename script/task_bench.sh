cd VLMEvalKit

export OPENAI_API_KEY='sk-miniphant'
export OPENAI_API_BASE='http://0.0.0.0:23333/v1/chat/completions'
export LOCAL_LLM='Qwen/Qwen2.5-VL-3B-Instructs'

python run.py \
    --data DocVQA_VAL InfoVQA_VAL MMMU_DEV_VAL MathVista_MINI Video-MME_8frame Video-MME_64frame MEGABench_open_16frame  \
    --model lmdeploy \
    --api-nproc 4 \
    --work-dir './outputs/w8a8_fp8'
    # --verbose

# MMMU_DEV_VAL 
# MEGABench_open_64frame MEGABench_open_16frame MEGABench_core_64frame MEGABench_core_16frame
# DocVQA_[VAL/TEST]
# InfoVQA_[VAL/TEST]
# VideoMMLU_CAP_16frame VideoMMLU_CAP_64frame VideoMMLU_QA_16frame VideoMMLU_QA_64frame
# Video-MME_8frame Video-MME_64frame Video-MME_8frame_subs Video-MME_1fps Video-MME_0.5fps Video-MME_0.5fps_subs
# MathVista_MINI

# --data DocVQA_VAL InfoVQA_VAL MMMU_DEV_VAL MathVista_MINI Video-MME_8frame MEGABench_open_16frame Video-MME_64frame \

# VideoMMLU_CAP_16frame VideoMMLU_QA_16frame Video-MME_8frame

# MEGABench_open_16frame Video-MME_64frame