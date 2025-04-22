cd VLMEvalKit

export LOCAL_LLM="Qwen/Qwen2.5-VL-3B-Instruct"

python run.py \
    --data MMMU_DEV_VAL MEGABench_core_16frame DocVQA_[VAL/TEST] InfoVQA_[VAL/TEST] VideoMMLU_CAP_16frame VideoMMLU_QA_16frame Video-MME_8frame MathVista_MINI\
    --model lmdeploy 
    # --verbose


# MMMU_DEV_VAL 
# MEGABench_open_64frame MEGABench_open_16frame MEGABench_core_64frame MEGABench_core_16frame
# DocVQA_[VAL/TEST]
# InfoVQA_[VAL/TEST]
# VideoMMLU_CAP_16frame VideoMMLU_CAP_64frame VideoMMLU_QA_16frame VideoMMLU_QA_64frame
# Video-MME_8frame Video-MME_64frame Video-MME_8frame_subs Video-MME_1fps Video-MME_0.5fps Video-MME_0.5fps_subs
# MathVista_MINI