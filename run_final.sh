#!/bin/bash
set -e

# =============================================================================
# Zindi Telco - Final Submission Pipeline
#
# Steps (each optional via flags):
#   1. Generate reasoning traces from train.csv (2400 Type A)
#   2. SFT training (Qwen3-32B, QLoRA)
#   3. GRPO training (3 rewards: boxed, think tags, accuracy)
#   4. Inference (vLLM, majority vote)
#
# Usage:
#   # Full pipeline from scratch:
#   bash run_final.sh --all
#
#   # Skip trace generation (pre-baked in repo):
#   bash run_final.sh --sft --grpo --infer
#
#   # Just SFT + push to hub:
#   bash run_final.sh --sft
#
#   # Just GRPO (after SFT is on hub):
#   bash run_final.sh --grpo
#
#   # Just inference from HF hub model:
#   bash run_final.sh --infer
#
#   # Inference from local GRPO output:
#   bash run_final.sh --infer --grpo-model ./outputs/qwen3-32b-grpo-final
#
#   # Inference on a different test CSV:
#   bash run_final.sh --infer --test-csv path/to/phase_3_test.csv
#
# Environment:
#   HF_TOKEN     - HuggingFace token (required for push-to-hub and private model access)
#   NUM_GPUS     - Number of GPUs (default: auto-detect)
# =============================================================================

# ---- Defaults ----
DO_TRACES=false
DO_SFT=false
DO_GRPO=false
DO_INFER=false

HF_USER="Phaedrus33"
SFT_REPO="${HF_USER}/SFT_final_submission"
GRPO_REPO="${HF_USER}/GRPO_final_submission"

TRACES_PATH="outputs/traces_final/traces_final.json"
SFT_OUTPUT="outputs/qwen3-32b-sft-final"
GRPO_OUTPUT="outputs/qwen3-32b-grpo-final"
INFER_OUTPUT="outputs/inference"

# Model for inference: default is HF hub GRPO model
GRPO_MODEL="${GRPO_REPO}"

# SFT model for GRPO: default is HF hub SFT model
SFT_MODEL="${SFT_REPO}"

BASE_MODEL="unsloth/Qwen3-32B-bnb-4bit"
TEST_CSV="the-ai-telco-troubleshooting-challenge20251127-8634-8qzscv/phase_2_test.csv"

NUM_GPUS="${NUM_GPUS:-1}"

# ---- Parse args ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --all)
            DO_TRACES=true; DO_SFT=true; DO_GRPO=true; DO_INFER=true ;;
        --traces)
            DO_TRACES=true ;;
        --sft)
            DO_SFT=true ;;
        --grpo)
            DO_GRPO=true ;;
        --infer)
            DO_INFER=true ;;
        --grpo-model)
            GRPO_MODEL="$2"; shift ;;
        --sft-model)
            SFT_MODEL="$2"; shift ;;
        --test-csv)
            TEST_CSV="$2"; shift ;;
        --num-gpus)
            NUM_GPUS="$2"; shift ;;
        --help|-h)
            head -35 "$0" | tail -30
            exit 0 ;;
        *)
            echo "Unknown flag: $1"; exit 1 ;;
    esac
    shift
done

if ! $DO_TRACES && ! $DO_SFT && ! $DO_GRPO && ! $DO_INFER; then
    echo "No steps selected. Use --all, or one or more of: --traces --sft --grpo --infer"
    echo "Run with --help for full usage."
    exit 1
fi

echo "============================================================"
echo "FINAL SUBMISSION PIPELINE"
echo "============================================================"
echo "Steps:  traces=$DO_TRACES  sft=$DO_SFT  grpo=$DO_GRPO  infer=$DO_INFER"
echo "SFT repo:   $SFT_REPO"
echo "GRPO repo:  $GRPO_REPO"
echo "GRPO model: $GRPO_MODEL"
echo "GPUs:       $NUM_GPUS"
echo "============================================================"

# =============================================================================
# Step 1: Generate reasoning traces from train.csv
# =============================================================================
if $DO_TRACES; then
    echo ""
    echo "============================================================"
    echo "STEP 1: Generate reasoning traces from train.csv"
    echo "============================================================"

    python generate_traces_final.py \
        --output "$TRACES_PATH"

    echo "Traces saved: $TRACES_PATH"
fi

# =============================================================================
# Step 2: SFT Training
# =============================================================================
if $DO_SFT; then
    echo ""
    echo "============================================================"
    echo "STEP 2: SFT Training"
    echo "============================================================"

    if [ ! -f "$TRACES_PATH" ]; then
        echo "ERROR: Traces not found at $TRACES_PATH"
        echo "Run with --traces first, or ensure pre-baked traces exist."
        exit 1
    fi

    python train_sft_final.py \
        --train-checkpoint "$TRACES_PATH" \
        --output-dir "$SFT_OUTPUT" \
        --model "$BASE_MODEL" \
        --epochs 10 \
        --batch-size 1 \
        --gradient-accumulation 8 \
        --learning-rate 5e-4 \
        --lora-r 32 \
        --lora-alpha 64 \
        --max-seq-length 8192 \
        --eval-steps 200 \
        --push-to-hub \
        --hf-repo "$SFT_REPO"

    # Use local SFT output for GRPO if running both
    SFT_MODEL="$SFT_OUTPUT"

    echo "SFT complete: $SFT_OUTPUT"
    echo "Pushed to: https://huggingface.co/$SFT_REPO"
fi

# =============================================================================
# Step 3: GRPO Training
# =============================================================================
if $DO_GRPO; then
    echo ""
    echo "============================================================"
    echo "STEP 3: GRPO Training"
    echo "============================================================"

    if [ ! -f "$TRACES_PATH" ]; then
        echo "ERROR: Traces not found at $TRACES_PATH"
        exit 1
    fi

    python train_grpo_final.py \
        --sft-model "$SFT_MODEL" \
        --base-model "$BASE_MODEL" \
        --train-checkpoint "$TRACES_PATH" \
        --output-dir "$GRPO_OUTPUT" \
        --max-steps 200 \
        --num-generations 6 \
        --learning-rate 5e-6 \
        --temperature 1.0 \
        --gradient-accumulation-steps 4 \
        --gpu-memory-utilization 0.95 \
        --no-augment \
        --push-to-hub \
        --merge-16bit \
        --hf-repo "$GRPO_REPO"

    # Use HF repo for inference (merged 16-bit model)
    GRPO_MODEL="$GRPO_REPO"

    echo "GRPO complete: $GRPO_OUTPUT"
    echo "Pushed to: https://huggingface.co/$GRPO_REPO"
fi

# =============================================================================
# Step 4: Inference
# =============================================================================
if $DO_INFER; then
    echo ""
    echo "============================================================"
    echo "STEP 4: Inference (model: $GRPO_MODEL)"
    echo "============================================================"

    if [ ! -f "$TEST_CSV" ]; then
        echo "ERROR: Test CSV not found at $TEST_CSV"
        exit 1
    fi

    python inference_grpo_final.py \
        --model "$GRPO_MODEL" \
        --test-csv "$TEST_CSV" \
        --output-dir "$INFER_OUTPUT" \
        --num-gpus "$NUM_GPUS" \
        --num-generations 4 \
        --temperature 0.6 \
        --max-tokens 4096 \
        --batch-size 32 \
        ${HF_TOKEN:+--hf-token "$HF_TOKEN"}

    echo ""
    echo "============================================================"
    echo "DONE - Final submissions in: $INFER_OUTPUT/"
    echo "============================================================"
    ls -lh "$INFER_OUTPUT"/submission_*.csv 2>/dev/null || echo "(no submission CSVs found)"
fi
