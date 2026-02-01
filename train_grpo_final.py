"""
Train Qwen3-32B with GRPO on reasoning traces.

Exactly 3 reward functions:
1. boxed_reward: response contains \\boxed{...}
2. think_tags_reward: response contains <think>...</think>
3. accuracy_reward: extracted answer matches ground truth

Uses Unsloth for efficient training with vLLM fast inference.
Loads reasoning traces (2400 Type A from train.csv) with question-type-aware prompts.

Usage:
    # Basic (SFT LoRA from HF)
    python train_grpo_final.py --sft-model USERNAME/sft-lora

    # With more steps
    python train_grpo_final.py --sft-model USERNAME/sft-lora --max-steps 200

    # Push merged model to HF after training
    python train_grpo_final.py \\
        --sft-model USERNAME/sft-lora \\
        --push-to-hub --merge-16bit \\
        --hf-repo USERNAME/grpo-final \\
        --hf-token hf_xxx

    # Dry run (validate data)
    python train_grpo_final.py --sft-model ./path/to/sft --dry-run
"""

# Enable Unsloth's memory-efficient standby mode for vLLM
# Must be set BEFORE importing unsloth
import os
os.environ["UNSLOTH_VLLM_STANDBY"] = "1"
import json
import re
import argparse
import logging
from typing import Dict, List
from collections import Counter

from tqdm import tqdm
from datasets import Dataset

# Metric computation (same as SFT training)
from telco_utils import parse_type_a_question
from generate_traces_final import (
    compute_all_metrics, format_metrics_block,
    compute_type_b_metrics, format_type_b_metrics_block,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# PatchFastRL - Required for Unsloth + GRPOTrainer integration
# =============================================================================
from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)


# =============================================================================
# V19 SYSTEM PROMPTS
# =============================================================================

TELCO_SYSTEM_PROMPT = """You are a 5G network root cause classifier. You receive pre-computed metrics and a multiple-choice question. Walk through the decision rules below IN ORDER, show your work for each check, identify the root cause, then match it to the correct option label.

OUTPUT FORMAT (mandatory):
1. Wrap ALL reasoning inside <think>...</think> tags.
2. After </think>, output EXACTLY \\boxed{LABEL} where LABEL is the option (e.g. C3, 7, M5).
3. Do NOT write anything after \\boxed{LABEL}. No explanation, no period, no newline text.
4. Every response MUST end with \\boxed{LABEL}. Omitting it is a failure.

DECISION RULES (apply first matching rule):

TIER 1 - check in order, return first match:
1. max_speed > 40 -> "speed exceeds 40km/h"
2. max_distance_low_tp > 1.0 -> "coverage distance exceeds 1km" (overshooting)
3. handover_count >= 3 -> "frequent handovers"
4. avg_rb < 170 -> "average scheduled RBs below 160"

TIER 2 - C1 detection (if ANY sub-rule matches -> "downtilt too large"):
5a. min_rsrp < -90 AND pci_collision = no AND c4_interference < 3
5b. strong_neighbor_count < 0.5 AND serving_tilt >= 15
5c. pci_collision = yes AND strong_neighbor_count < 0.5
  OVERRIDES on C1:
  - post_ho_good_streak >= 2 -> "neighboring cell higher throughput" instead
  - pci_collision_ratio > 0.70 -> "PCI mod 30 collision" instead
  - avg_rsrp > -79 AND strong_neighbor_count > 1.0 -> "neighboring cell higher throughput" instead

TIER 3 - interference:
6. c4_interference >= 3 -> "overlapping coverage/interference"
  SKIP if: (min_neighbor_diff / c4_interference) < -0.5 AND c4_interference < 12

TIER 4 - PCI collision (pci_collision = yes):
7. If pci_collision_ratio >= 1.0 -> "PCI mod 30 collision"
   If pci_collision_ratio < 1.0:
   - serving_tilt > 10 AND rsrp_trend > 0.4 -> "downtilt too large"
   - else -> "neighboring cell higher throughput"
   If avg_off_axis > 30:
   - min_rsrp < -90 -> "downtilt too large" (with override checks from Tier 2)
   - else -> "neighboring cell higher throughput" (with override checks from Tier 5)

TIER 5 - C1 vs C3 tiebreaker:
8. serving_tilt >= 28 AND avg_sinr >= 12 -> "neighboring cell higher throughput" (SINR gate)
   serving_tilt >= 28 AND avg_sinr < 12 -> "downtilt too large"
   serving_tilt < 12 -> "neighboring cell higher throughput"
   avg_rsrp < -90 -> "downtilt too large"
   avg_rsrp > -82 -> "neighboring cell higher throughput"
   Low confidence (avg_rsrp -90 to -82) -> rescue rules:
     R1: pci_collision_ratio >= 0.9 -> "PCI mod 30 collision"
     R2: strong_neighbor_count < 0.8 -> "downtilt too large"
     R3: c4_interference >= 3.0 -> "downtilt too large"
     R4: default -> "neighboring cell higher throughput"
  OVERRIDES if "neighboring cell" (high/medium confidence):
  - pci_collision_ratio > 0.70 -> "PCI mod 30 collision"
  - rsrp_change > 5 AND rsrp_trend > 0.5 AND nb_within_5db < 1.0 -> "downtilt too large"
  - rsrp_recovery > 15 -> "downtilt too large"
  - serving_tilt > 6 AND nb_within_5db < 1.0 -> "downtilt too large"
  OVERRIDES if "downtilt" (high/medium confidence):
  - pci_collision_ratio > 0.70 -> "PCI mod 30 collision"
  - avg_rsrp > -79 AND strong_neighbor_count > 1.0 -> "neighboring cell higher throughput"

Show your reasoning inside <think> tags, checking each tier in order. Then match the identified root cause to the option that describes it and answer with EXACTLY \\boxed{LABEL}. You MUST always end your response with \\boxed{LABEL}.
Examples: \\boxed{C3}, \\boxed{7}, \\boxed{M5}"""


TYPE_B_SYSTEM_PROMPT = """You are a 5G drive test root cause analyzer. You receive pre-computed metrics and a multiple-choice question about throughput drops. Walk through the decision rules below IN ORDER, show your work for each check, identify the root cause, then match it to the correct option label.

OUTPUT FORMAT (mandatory):
1. Wrap ALL reasoning inside <think>...</think> tags.
2. After </think>, output EXACTLY \\boxed{LABEL} where LABEL is the option letter (e.g. A, D, G).
3. Do NOT write anything after \\boxed{LABEL}. No explanation, no period, no newline text.
4. Every response MUST end with \\boxed{LABEL}. Omitting it is a failure.

IMPORTANT: Options are SHUFFLED per question - identify the root cause FIRST, then find which option letter matches it.

DECISION RULES (apply first matching rule):

1. avg_cce_fail > 0.25 -> "PDCCH congestion" (I)
2. actual_handovers >= 3 -> "intra-freq threshold too low / ping-pong" (H)
3. ratio_a3_ho >= 3 AND a3_events >= 2, OR rrc_reestablish > 0 AND a3_events >= 1:
   -> Check n1_in_config:
     If n1_in_config = False -> "missing neighbor cell configuration" (E)
     If n1_in_config = True -> "intra-freq threshold too high" (G)
4. rsrp_var_norm > 0.08 AND avg_rsrp > -95 -> "overlap coverage" (A)
5. avg_rsrp < -95 -> "weak coverage" (F)

PHY HEALTH ANALYSIS (if no rule above matches):
6. If phy_healthy_during_low_tp = True AND neighbors_within_3dB = 0 AND avg_sinr > 10:
   -> "transport/server-side anomaly" (D)
   Meaning: Radio link healthy during TP drops, bottleneck above PHY layer.
7. If phy_healthy_during_low_tp = False AND low_tp_avg_mcs < 12 AND neighbors_within_3dB >= 1:
   -> "overlap coverage" (A)
   Meaning: MCS crashes with strong neighbor present = interference/pilot pollution.

CONFIGURATION CHECK (if no rule above matches):
8. If inter_freq_ho = True AND a2_thld > -100 AND n_configured_neighbors >= 6:
   -> "inter-freq HO threshold unreasonable" (B)
   Meaning: Inter-frequency handover triggered with unreasonable A2 threshold.

Show your reasoning inside <think> tags. Then match the root cause to the option that describes it and answer with EXACTLY \\boxed{LABEL}. You MUST always end your response with \\boxed{LABEL}.
Examples: \\boxed{A}, \\boxed{D}, \\boxed{G}"""


GENERIC_SYSTEM_PROMPT = """You are an expert problem solver. Analyze questions carefully and select the correct answer.

IMPORTANT - Answer Format:
- Use the EXACT option number/label from the question
- Examples: \\boxed{2}, \\boxed{B}, \\boxed{72}

You must strictly output your reasoning process within <think>...</think> tags before the final answer."""


# =============================================================================
# ANSWER NORMALIZATION
# =============================================================================

def normalize_answer(answer: str) -> str:
    """Normalize answer format for comparison. C1 -> 1, c1 -> 1, 1 -> 1."""
    if not answer:
        return ""
    answer = answer.strip()
    match = re.match(r'^[Cc](\d+)$', answer)
    if match:
        return match.group(1)
    return answer


# =============================================================================
# REWARD FUNCTIONS (exactly 3)
# =============================================================================

BOXED_PATTERN = re.compile(r'\\boxed\s*\{\s*([^}]+?)\s*\}')


def boxed_reward(prompts, completions, **kwargs):
    """
    Reward for \\boxed{} presence.

    Returns:
        +0.5 if \\boxed{...} with content is present
        -0.5 if missing
    """
    scores = []
    for completion in completions:
        response = completion[0]["content"]
        if BOXED_PATTERN.search(response):
            scores.append(0.5)
        else:
            scores.append(-0.5)
    return scores


def think_tags_reward(prompts, completions, **kwargs):
    """
    Reward for <think>...</think> tags with non-trivial content.

    Returns:
        +1.0 if both tags present AND content >= 50 chars
        -0.5 if tags present but content too short (degenerate)
        -1.0 if either tag is missing
    """
    scores = []
    for completion in completions:
        response = completion[0]["content"]
        if '<think>' in response and '</think>' in response:
            # Extract content between think tags
            start = response.index('<think>') + len('<think>')
            end = response.index('</think>')
            think_content = response[start:end].strip()
            if len(think_content) >= 200:
                scores.append(1.0)
            else:
                scores.append(-0.5)
        else:
            scores.append(-1.0)
    return scores


def accuracy_reward(prompts, completions, answer, **kwargs):
    """
    Reward for correct answer matching ground truth.

    Returns:
        +5.0 for exact match
        +3.0 for match after normalization (C1 == 1)
        -2.0 for wrong answer
        -3.0 for no answer extracted
    """
    scores = []
    for completion, true_answer in zip(completions, answer):
        response = completion[0]["content"]

        match = BOXED_PATTERN.search(response)
        if not match:
            scores.append(-3.0)
            continue

        pred = match.group(1).strip()
        true = true_answer.strip()

        if pred == true:
            scores.append(5.0)
            continue

        pred_norm = normalize_answer(pred)
        true_norm = normalize_answer(true)
        if pred_norm == true_norm:
            scores.append(3.0)
            continue

        scores.append(-2.0)
    return scores


# =============================================================================
# QUESTION TYPE DETECTION
# =============================================================================

def get_question_type(question: str, source_type: str = None) -> str:
    """Detect question type: 'type_a', 'type_b', or 'generic'.

    Uses source_type from trace data when available, falls back to heuristics.
    """
    if source_type and source_type in ('type_a', 'type_b', 'generic'):
        return source_type

    # Heuristic: Type B questions have drive test throughput drop analysis
    if 'throughput drop' in question.lower() and 'drive test' in question.lower():
        return 'type_b'

    # Heuristic: Type A questions have telco data tables
    if question.strip().startswith("Analyze the following question"):
        if '|' in question and question.count('|') >= 4:
            return 'type_a'

    # Default to generic
    return 'generic'


# =============================================================================
# DATA LOADING
# =============================================================================

def load_v19_traces(checkpoint_path: str) -> List[Dict]:
    """Load V19 reasoning traces.

    V19 traces have: question, expected_answer, reasoning_trace, question_type.
    All traces are pre-validated (success=True equivalent).
    """
    logger.info(f"Loading V19 traces from {checkpoint_path}")

    with open(checkpoint_path, 'r') as f:
        checkpoint = json.load(f)

    traces = []
    for row_id, data in checkpoint.items():
        traces.append({
            'row_id': row_id,
            'question': data['question'],
            'answer': data['expected_answer'],
            'question_type': data.get('question_type', 'type_a'),
            'source': 'v19_train',
        })

    logger.info(f"Loaded {len(traces)} V19 traces")

    type_counts = Counter(t['question_type'] for t in traces)
    for qt, count in sorted(type_counts.items()):
        logger.info(f"  {qt}: {count}")

    return traces


def load_test_augmentation(
    checkpoint_path: str,
    test_csv_path: str,
    min_agreement: int = 3,
) -> List[Dict]:
    """Load high-confidence test predictions for GRPO training."""
    import pandas as pd

    logger.info(f"Loading test augmentation from {checkpoint_path}")

    with open(checkpoint_path, 'r') as f:
        checkpoint = json.load(f)

    test_df = pd.read_csv(test_csv_path)
    id_to_question = dict(zip(test_df['ID'], test_df['question']))

    samples = []
    for row_key, data in checkpoint.items():
        question_id = data['ID']
        responses = data['responses']

        answers = [r['answer'] for r in responses if r.get('answer')]
        if not answers:
            continue

        answer_counts = Counter(answers)
        most_common_answer, count = answer_counts.most_common(1)[0]

        if count < min_agreement:
            continue

        full_question = id_to_question.get(question_id, data.get('question', ''))
        if not full_question:
            continue

        samples.append({
            'row_id': f"aug_{question_id}",
            'question': full_question,
            'answer': most_common_answer,
            'question_type': get_question_type(full_question),
            'source': 'augmentation',
        })

    logger.info(f"Loaded {len(samples)} augmentation samples (>={min_agreement}/4 agreement)")
    return samples


# =============================================================================
# DATASET PREPARATION
# =============================================================================

def compute_sample_weights(samples: List[Dict]) -> List[float]:
    """Compute per-sample weights for type-balanced sampling.

    Weight = total / (n_types * count_for_type), so each type contributes
    equally to expected samples drawn per epoch.
    """
    type_counts = Counter(s.get('question_type', 'type_a') for s in samples)
    n_types = len(type_counts)
    total = len(samples)

    weights = []
    for s in samples:
        qt = s.get('question_type', 'type_a')
        weights.append(total / (n_types * type_counts[qt]))

    logger.info("Type-balanced sampling weights:")
    for qt in sorted(type_counts):
        w = total / (n_types * type_counts[qt])
        logger.info(f"  {qt}: {w:.2f}x ({type_counts[qt]} samples)")

    return weights


def strip_raw_tables(question: str) -> str:
    """Strip raw data tables from a telco question, keeping instructions + options.

    Works for both Type A and Type B questions.
    Must match the SFT training format exactly.
    """
    lines = question.split('\n')
    preamble_lines = []
    for line in lines:
        if line.count('|') >= 3:
            while preamble_lines and preamble_lines[-1].strip() == '':
                preamble_lines.pop()
            if preamble_lines and 'data as follows' in preamble_lines[-1].lower():
                preamble_lines.pop()
            break
        preamble_lines.append(line)

    result = '\n'.join(preamble_lines).strip()
    return result if result else question


def compute_type_a_metrics_for_question(question: str):
    """Compute Type A metrics block for a question. Returns formatted string or None."""
    try:
        drive_test, cells = parse_type_a_question(question)
        if drive_test:
            metrics = compute_all_metrics(question, drive_test, cells)
            return format_metrics_block(metrics)
    except Exception as e:
        logger.debug(f"Failed to compute Type A metrics: {e}")
    return None


def compute_type_b_metrics_for_question(question: str):
    """Compute Type B metrics block for a question. Returns formatted string or None."""
    try:
        m = compute_type_b_metrics(question)
        if m is not None:
            return format_type_b_metrics_block(m)
    except Exception as e:
        logger.debug(f"Failed to compute Type B metrics: {e}")
    return None


def prepare_grpo_dataset(
    samples: List[Dict],
    tokenizer,
) -> Dataset:
    """
    Prepare dataset for GRPO training.

    Pre-computes metrics and strips raw tables to match SFT training format.
    GRPO needs:
    - prompt: list of messages (system + user)
    - answer: ground truth for reward computation
    """
    formatted = []
    metrics_computed = 0
    metrics_failed = 0

    for sample in tqdm(samples, desc="Formatting prompts"):
        question = sample['question']
        answer = sample['answer']
        question_type = sample.get('question_type', 'type_a')

        # Select system prompt and compute metrics - must match SFT format
        if question_type == 'type_b':
            system_prompt = TYPE_B_SYSTEM_PROMPT
            metrics_block = compute_type_b_metrics_for_question(question)
        elif question_type == 'generic':
            system_prompt = GENERIC_SYSTEM_PROMPT
            metrics_block = None
        else:
            system_prompt = TELCO_SYSTEM_PROMPT
            metrics_block = compute_type_a_metrics_for_question(question)

        # Build user message matching SFT training format
        if metrics_block:
            question_preamble = strip_raw_tables(question)
            user_content = f"## Pre-computed Metrics\n\n{metrics_block}\n\n## Question\n\n{question_preamble}"
            metrics_computed += 1
        else:
            user_content = question
            if question_type != 'generic':
                metrics_failed += 1

        prompt = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_content},
        ]

        formatted.append({
            'prompt': prompt,
            'answer': answer,
            'row_id': sample.get('row_id', 'unknown'),
            'source': sample.get('source', 'unknown'),
        })

    logger.info(f"Metrics computed: {metrics_computed}, failed: {metrics_failed}")

    dataset = Dataset.from_list(formatted)

    # Analyze prompt lengths
    logger.info("Analyzing prompt lengths...")
    lengths = []
    for i in range(min(50, len(dataset))):
        text = tokenizer.apply_chat_template(
            dataset[i]['prompt'],
            tokenize=True,
            add_generation_prompt=True,
        )
        lengths.append(len(text))

    logger.info(f"Prompt length stats (first {len(lengths)} samples):")
    logger.info(f"  Min: {min(lengths)}")
    logger.info(f"  Max: {max(lengths)}")
    logger.info(f"  Mean: {sum(lengths)/len(lengths):.0f}")

    return dataset


# =============================================================================
# LOAD SFT ADAPTER CONFIG
# =============================================================================

def load_sft_adapter_config(sft_model_path: str):
    """
    Load adapter configuration from SFT model.
    Returns (rank, target_modules, lora_alpha) or defaults if not found.
    """
    from peft import PeftConfig
    from huggingface_hub import hf_hub_download

    try:
        config = PeftConfig.from_pretrained(sft_model_path)
        logger.info(f"Loaded SFT adapter config from: {sft_model_path}")
        logger.info(f"  rank (r): {config.r}")
        logger.info(f"  target_modules: {list(config.target_modules)}")
        logger.info(f"  lora_alpha: {config.lora_alpha}")
        return config.r, list(config.target_modules), config.lora_alpha
    except Exception as e:
        logger.warning(f"Could not load PeftConfig: {e}")

    try:
        config_path = os.path.join(sft_model_path, "adapter_config.json")
        if not os.path.exists(config_path):
            config_path = hf_hub_download(
                repo_id=sft_model_path,
                filename="adapter_config.json",
            )

        with open(config_path, 'r') as f:
            config = json.load(f)

        rank = config.get('r', 32)
        target_modules = config.get('target_modules', [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ])
        lora_alpha = config.get('lora_alpha', rank * 2)

        logger.info(f"Loaded SFT adapter config from adapter_config.json")
        logger.info(f"  rank (r): {rank}")
        logger.info(f"  target_modules: {target_modules}")
        logger.info(f"  lora_alpha: {lora_alpha}")
        return rank, target_modules, lora_alpha
    except Exception as e:
        logger.warning(f"Could not load adapter_config.json: {e}")

    logger.warning("Using default LoRA config (r=32)")
    return 32, [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ], 64


# =============================================================================
# TRAINING
# =============================================================================

def train(
    sft_model_path: str,
    base_model: str,
    train_checkpoint_path: str,
    test_checkpoint_path: str,
    test_csv_path: str,
    output_dir: str,
    hf_repo: str = None,
    hf_token: str = None,
    max_seq_length: int = 8192,
    lora_rank: int = None,
    max_steps: int = 100,
    num_generations: int = 6,
    learning_rate: float = 5e-6,
    temperature: float = 1.0,
    gradient_accumulation_steps: int = 4,
    gpu_memory_utilization: float = 0.95,
    min_agreement: int = 3,
    use_augmentation: bool = True,
    push_to_hub: bool = False,
    merge_16bit: bool = False,
    fast_inference: bool = True,
    dry_run: bool = False,
    seed: int = 42,
):
    """Main GRPO training function."""

    logger.info("=" * 60)
    logger.info("QWEN3-32B V19 GRPO TRAINING")
    logger.info("Rewards: boxed_reward, think_tags_reward, accuracy_reward")
    logger.info("=" * 60)

    # =================================
    # Load data
    # =================================
    train_traces = load_v19_traces(train_checkpoint_path)

    augmentation_samples = []
    if use_augmentation and os.path.exists(test_checkpoint_path):
        augmentation_samples = load_test_augmentation(
            test_checkpoint_path,
            test_csv_path,
            min_agreement=min_agreement,
        )

    all_samples = train_traces + augmentation_samples

    logger.info(f"\nDataset summary:")
    logger.info(f"  V19 traces: {len(train_traces)}")
    logger.info(f"  Augmentation: {len(augmentation_samples)}")
    logger.info(f"  Total: {len(all_samples)}")

    if len(all_samples) == 0:
        logger.error("No samples found!")
        return

    # Analyze answer distribution
    answer_counts = Counter(s['answer'] for s in all_samples)
    logger.info(f"\nAnswer distribution ({len(answer_counts)} unique):")
    for ans, cnt in sorted(answer_counts.items(), key=lambda x: -x[1])[:10]:
        logger.info(f"  {ans}: {cnt}")

    if dry_run:
        logger.info("\nDRY RUN - Data validation complete!")
        logger.info("Sample prompts:")
        for i, sample in enumerate(all_samples[:3]):
            logger.info(f"\n--- Sample {i+1} ({sample.get('question_type', '?')}) ---")
            logger.info(f"Question: {sample['question'][:200]}...")
            logger.info(f"Answer: {sample['answer']}")
        return

    # =================================
    # Load SFT adapter config
    # =================================
    sft_rank, sft_target_modules, sft_lora_alpha = load_sft_adapter_config(sft_model_path)

    if lora_rank is None:
        lora_rank = sft_rank
    elif lora_rank != sft_rank:
        logger.warning(f"CLI --lora-rank={lora_rank} differs from SFT rank={sft_rank}. Using CLI value.")

    # =================================
    # Load model
    # =================================
    from unsloth import is_bfloat16_supported

    logger.info(f"\nLoading base model: {base_model}")
    logger.info(f"SFT LoRA adapter: {sft_model_path}")
    logger.info(f"Fast inference (vLLM): {fast_inference}")

    from_pretrained_kwargs = {
        "model_name": base_model,
        "max_seq_length": max_seq_length,
        "load_in_4bit": True,
        "fast_inference": fast_inference,
    }
    if fast_inference:
        from_pretrained_kwargs["max_lora_rank"] = lora_rank
        from_pretrained_kwargs["gpu_memory_utilization"] = gpu_memory_utilization

    model, tokenizer = FastLanguageModel.from_pretrained(**from_pretrained_kwargs)

    logger.info(f"Setting up LoRA: rank={lora_rank}, target_modules={sft_target_modules}")

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=sft_target_modules,
        lora_alpha=sft_lora_alpha,
        use_gradient_checkpointing="unsloth",
        random_state=seed,
    )

    # Load SFT LoRA weights
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    from huggingface_hub import hf_hub_download

    local_weights_path = os.path.join(sft_model_path, "adapter_model.safetensors")
    if os.path.exists(local_weights_path):
        sft_weights_path = local_weights_path
    else:
        try:
            sft_weights_path = hf_hub_download(
                repo_id=sft_model_path,
                filename="adapter_model.safetensors",
            )
        except Exception as e:
            logger.error(f"Could not download SFT weights: {e}")
            raise RuntimeError("Failed to load SFT weights. Check --sft-model path.")

    sft_state_dict = load_file(sft_weights_path)
    logger.info(f"Loading {len(sft_state_dict)} weight tensors from SFT")

    sft_keys = list(sft_state_dict.keys())
    model_keys = [k for k in model.state_dict().keys() if 'lora' in k.lower()]
    logger.info(f"SFT adapter key example: {sft_keys[0] if sft_keys else 'none'}")
    logger.info(f"Model LoRA key example: {model_keys[0] if model_keys else 'none'}")

    try:
        set_peft_model_state_dict(model, sft_state_dict)
        logger.info(f"Loaded SFT weights via set_peft_model_state_dict from: {sft_model_path}")
    except Exception as e:
        logger.warning(f"set_peft_model_state_dict failed: {e}, trying manual key mapping...")
        fixed_state_dict = {}
        for key, value in sft_state_dict.items():
            new_key = key
            for prefix in ['base_model.model.', 'base_model.']:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
                    break
            fixed_state_dict[new_key] = value
        missing, unexpected = model.load_state_dict(fixed_state_dict, strict=False)
        loaded = len(sft_state_dict) - len(unexpected)
        logger.info(f"Manual loading: {loaded}/{len(sft_state_dict)} tensors loaded")
        if unexpected:
            logger.warning(f"Could not load {len(unexpected)} tensors (key mismatch)")

    model.print_trainable_parameters()

    # =================================
    # Prepare dataset
    # =================================
    dataset = prepare_grpo_dataset(all_samples, tokenizer)
    logger.info(f"\nGRPO dataset: {len(dataset)} samples")

    logger.info("\nSample prompt:")
    sample_text = tokenizer.apply_chat_template(
        dataset[0]['prompt'],
        tokenize=False,
        add_generation_prompt=True,
    )
    logger.info(f"Length: {len(sample_text)} chars")
    logger.info(f"Preview:\n{sample_text[:500]}...")

    # =================================
    # GRPO Configuration
    # =================================
    from vllm import SamplingParams
    from trl import GRPOConfig, GRPOTrainer

    # With pre-computed metrics, prompts are ~1650 tokens max (Type A).
    # 2048 gives comfortable headroom without wasting seq budget on padding.
    min_prompt_budget = 2048
    max_prompt_length = min(min_prompt_budget, max_seq_length - 1024)
    max_completion_length = max_seq_length - max_prompt_length

    logger.info(f"Token budget: prompt={max_prompt_length}, completion={max_completion_length}")
    if max_prompt_length < 3200:
        logger.warning(
            f"max_prompt_length={max_prompt_length} may truncate long prompts. "
            f"Recommend --max-seq-length 4500 or higher."
        )
    if max_completion_length < 1500:
        logger.warning(
            f"max_completion_length={max_completion_length} limits reasoning space. "
            f"Recommend --max-seq-length 5500 or higher."
        )

    vllm_sampling_params = SamplingParams(
        min_p=0.1,
        top_p=0.95,
        top_k=50,
        repetition_penalty=1.05,
        seed=seed,
        stop=[tokenizer.eos_token],
        include_stop_str_in_output=True,
    )

    training_args = GRPOConfig(
        output_dir=f"{output_dir}/checkpoints",
        vllm_sampling_params=vllm_sampling_params,
        temperature=temperature,
        learning_rate=learning_rate,
        weight_decay=0.001,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        optim="adamw_8bit",
        logging_steps=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_generations=num_generations,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        max_steps=max_steps,
        max_grad_norm=1.0,
        save_steps=max(50, max_steps // 2),
        report_to="none",
        seed=seed,
    )

    logger.info("\n" + "=" * 60)
    logger.info("GRPO CONFIGURATION")
    logger.info("=" * 60)
    logger.info(f"SFT Model: {sft_model_path}")
    logger.info(f"Max steps: {max_steps}")
    logger.info(f"Num generations: {num_generations}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"Temperature: {temperature}")
    logger.info(f"Max prompt length: {max_prompt_length}")
    logger.info(f"Max completion length: {max_completion_length}")
    logger.info(f"Reward functions: boxed_reward, think_tags_reward, accuracy_reward")

    # =================================
    # Create trainer (exactly 3 rewards)
    # =================================
    # Compute per-sample weights for type-balanced sampling
    sample_weights = compute_sample_weights(all_samples)

    # Note: TypeBalancedGRPOTrainer with WeightedRandomSampler was removed because
    # GRPO's dataloader has special requirements for num_generations grouping.
    # Overriding get_train_dataloader breaks the reward reshaping.
    # Type balancing for GRPO is handled via the dataset composition instead.

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            boxed_reward,
            think_tags_reward,
            accuracy_reward,
        ],
        args=training_args,
        train_dataset=dataset,
    )

    logger.info("\n" + "=" * 60)
    logger.info("STARTING GRPO TRAINING")
    logger.info("=" * 60)

    trainer.train()

    # =================================
    # Save model
    # =================================
    logger.info("\n" + "=" * 60)
    logger.info("SAVING MODEL")
    logger.info("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    lora_output_dir = f"{output_dir}/lora"
    model.save_pretrained(lora_output_dir)
    tokenizer.save_pretrained(lora_output_dir)
    logger.info(f"LoRA adapter saved to: {lora_output_dir}")

    config = {
        'base_model': base_model,
        'sft_model': sft_model_path,
        'lora_rank': lora_rank,
        'target_modules': sft_target_modules,
        'lora_alpha': sft_lora_alpha,
        'max_seq_length': max_seq_length,
        'max_steps': max_steps,
        'num_generations': num_generations,
        'learning_rate': learning_rate,
        'temperature': temperature,
        'min_agreement': min_agreement,
        'train_samples': len(train_traces),
        'augmentation_samples': len(augmentation_samples),
        'total_samples': len(dataset),
        'reward_functions': ['boxed_reward', 'think_tags_reward', 'accuracy_reward'],
    }

    with open(f"{output_dir}/grpo_config.json", 'w') as f:
        json.dump(config, f, indent=2)

    # =================================
    # Merge to 16-bit (optional)
    # =================================
    merged_output_dir = None
    if merge_16bit:
        logger.info("\nMerging LoRA to 16-bit model...")
        merged_output_dir = f"{output_dir}/merged_16bit"

        from unsloth import FastLanguageModel as FLM

        merge_model, merge_tokenizer = FLM.from_pretrained(
            model_name=base_model,
            max_seq_length=max_seq_length,
            load_in_4bit=True,
            fast_inference=False,
        )

        merge_model = FLM.get_peft_model(
            merge_model,
            r=lora_rank,
            target_modules=sft_target_modules,
            lora_alpha=sft_lora_alpha,
        )

        from safetensors.torch import load_file as load_safetensors
        lora_weights_path = f"{lora_output_dir}/adapter_model.safetensors"
        if os.path.exists(lora_weights_path):
            state_dict = load_safetensors(lora_weights_path)
            merge_model.load_state_dict(state_dict, strict=False)
            logger.info(f"Loaded LoRA weights from {lora_weights_path}")

        merge_model.save_pretrained_merged(
            merged_output_dir,
            merge_tokenizer,
            save_method="merged_16bit",
        )
        logger.info(f"Merged model saved to: {merged_output_dir}")

    # =================================
    # Push to HuggingFace (optional)
    # =================================
    if push_to_hub and hf_repo:
        logger.info(f"\nPushing to HuggingFace: {hf_repo}")

        if hf_token:
            from huggingface_hub import login
            login(token=hf_token)

        from huggingface_hub import HfApi
        api = HfApi()

        if merge_16bit and merged_output_dir:
            logger.info("Pushing merged 16-bit model...")
            api.create_repo(repo_id=hf_repo, exist_ok=True)
            api.upload_folder(
                folder_path=merged_output_dir,
                repo_id=hf_repo,
                repo_type="model",
                commit_message="Upload V19 GRPO-trained merged model",
            )
            logger.info(f"Merged model pushed to: https://huggingface.co/{hf_repo}")
        else:
            lora_repo = f"{hf_repo}-lora" if not hf_repo.endswith("-lora") else hf_repo
            logger.info(f"Pushing LoRA adapter to: {lora_repo}")
            api.create_repo(repo_id=lora_repo, exist_ok=True)
            api.upload_folder(
                folder_path=lora_output_dir,
                repo_id=lora_repo,
                repo_type="model",
                commit_message="Upload V19 GRPO-trained LoRA adapter",
            )
            logger.info(f"LoRA adapter pushed to: https://huggingface.co/{lora_repo}")

    logger.info("\n" + "=" * 60)
    logger.info("V19 GRPO TRAINING COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"LoRA adapter: {lora_output_dir}")
    if merged_output_dir:
        logger.info(f"Merged model: {merged_output_dir}")
    if push_to_hub and hf_repo:
        logger.info(f"HuggingFace: https://huggingface.co/{hf_repo}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train Qwen3-32B with V19 GRPO (3 rewards: boxed, think tags, accuracy)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model paths
    parser.add_argument(
        '--base-model', type=str,
        default='unsloth/Qwen3-32B-bnb-4bit',
        help='Base model (from HuggingFace)',
    )
    parser.add_argument(
        '--sft-model', type=str, required=True,
        help='Path or HF repo for V19 SFT LoRA adapter',
    )
    parser.add_argument(
        '--output-dir', type=str,
        default='./outputs/qwen3-32b-v19-grpo',
        help='Output directory for GRPO model',
    )

    # HuggingFace Hub
    parser.add_argument(
        '--hf-repo', type=str, default=None,
        help='HuggingFace repo to push model',
    )
    parser.add_argument(
        '--hf-token', type=str, default=None,
        help='HuggingFace token for pushing model',
    )
    parser.add_argument(
        '--push-to-hub', action='store_true',
        help='Push model to HuggingFace Hub after training',
    )
    parser.add_argument(
        '--merge-16bit', action='store_true',
        help='Merge LoRA into 16-bit model before pushing',
    )

    # Data paths
    parser.add_argument(
        '--train-checkpoint', type=str,
        default='./outputs/traces_final/traces_final.json',
        help='Path to training traces JSON',
    )
    parser.add_argument(
        '--test-checkpoint', type=str,
        default='',
        help='(unused) Path to test predictions for augmentation',
    )
    parser.add_argument(
        '--test-csv', type=str,
        default='',
        help='(unused) Path to test CSV for full questions',
    )

    # Model config
    parser.add_argument(
        '--max-seq-length', type=int, default=8192,
        help='Maximum sequence length',
    )
    parser.add_argument(
        '--lora-rank', type=int, default=None,
        help='LoRA rank (default: read from SFT adapter config)',
    )

    # Training config
    parser.add_argument(
        '--max-steps', type=int, default=100,
        help='Maximum training steps',
    )
    parser.add_argument(
        '--num-generations', type=int, default=6,
        help='Number of completions per prompt',
    )
    parser.add_argument(
        '--learning-rate', type=float, default=5e-6,
        help='Learning rate',
    )
    parser.add_argument(
        '--temperature', type=float, default=1.0,
        help='Sampling temperature for generation',
    )
    parser.add_argument(
        '--gradient-accumulation-steps', type=int, default=4,
        help='Gradient accumulation steps',
    )
    parser.add_argument(
        '--gpu-memory-utilization', type=float, default=0.95,
        help='GPU memory utilization for vLLM',
    )

    # Data config
    parser.add_argument(
        '--min-agreement', type=int, default=3, choices=[3, 4],
        help='Minimum agreement for augmentation samples',
    )
    parser.add_argument(
        '--no-augment', action='store_true',
        help='Disable test set augmentation',
    )

    # Utility
    parser.add_argument(
        '--no-fast-inference', action='store_true',
        help='Disable vLLM fast inference',
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Validate data without training',
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed',
    )

    args = parser.parse_args()

    train(
        sft_model_path=args.sft_model,
        base_model=args.base_model,
        train_checkpoint_path=args.train_checkpoint,
        test_checkpoint_path=args.test_checkpoint,
        test_csv_path=args.test_csv,
        output_dir=args.output_dir,
        hf_repo=args.hf_repo,
        hf_token=args.hf_token,
        max_seq_length=args.max_seq_length,
        lora_rank=args.lora_rank,
        max_steps=args.max_steps,
        num_generations=args.num_generations,
        learning_rate=args.learning_rate,
        temperature=args.temperature,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gpu_memory_utilization=args.gpu_memory_utilization,
        min_agreement=args.min_agreement,
        use_augmentation=not args.no_augment,
        push_to_hub=args.push_to_hub,
        merge_16bit=args.merge_16bit,
        fast_inference=not args.no_fast_inference,
        dry_run=args.dry_run,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
