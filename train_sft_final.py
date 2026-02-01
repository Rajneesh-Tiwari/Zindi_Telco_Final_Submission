"""
Train Qwen3-32B with Reasoning Traces (SFT)

Fine-tunes Qwen3-32B on 2400 Type A reasoning traces generated from
the labeled training set (train.csv).

Input format per question type:
- Type A: TELCO_SYSTEM_PROMPT + pre-computed Type A metrics + question preamble
- Type B: TYPE_B_SYSTEM_PROMPT + pre-computed Type B metrics + question preamble
- Both: assistant = <think>cascade walk</think> \\boxed{ANSWER}

Uses Unsloth for efficient 4-bit training with LoRA adapters.

Usage:
    python train_sft_final.py                          # Default training
    python train_sft_final.py --epochs 10              # 10 epochs
    python train_sft_final.py --dry-run                # Validate data without training
"""

import os
import json
import re
import argparse
import logging
from typing import Dict, List, Optional
from collections import Counter

from tqdm import tqdm
from datasets import Dataset

# Metric computation (Type A and Type B)
from telco_utils import parse_type_a_question
from generate_traces_final import (
    compute_all_metrics, format_metrics_block,
    compute_type_b_metrics, format_type_b_metrics_block,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# ANSWER EXTRACTION
# =============================================================================

def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract answer from \\boxed{...} in generated text."""
    matches = re.findall(r'\\boxed\{([^}]+)\}', text)
    if matches:
        return matches[-1].strip()
    return None


# =============================================================================
# SYSTEM PROMPTS
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

OUTPUT FORMAT (mandatory):
1. Wrap ALL reasoning inside <think>...</think> tags.
2. After </think>, output EXACTLY \\boxed{LABEL} where LABEL is the option number/label.
3. Do NOT write anything after \\boxed{LABEL}.
4. Every response MUST end with \\boxed{LABEL}. Omitting it is a failure.

You must strictly output your reasoning process within <think>...</think> tags before the final answer.
Examples: \\boxed{2}, \\boxed{B}, \\boxed{72}"""


# =============================================================================
# PASS@1 EVALUATION CALLBACK
# =============================================================================

def create_pass1_callback(
    model,
    tokenizer,
    val_samples: List[Dict],
    eval_steps: int = 200,
    max_new_tokens: int = 4096,
    eval_subset: Optional[int] = 20,
):
    """Create a TrainerCallback for step-based pass@1 evaluation."""
    from transformers import TrainerCallback
    import torch

    eval_subset = eval_subset or len(val_samples)
    samples_to_eval = val_samples[:eval_subset]

    class Pass1EvalCallback(TrainerCallback):
        def __init__(self):
            self.last_eval_step = -1

        def on_step_end(self, args, state, control, **kwargs):
            current_step = state.global_step
            if current_step > 0 and current_step % eval_steps == 0 and current_step != self.last_eval_step:
                self.last_eval_step = current_step
                self._run_evaluation(current_step)

        def _run_evaluation(self, step: int):
            from unsloth import FastLanguageModel

            logger.info(f"\n{'='*60}")
            logger.info(f"PASS@1 EVALUATION at step {step}")
            logger.info(f"{'='*60}")

            FastLanguageModel.for_inference(model)

            correct = 0
            total = 0
            by_class = Counter()
            correct_by_class = Counter()
            by_type = Counter()
            correct_by_type = Counter()

            for sample in tqdm(samples_to_eval, desc="Evaluating"):
                question = sample['question']
                expected = sample['expected_answer']
                question_type = sample.get('question_type', 'type_a')

                # Build prompt based on question type
                if question_type == 'type_b':
                    system_prompt = TYPE_B_SYSTEM_PROMPT
                    metrics_block = compute_type_b_metrics_for_question(question)
                elif question_type == 'generic':
                    system_prompt = GENERIC_SYSTEM_PROMPT
                    metrics_block = None
                else:
                    system_prompt = TELCO_SYSTEM_PROMPT
                    metrics_block = compute_type_a_metrics_for_question(question)

                if metrics_block:
                    question_preamble = strip_raw_tables(question)
                    user_content = f"## Pre-computed Metrics\n\n{metrics_block}\n\n## Question\n\n{question_preamble}"
                else:
                    user_content = question

                messages = [
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_content},
                ]

                inputs = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    enable_thinking=True,
                ).to(model.device)

                with torch.no_grad():
                    outputs = model.generate(
                        input_ids=inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=0.6,
                        top_p=0.95,
                        do_sample=True,
                        pad_token_id=tokenizer.pad_token_id,
                    )

                response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
                predicted = extract_boxed_answer(response)

                total += 1
                by_class[expected] += 1
                by_type[question_type] += 1
                if predicted == expected:
                    correct += 1
                    correct_by_class[expected] += 1
                    correct_by_type[question_type] += 1

            FastLanguageModel.for_training(model)

            accuracy = correct / total if total > 0 else 0
            logger.info(f"\nStep {step} - Pass@1 Accuracy: {correct}/{total} = {accuracy:.1%}")

            logger.info("Per-class accuracy:")
            for cls in sorted(by_class.keys()):
                cls_correct = correct_by_class[cls]
                cls_total = by_class[cls]
                cls_acc = cls_correct / cls_total if cls_total > 0 else 0
                logger.info(f"  {cls}: {cls_correct}/{cls_total} = {cls_acc:.1%}")

            logger.info("Per-type accuracy:")
            for qt in sorted(by_type.keys()):
                qt_correct = correct_by_type[qt]
                qt_total = by_type[qt]
                qt_acc = qt_correct / qt_total if qt_total > 0 else 0
                logger.info(f"  {qt}: {qt_correct}/{qt_total} = {qt_acc:.1%}")

            logger.info(f"{'='*60}\n")
            return accuracy

    return Pass1EvalCallback()


# =============================================================================
# DATA LOADING
# =============================================================================

def load_training_data(checkpoint_path: str) -> List[Dict]:
    """Load training samples from combined traces JSON.

    Each sample has: question, expected_answer, reasoning_trace, question_type, row_id.
    """
    logger.info(f"Loading training data from {checkpoint_path}")

    with open(checkpoint_path, 'r') as f:
        checkpoint = json.load(f)

    samples = []
    for row_id, data in checkpoint.items():
        samples.append({
            'row_id': row_id,
            'question': data['question'],
            'expected_answer': data['expected_answer'],
            'reasoning_trace': data.get('reasoning_trace', ''),
            'question_type': data.get('question_type', 'type_a'),
        })

    logger.info(f"Loaded {len(samples)} training samples")

    # Show type distribution
    type_counts = Counter(s['question_type'] for s in samples)
    for qt, count in sorted(type_counts.items()):
        logger.info(f"  {qt}: {count}")

    return samples


# =============================================================================
# DATA FORMATTING
# =============================================================================

def strip_raw_tables(question: str) -> str:
    """Strip raw data tables from a telco question, keeping instructions + options.

    Works for both Type A and Type B questions.
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


def compute_type_a_metrics_for_question(question: str) -> Optional[str]:
    """Compute Type A metrics block for a question. Returns formatted string or None."""
    try:
        drive_test, cells = parse_type_a_question(question)
        if drive_test:
            metrics = compute_all_metrics(question, drive_test, cells)
            return format_metrics_block(metrics)
    except Exception as e:
        logger.debug(f"Failed to compute Type A metrics: {e}")
    return None


def compute_type_b_metrics_for_question(question: str) -> Optional[str]:
    """Compute Type B metrics block for a question. Returns formatted string or None."""
    try:
        m = compute_type_b_metrics(question)
        if m is not None:
            return format_type_b_metrics_block(m)
    except Exception as e:
        logger.debug(f"Failed to compute Type B metrics: {e}")
    return None


def strip_metrics_from_trace(trace: str) -> str:
    """Strip the 'Extracted metrics:' block from reasoning trace.

    The metrics are provided in the user message, so remove them
    from the assistant response to avoid redundancy.
    """
    if not trace:
        return trace

    pattern = r'Extracted metrics:\s*\n(?:\s+\S+\s*=\s*[^\n]+\n)+\s*\n?'
    trace = re.sub(pattern, '', trace)

    intro_pattern = r'^I need to identify[^\n]+\n\n+'
    trace = re.sub(intro_pattern, '', trace.strip())

    return trace.strip()


def format_training_example(sample: Dict) -> Dict:
    """Format a single sample into reasoning SFT training format.

    Selects system prompt and metric computation based on question_type.
    """
    question = sample['question']
    answer = sample['expected_answer']
    reasoning = sample.get('reasoning_trace', '')
    question_type = sample.get('question_type', 'type_a')

    # Select system prompt and compute metrics based on question type
    if question_type == 'type_b':
        system_prompt = TYPE_B_SYSTEM_PROMPT
        metrics_block = compute_type_b_metrics_for_question(question)
    elif question_type == 'generic':
        system_prompt = GENERIC_SYSTEM_PROMPT
        metrics_block = None  # Generic questions use raw question text
    else:
        system_prompt = TELCO_SYSTEM_PROMPT
        metrics_block = compute_type_a_metrics_for_question(question)

    # Build user message
    user_content = question
    if metrics_block:
        question_preamble = strip_raw_tables(question)
        user_content = f"## Pre-computed Metrics\n\n{metrics_block}\n\n## Question\n\n{question_preamble}"
        reasoning = strip_metrics_from_trace(reasoning)

    # Format assistant response: reasoning + boxed answer
    if reasoning:
        if '<think>' not in reasoning.lower():
            reasoning_clean = re.sub(r'\\boxed\{[^}]*\}', '', reasoning).strip()
            reasoning = f"<think>\n{reasoning_clean}\n</think>"
        response = f"{reasoning}\n\n\\boxed{{{answer}}}"
    else:
        response = f"<think>\nApplying decision rules to the metrics.\n</think>\n\n\\boxed{{{answer}}}"

    return {
        'messages': [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_content},
            {'role': 'assistant', 'content': response},
        ],
        'answer': answer,
        'row_id': sample.get('row_id', 'unknown'),
        'question_type': question_type,
    }


# =============================================================================
# CLASS ANALYSIS
# =============================================================================

def analyze_class_distribution(samples: List[Dict]) -> Dict[str, int]:
    """Analyze answer class distribution by question type."""
    answers = [s['expected_answer'] for s in samples]
    distribution = Counter(answers)

    logger.info(f"Answer distribution ({len(distribution)} unique):")
    for answer, count in sorted(distribution.items(), key=lambda x: -x[1])[:15]:
        pct = count / len(samples) * 100
        logger.info(f"  {answer}: {count} ({pct:.1f}%)")

    if len(distribution) > 15:
        logger.info(f"  ... and {len(distribution) - 15} more")

    # Per-type breakdown
    for qt in sorted(set(s.get('question_type', 'type_a') for s in samples)):
        qt_samples = [s for s in samples if s.get('question_type', 'type_a') == qt]
        qt_dist = Counter(s['expected_answer'] for s in qt_samples)
        logger.info(f"\n  {qt} ({len(qt_samples)} samples):")
        for answer, count in sorted(qt_dist.items(), key=lambda x: -x[1]):
            logger.info(f"    {answer}: {count}")

    return dict(distribution)


# =============================================================================
# TYPE-BALANCED LOSS WEIGHTING
# =============================================================================

def compute_type_weights(samples: List[Dict]) -> Dict[str, float]:
    """Compute per-type loss weights inversely proportional to frequency.

    Normalizes so the average weight across all samples is 1.0,
    meaning total gradient magnitude is unchanged - only the
    relative contribution per type changes.
    """
    type_counts = Counter(s.get('question_type', 'type_a') for s in samples)
    n_types = len(type_counts)
    total = len(samples)

    # Inverse frequency: weight = total / (n_types * count_for_type)
    weights = {}
    for qt, count in type_counts.items():
        weights[qt] = total / (n_types * count)

    logger.info("Type-balanced loss weights:")
    for qt in sorted(weights):
        logger.info(f"  {qt}: {weights[qt]:.2f}x ({type_counts[qt]} samples)")

    return weights


# =============================================================================
# DATASET PREPARATION
# =============================================================================

def prepare_dataset(
    samples: List[Dict],
    tokenizer,
    type_weights: Dict[str, float] = None,
) -> 'Dataset':
    """Prepare HuggingFace Dataset for training.

    If type_weights is provided, adds a 'loss_weight' column used by
    WeightedSFTTrainer to scale per-sample loss.
    """
    formatted = []
    for sample in tqdm(samples, desc="Formatting examples"):
        example = format_training_example(sample)

        text = tokenizer.apply_chat_template(
            example['messages'],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=True,
        )

        entry = {
            'text': text,
            'answer': example['answer'],
            'row_id': example['row_id'],
            'question_type': example['question_type'],
        }

        if type_weights:
            entry['loss_weight'] = type_weights.get(example['question_type'], 1.0)

        formatted.append(entry)

    dataset = Dataset.from_list(formatted)

    # Analyze token lengths
    logger.info("Analyzing token lengths...")
    token_lengths = []
    for i in range(min(100, len(dataset))):
        tokens = tokenizer(dataset[i]['text'], return_tensors=None)
        token_lengths.append(len(tokens['input_ids']))

    logger.info(f"Token length stats (first {len(token_lengths)} samples):")
    logger.info(f"  Min: {min(token_lengths)}")
    logger.info(f"  Max: {max(token_lengths)}")
    logger.info(f"  Mean: {sum(token_lengths)/len(token_lengths):.0f}")

    return dataset


# =============================================================================
# TRAINING
# =============================================================================

def train(
    train_checkpoint_path: str,
    output_dir: str = 'outputs/qwen3-32b-v19-sft',
    model_name: str = "unsloth/Qwen3-32B-bnb-4bit",
    max_seq_length: int = 8192,
    lora_r: int = 32,
    lora_alpha: int = 64,
    epochs: int = 10,
    batch_size: int = 1,
    gradient_accumulation: int = 8,
    learning_rate: float = 5e-4,
    warmup_ratio: float = 0.1,
    val_split: float = 0.1,
    dry_run: bool = False,
    seed: int = 42,
    eval_steps: int = 200,
    eval_subset: Optional[int] = 20,
    push_to_hub: bool = False,
    hf_repo: Optional[str] = None,
    hf_token: Optional[str] = None,
):
    """Main training function."""
    from sklearn.model_selection import train_test_split

    logger.info("=" * 60)
    logger.info("QWEN3-32B V19 SFT TRAINING (Type A + Type B)")
    logger.info("=" * 60)

    # Load training data
    all_samples = load_training_data(train_checkpoint_path)

    if len(all_samples) == 0:
        logger.error("No samples found!")
        return

    analyze_class_distribution(all_samples)

    # Train/val split - stratify by question_type + answer
    strat_key = [f"{s.get('question_type', 'type_a')}_{s['expected_answer']}" for s in all_samples]

    val_count = max(int(len(all_samples) * val_split), 10)
    val_count = min(val_count, len(all_samples) - 10)

    try:
        train_samples, val_samples = train_test_split(
            all_samples,
            test_size=val_count,
            random_state=seed,
            stratify=strat_key,
        )
    except ValueError:
        logger.warning("Stratified split failed, trying simpler stratification")
        try:
            train_samples, val_samples = train_test_split(
                all_samples,
                test_size=val_count,
                random_state=seed,
                stratify=[s.get('question_type', 'type_a') for s in all_samples],
            )
        except ValueError:
            logger.warning("Type-stratified split also failed, using random split")
            train_samples, val_samples = train_test_split(
                all_samples,
                test_size=val_count,
                random_state=seed,
            )

    logger.info(f"\nTrain: {len(train_samples)}, Val: {len(val_samples)}")

    # Show type distribution in splits
    for label, split in [("Train", train_samples), ("Val", val_samples)]:
        type_counts = Counter(s.get('question_type', 'type_a') for s in split)
        logger.info(f"  {label}: {dict(type_counts)}")

    if dry_run:
        logger.info("\n" + "=" * 60)
        logger.info("DRY RUN - Data validation")
        logger.info("=" * 60)

        # Show sample formatted examples for each type
        for qt in ['type_a', 'type_b']:
            qt_samples = [s for s in train_samples if s.get('question_type', 'type_a') == qt]
            if qt_samples:
                sample = qt_samples[0]
                example = format_training_example(sample)
                logger.info(f"\n--- {qt.upper()} sample (answer: {sample['expected_answer']}) ---")
                logger.info(f"System prompt: {example['messages'][0]['content'][:200]}...")
                logger.info(f"User message length: {len(example['messages'][1]['content'])} chars")
                logger.info(f"User message preview:\n{example['messages'][1]['content'][:500]}")
                logger.info(f"\nAssistant response:\n{example['messages'][2]['content'][:1000]}")
                if len(example['messages'][2]['content']) > 1000:
                    logger.info(f"... ({len(example['messages'][2]['content'])} chars total)")

        logger.info("\n" + "=" * 60)
        logger.info("DRY RUN COMPLETE - data looks valid")
        logger.info("=" * 60)
        return

    # Load model
    from unsloth import FastLanguageModel
    from unsloth import is_bfloat16_supported
    from trl import SFTTrainer
    from transformers import TrainingArguments

    logger.info(f"\nLoading model: {model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        dtype=None,
    )

    logger.info(f"Model loaded. Max sequence length: {max_seq_length}")

    # Setup LoRA
    logger.info(f"Setting up LoRA: r={lora_r}, alpha={lora_alpha}")
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=seed,
    )

    model.print_trainable_parameters()

    # Compute type-balanced loss weights
    type_weights = compute_type_weights(train_samples)

    # Prepare datasets
    train_dataset = prepare_dataset(train_samples, tokenizer, type_weights=type_weights)
    val_dataset = prepare_dataset(val_samples, tokenizer)

    logger.info(f"\nTrain dataset: {len(train_dataset)} samples")
    logger.info(f"Val dataset: {len(val_dataset)} samples")

    # Preview formatted example
    logger.info("\n" + "=" * 60)
    logger.info("SAMPLE FORMATTED EXAMPLE (with chat template)")
    logger.info("=" * 60)
    sample = train_dataset[0]['text']
    logger.info(f"Length: {len(sample)} chars")
    logger.info(f"First 800 chars:\n{sample[:800]}")
    logger.info(f"\nLast 600 chars:\n{sample[-600:]}")

    # Training
    total_steps = len(train_dataset) * epochs // (batch_size * gradient_accumulation)

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING CONFIGURATION")
    logger.info("=" * 60)
    logger.info(f"Epochs: {epochs}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Gradient accumulation: {gradient_accumulation}")
    logger.info(f"Effective batch size: {batch_size * gradient_accumulation}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"Total steps: ~{total_steps}")
    logger.info(f"Pass@1 eval: every {eval_steps} steps on {eval_subset or len(val_samples)} samples")

    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=f"{output_dir}/checkpoints",
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        warmup_ratio=warmup_ratio,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        seed=seed,
        save_strategy="epoch",
        save_total_limit=3,
        eval_strategy="epoch",
        report_to="none",
    )

    callbacks = []
    if eval_steps > 0:
        pass1_callback = create_pass1_callback(
            model=model,
            tokenizer=tokenizer,
            val_samples=val_samples,
            eval_steps=eval_steps,
            max_new_tokens=4096,
            eval_subset=eval_subset,
        )
        callbacks.append(pass1_callback)
    else:
        logger.info("Pass@1 evaluation disabled (eval_steps=0)")

    # Custom trainer that scales loss by per-sample type weight
    class WeightedSFTTrainer(SFTTrainer):
        """SFTTrainer with per-sample loss weighting for type balancing."""

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            # Extract weights before they reach the model (not a model input)
            weights = inputs.pop('loss_weight', None)

            loss, outputs = super().compute_loss(model, inputs, return_outputs=True, **kwargs)

            if weights is not None:
                # weights shape: (batch_size,) - scale the mean loss
                import torch
                weights = weights.to(loss.device).float()
                loss = loss * weights.mean()

            return (loss, outputs) if return_outputs else loss

    trainer = WeightedSFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        packing=False,
        callbacks=callbacks if callbacks else None,
    )

    logger.info("\n" + "=" * 60)
    logger.info("STARTING TRAINING")
    logger.info("=" * 60)

    trainer.train()

    # Save model
    logger.info("\n" + "=" * 60)
    logger.info("SAVING MODEL")
    logger.info("=" * 60)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    config = {
        'model': model_name,
        'lora_r': lora_r,
        'lora_alpha': lora_alpha,
        'max_seq_length': max_seq_length,
        'epochs': epochs,
        'learning_rate': learning_rate,
        'total_train_samples': len(train_dataset),
        'val_samples': len(val_dataset),
        'question_types': dict(Counter(s.get('question_type', 'type_a') for s in all_samples)),
    }

    with open(f"{output_dir}/training_config.json", 'w') as f:
        json.dump(config, f, indent=2)

    logger.info(f"Model saved to: {output_dir}")

    if push_to_hub and hf_repo:
        logger.info(f"\nPushing LoRA adapter to HuggingFace Hub: {hf_repo}")
        model.push_to_hub(hf_repo, token=hf_token)
        tokenizer.push_to_hub(hf_repo, token=hf_token)
        logger.info(f"Pushed to: https://huggingface.co/{hf_repo}")

    logger.info("Training complete!")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train Qwen3-32B with V19 Type A + Type B reasoning traces",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        '--train-checkpoint',
        type=str,
        default='outputs/traces_final/traces_final.json',
        help='Path to training traces JSON',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/qwen3-32b-v19-sft',
        help='Output directory for trained model',
    )
    parser.add_argument(
        '--model',
        type=str,
        default='unsloth/Qwen3-32B-bnb-4bit',
        help='Base model name',
    )
    parser.add_argument(
        '--max-seq-length',
        type=int,
        default=8192,
        help='Maximum sequence length',
    )
    parser.add_argument(
        '--lora-r',
        type=int,
        default=32,
        help='LoRA rank',
    )
    parser.add_argument(
        '--lora-alpha',
        type=int,
        default=64,
        help='LoRA alpha (typically 2x rank)',
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Number of training epochs',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='Per-device batch size',
    )
    parser.add_argument(
        '--gradient-accumulation',
        type=int,
        default=8,
        help='Gradient accumulation steps',
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=5e-4,
        help='Learning rate',
    )
    parser.add_argument(
        '--warmup-ratio',
        type=float,
        default=0.1,
        help='Warmup ratio',
    )
    parser.add_argument(
        '--val-split',
        type=float,
        default=0.1,
        help='Validation split ratio',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate data without training',
    )
    parser.add_argument(
        '--eval-steps',
        type=int,
        default=200,
        help='Run pass@1 evaluation every N steps (0 to disable)',
    )
    parser.add_argument(
        '--eval-subset',
        type=int,
        default=20,
        help='Evaluate on subset of validation samples',
    )
    parser.add_argument(
        '--push-to-hub',
        action='store_true',
        help='Push LoRA adapter to HuggingFace Hub after training',
    )
    parser.add_argument(
        '--hf-repo',
        type=str,
        default=None,
        help='HuggingFace repo name (e.g. Phaedrus33/qwen3-32b-v19-sft)',
    )
    parser.add_argument(
        '--hf-token',
        type=str,
        default=None,
        help='HuggingFace token (uses cached login if not set)',
    )

    args = parser.parse_args()

    train(
        train_checkpoint_path=args.train_checkpoint,
        output_dir=args.output_dir,
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        val_split=args.val_split,
        dry_run=args.dry_run,
        seed=args.seed,
        eval_steps=args.eval_steps,
        eval_subset=args.eval_subset,
        push_to_hub=args.push_to_hub,
        hf_repo=args.hf_repo,
        hf_token=args.hf_token,
    )


if __name__ == "__main__":
    main()
