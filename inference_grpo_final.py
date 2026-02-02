"""
vLLM Inference for GRPO-trained Qwen3-32B.

Runs inference using a GRPO/SFT-trained model with vLLM.
System prompts embed the full rule set for Type A, Type B, and Generic questions.

Supports two loading modes:
1. Merged model (recommended): Load pre-merged model from HF
2. Base + LoRA: Load base model and apply LoRA adapter

Features:
- System prompts for Type A, Type B, and Generic questions
- Checkpointing with resume support
- Multi-GPU tensor parallelism
- Majority voting across generations

Usage:
    # Merged model (recommended)
    python inference_grpo_final.py --model USERNAME/grpo-final

    # Base + LoRA
    python inference_grpo_final.py --model unsloth/Qwen3-32B --lora USERNAME/grpo-lora

    # Resume from checkpoint
    python inference_grpo_final.py --model USERNAME/model --checkpoint ./outputs/inference/checkpoint.json

    # Dry run
    python inference_grpo_final.py --model USERNAME/model --dry-run
"""

import os
import re
import json
import argparse
import logging
from datetime import datetime
from typing import List, Dict, Optional, Set
from collections import Counter

import torch
import pandas as pd
from tqdm import tqdm

# Metric computation (must match SFT training format)
from telco_utils import (
    parse_type_a_question,
    extract_type_a_options,
    classify_question_type,
    parse_tables_generic,
)
from generate_traces_final import (
    compute_all_metrics, format_metrics_block,
    compute_type_b_metrics, format_type_b_metrics_block,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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
7. If phy_healthy_during_low_tp = False AND low_tp_avg_mcs < 12 AND neighbors_within_3dB >= 1:
   -> "overlap coverage" (A)

CONFIGURATION CHECK (if no rule above matches):
8. If inter_freq_ho = True AND a2_thld > -100 AND n_configured_neighbors >= 6:
   -> "inter-freq HO threshold unreasonable" (B)

Show your reasoning inside <think> tags. Then match the root cause to the option that describes it and answer with EXACTLY \\boxed{LABEL}. You MUST always end your response with \\boxed{LABEL}.
Examples: \\boxed{A}, \\boxed{D}, \\boxed{G}"""


GENERIC_SYSTEM_PROMPT = """You are an expert problem solver. Analyze questions carefully and select the correct answer.

IMPORTANT - Answer Format:
- Use the EXACT option number/label from the question
- Examples: \\boxed{2}, \\boxed{B}, \\boxed{72}

You must strictly output your reasoning process within <think>...</think> tags before the final answer."""


# =============================================================================
# QUESTION TYPE DETECTION
# =============================================================================

def get_question_type(question: str) -> str:
    """Detect question type: 'type_a', 'type_b', or 'generic'.

    Type A: "Analyze the 5G wireless network..." with N potential root causes (681 questions)
    Type B: "Based on the following drive test data segment..." (100 questions)
    Generic: "Analyze the following question..." - math/history (82 questions)
    """
    q_stripped = question.strip()

    # Type A: 5G network analysis with 5/6/7/8 root causes
    if q_stripped.startswith("Analyze the 5G wireless network"):
        return 'type_a'

    # Type B: drive test data segment analysis
    if q_stripped.startswith("Based on the following drive test"):
        return 'type_b'

    # Generic: math, history, reading comprehension
    if q_stripped.startswith("Analyze the following question"):
        return 'generic'

    # Fallback heuristics for unexpected formats
    q_lower = question.lower()
    # Type A mentions N root causes (5/6/7/8) and uses 600Mbps threshold
    if 'potential root causes' in q_lower and '|' in question and '600' in question:
        return 'type_a'
    # Type B uses 100Mbps threshold - the key discriminator vs Type A
    if 'throughput drop' in q_lower and '100mbps' in q_lower.replace(' ', ''):
        return 'type_b'

    # Content-based fallback for differently-worded questions
    has_tables = question.count('|') >= 6

    if has_tables:
        telco_option_keywords = [
            'downtilt', 'overshooting', 'over-shooting', 'handover',
            'pci mod 30', 'pci collision', 'scheduled rbs', 'overlapping coverage',
            'neighboring cell', 'weak coverage', 'interference',
        ]
        telco_column_keywords = [
            'rsrp', 'sinr', 'pci', 'throughput', 'gnodeb', 'bler',
            'handover', 'arfcn',
        ]

        option_hits = sum(1 for kw in telco_option_keywords if kw in q_lower)
        column_hits = sum(1 for kw in telco_column_keywords if kw in q_lower)

        if option_hits >= 2 or column_hits >= 3:
            # Distinguish Type A vs Type B by table style
            # Type B uses markdown tables (leading/trailing pipes, separator rows with ---)
            if re.search(r'^\s*\|.*\|.*\|\s*$', question, re.MULTILINE) and '---' in question:
                return 'type_b'
            else:
                return 'type_a'

    return 'generic'


# =============================================================================
# PROMPT FORMATTING
# =============================================================================

def strip_raw_tables(question: str) -> str:
    """Strip raw data tables from a telco question, keeping instructions + options."""
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


def format_prompt(question: str, tokenizer=None) -> str:
    """Format question with appropriate V19 system prompt and pre-computed metrics.

    Must match SFT training format: pre-computed metrics block + question preamble
    (raw tables stripped) for Type A and Type B. Generic questions pass through as-is.
    """
    question = question.replace('\uff1a', ':')

    q_type = get_question_type(question)
    if q_type == 'type_b':
        system_prompt = TYPE_B_SYSTEM_PROMPT
        metrics_block = compute_type_b_metrics_for_question(question)
    elif q_type == 'type_a':
        system_prompt = TELCO_SYSTEM_PROMPT
        metrics_block = compute_type_a_metrics_for_question(question)
    else:
        system_prompt = GENERIC_SYSTEM_PROMPT
        metrics_block = None

    # Fallback: if specific parser failed but tables exist, use generic parser
    if metrics_block is None and question.count('|') >= 6:
        generic_summary = parse_tables_generic(question)
        if generic_summary:
            metrics_block = generic_summary
            logger.warning(f"Using generic table fallback for {q_type} question")

    # Build user message matching SFT training format
    if metrics_block:
        question_preamble = strip_raw_tables(question)
        user_content = f"## Pre-computed Metrics\n\n{metrics_block}\n\n## Question\n\n{question_preamble}"
    else:
        user_content = question

    if tokenizer is not None:
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_content},
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    # Fallback to manual Qwen format
    return f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{user_content}<|im_end|>
<|im_start|>assistant
"""


# =============================================================================
# ANSWER EXTRACTION
# =============================================================================

BOXED_PATTERN = re.compile(r'\\boxed\s*\{\s*([^}]+?)\s*\}')


def extract_answer(response: str) -> Optional[str]:
    """Extract answer from \\boxed{...} in response."""
    match = BOXED_PATTERN.search(response)
    if match:
        return match.group(1).strip()

    # Fallback: "the answer is X" or "answer: X"
    answer_patterns = [
        r'(?:the\s+)?answer\s+is\s*:?\s*([A-Z]?\d+|[A-Z])',
        r'(?:final\s+)?answer\s*:\s*([A-Z]?\d+|[A-Z])',
        r'\\boxed\s*([A-Z]?\d+|[A-Z])',
    ]
    for pattern in answer_patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    # Fallback: last C1-C8 pattern
    match = re.search(r'\b(C[1-8])\b(?!.*\b(C[1-8])\b)', response)
    if match:
        return match.group(1)

    return None


def format_response(response: str) -> str:
    """Ensure response has boxed answer."""
    response = response.strip()
    if BOXED_PATTERN.search(response):
        return response
    answer = extract_answer(response)
    if answer:
        return f"{response}\n\n\\boxed{{{answer}}}"
    return response


# =============================================================================
# OPTION LABEL REMAPPING
# =============================================================================

def remap_submission(sub_path: str, test_df: pd.DataFrame, output_path: str = None):
    """Remap model answers to valid option labels.

    The model was trained on train.csv where Type A options always use the C
    prefix (C1-C8). At inference on Phase 2, option prefixes are shuffled
    (E1-E8, Z1-Z8, bare 1-8, etc.). This function handles three cases:
    1. Answer is already a valid option label - leave as-is
    2. Answer is a bare digit (e.g. "3") but options are letter-prefixed - add prefix
    3. Answer is C[1-8] but options use a different prefix - map canonical to option
    """
    sub = pd.read_csv(sub_path)
    if output_path is None:
        output_path = sub_path  # overwrite in place

    # Build question ID -> option map
    id_to_optmap = {}
    for _, row in test_df.iterrows():
        qid = row['ID']
        qt = classify_question_type(row['question'])
        if qt == 'type_a_telco':
            id_to_optmap[qid] = extract_type_a_options(row['question'])

    remapped_count = 0
    new_answers = []
    for _, row in sub.iterrows():
        response = row['Qwen3-32B']
        base_id = '_'.join(row['ID'].rsplit('_', 1)[:-1])

        optmap = id_to_optmap.get(base_id, {})
        if not optmap:
            new_answers.append(response)
            continue

        match = BOXED_PATTERN.search(str(response))
        if not match:
            new_answers.append(response)
            continue

        answer = match.group(1).strip()
        option_labels = set(optmap.keys())
        reverse_map = {v: k for k, v in optmap.items()}
        new_answer = answer

        if answer in option_labels:
            # Already valid
            pass
        elif re.match(r'^\d+$', answer):
            # Bare digit: find option label ending with that digit
            for label in option_labels:
                if label.endswith(answer) and len(label) > len(answer):
                    new_answer = label
                    break
        elif re.match(r'^C[1-8]$', answer) and answer not in option_labels:
            # Canonical cause: map to option label
            if answer in reverse_map:
                new_answer = reverse_map[answer]

        if new_answer != answer:
            new_response = response.replace(
                f'\\boxed{{{answer}}}', f'\\boxed{{{new_answer}}}'
            )
            new_answers.append(new_response)
            remapped_count += 1
        else:
            new_answers.append(response)

    sub['Qwen3-32B'] = new_answers
    sub.to_csv(output_path, index=False)
    logger.info(f"Remapped {remapped_count} answers in {output_path}")
    return remapped_count


def majority_vote(answers: List[str]) -> str:
    """Pick the most common answer. Break ties by first occurrence."""
    if not answers:
        return "PLACEHOLDER"
    counts = Counter(answers)
    max_count = max(counts.values())
    # Among tied answers, pick the one that appeared first
    for ans in answers:
        if counts[ans] == max_count:
            return ans
    return answers[0]


# =============================================================================
# CHECKPOINTING
# =============================================================================

def load_checkpoint(checkpoint_path: str) -> Dict:
    """Load checkpoint from disk."""
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            checkpoint = json.load(f)
        logger.info(f"Loaded checkpoint with {len(checkpoint.get('results', []))} results")
        return checkpoint
    return {'results': [], 'processed_ids': []}


def save_checkpoint(checkpoint_path: str, results: List[Dict], processed_ids: List[str]):
    """Save checkpoint to disk (atomic write)."""
    checkpoint = {
        'results': results,
        'processed_ids': processed_ids,
        'last_updated': datetime.now().isoformat(),
    }
    temp_path = checkpoint_path + '.tmp'
    with open(temp_path, 'w') as f:
        json.dump(checkpoint, f)
    os.replace(temp_path, checkpoint_path)
    logger.info(f"Checkpoint saved: {len(results)} results, {len(processed_ids)} questions processed")


def get_processed_ids(checkpoint: Dict) -> Set[str]:
    """Get set of already-processed question IDs."""
    return set(checkpoint.get('processed_ids', []))


# =============================================================================
# GPU SETUP
# =============================================================================

def setup_gpu():
    """Clear GPU memory and show status."""
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    for i in range(torch.cuda.device_count()):
        with torch.cuda.device(i):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"Number of GPUs: {torch.cuda.device_count()}")

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        free_mem = torch.cuda.mem_get_info(i)[0] / 1e9
        total_mem = props.total_memory / 1e9
        logger.info(f"  GPU {i}: {props.name} - {free_mem:.1f}/{total_mem:.1f} GB FREE")

    return torch.cuda.device_count()


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model(
    model_path: str,
    lora_path: Optional[str] = None,
    num_gpus: int = 1,
    max_model_len: int = 8192,
    gpu_memory_utilization: float = 0.95,
    hf_token: Optional[str] = None,
):
    """Load model with vLLM."""
    from vllm import LLM

    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token

    if num_gpus > 1:
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        os.environ["NCCL_DEBUG"] = "WARN"

    logger.info(f"Loading model: {model_path}")
    logger.info(f"  GPUs: {num_gpus}")
    logger.info(f"  Max context: {max_model_len}")
    logger.info(f"  GPU utilization: {gpu_memory_utilization}")
    if lora_path:
        logger.info(f"  LoRA adapter: {lora_path}")

    gpu_name = torch.cuda.get_device_properties(0).name.lower()
    if 'a100' in gpu_name or 'h100' in gpu_name or 'h200' in gpu_name:
        dtype = "bfloat16"
    else:
        dtype = "float16"
    logger.info(f"  Dtype: {dtype}")

    llm_kwargs = {
        "model": model_path,
        "max_model_len": max_model_len,
        "gpu_memory_utilization": gpu_memory_utilization,
        "tensor_parallel_size": num_gpus,
        "trust_remote_code": True,
        "dtype": dtype,
        "enforce_eager": False,
    }

    if hf_token:
        llm_kwargs["hf_token"] = hf_token

    if lora_path:
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = 64

    llm = LLM(**llm_kwargs)
    logger.info("Model loaded successfully!")
    return llm, lora_path


def load_tokenizer(model_path: str, hf_token: Optional[str] = None):
    """Load tokenizer for chat template."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        token=hf_token,
    )
    return tokenizer


# =============================================================================
# INFERENCE
# =============================================================================

def run_inference(
    llm,
    test_df: pd.DataFrame,
    tokenizer,
    checkpoint_path: str,
    lora_path: Optional[str] = None,
    num_generations: int = 4,
    temperature: float = 0.6,
    max_tokens: int = 4096,
    batch_size: int = 32,
    checkpoint_every: int = 10,
) -> pd.DataFrame:
    """Run batch inference on test set with checkpointing."""
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        n=num_generations,
        temperature=temperature,
        min_p=0.1,
        top_p=0.95,
        top_k=50,
        max_tokens=max_tokens,
        repetition_penalty=1.05,
    )

    lora_request = None
    if lora_path:
        from vllm.lora.request import LoRARequest
        lora_request = LoRARequest("grpo_adapter", 1, lora_path)

    checkpoint = load_checkpoint(checkpoint_path)
    results = checkpoint.get('results', [])
    processed_ids = get_processed_ids(checkpoint)

    remaining_df = test_df[~test_df['ID'].isin(processed_ids)]

    if len(remaining_df) < len(test_df):
        logger.info(f"Resuming: {len(processed_ids)} already done, {len(remaining_df)} remaining")

    if len(remaining_df) == 0:
        logger.info("All questions already processed!")
        return pd.DataFrame(results)

    total = len(remaining_df)
    batches_since_checkpoint = 0

    logger.info(f"Running inference on {total} questions")
    logger.info(f"  Generations per question: {num_generations}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Total outputs: {total * num_generations}")

    for i in tqdm(range(0, total, batch_size), desc="Inference"):
        batch = remaining_df.iloc[i:min(i + batch_size, total)]

        prompts = [format_prompt(row['question'], tokenizer) for _, row in batch.iterrows()]
        ids = batch['ID'].tolist()

        if lora_request:
            outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
        else:
            outputs = llm.generate(prompts, sampling_params)

        for qid, output in zip(ids, outputs):
            for j, gen in enumerate(output.outputs, 1):
                response = format_response(gen.text)
                results.append({
                    'ID': f'{qid}_{j}',
                    'Qwen3-32B': response,
                    'Qwen2.5-7B-Instruct': 'placeholder',
                    'Qwen2.5-1.5B-Instruct': 'placeholder',
                })
            processed_ids.add(qid)

        batches_since_checkpoint += 1
        if batches_since_checkpoint >= checkpoint_every:
            save_checkpoint(checkpoint_path, results, list(processed_ids))
            batches_since_checkpoint = 0

    save_checkpoint(checkpoint_path, results, list(processed_ids))
    return pd.DataFrame(results)


def _group_by_question(results_df: pd.DataFrame) -> Dict[str, List]:
    """Group result rows by base question ID."""
    rows_by_qid = {}
    for _, row in results_df.iterrows():
        full_id = row['ID']
        parts = full_id.rsplit('_', 1)
        base_id = parts[0]
        if base_id not in rows_by_qid:
            rows_by_qid[base_id] = []
        rows_by_qid[base_id].append(row)
    return rows_by_qid


def _build_submission(rows_by_qid: Dict, use_plurality: bool = False) -> pd.DataFrame:
    """Build submission CSV from grouped results with voting.

    Args:
        rows_by_qid: question ID -> list of result rows
        use_plurality: if True, also apply 2-1-1 plurality voting.
            if False, only apply when top answer has 3+ votes (strict majority).
    """
    submission_rows = []
    vote_stats = {'majority': 0, 'plurality': 0, 'tie': 0, 'unanimous': 0}

    for base_id, rows in rows_by_qid.items():
        answers = []
        for r in rows:
            ans = extract_answer(r['Qwen3-32B'])
            if ans:
                answers.append(ans)

        if not answers:
            best_answer = "PLACEHOLDER"
            keep_raw = False
        else:
            counts = Counter(answers)
            top_answer, top_count = counts.most_common(1)[0]
            keep_raw = False

            if top_count == len(answers):
                best_answer = top_answer
                vote_stats['unanimous'] += 1
            elif top_count >= 3:
                best_answer = top_answer
                vote_stats['majority'] += 1
            elif use_plurality and top_count >= 2:
                # Check it's not a tie (e.g., 2-2)
                second_count = counts.most_common(2)[1][1] if len(counts) > 1 else 0
                if top_count > second_count:
                    best_answer = top_answer
                    vote_stats['plurality'] += 1
                else:
                    # 2-2 tie: keep original raw answers (both options represented)
                    keep_raw = True
                    best_answer = None
                    vote_stats['tie'] += 1
            else:
                # No majority - pick most common (first occurrence for ties)
                best_answer = majority_vote(answers)
                vote_stats['tie'] += 1

        if keep_raw:
            # Preserve each generation's own answer
            for r in rows:
                ans = extract_answer(r['Qwen3-32B'])
                if ans:
                    formatted = f"Based on the analysis, the root cause is: \\boxed{{{ans}}}"
                else:
                    formatted = r['Qwen3-32B']
                submission_rows.append({
                    'ID': r['ID'],
                    'Qwen3-32B': formatted,
                    'Qwen2.5-7B-Instruct': 'placeholder',
                    'Qwen2.5-1.5B-Instruct': 'placeholder',
                })
        else:
            formatted = f"Based on the analysis, the root cause is: \\boxed{{{best_answer}}}"
            for r in rows:
                submission_rows.append({
                    'ID': r['ID'],
                    'Qwen3-32B': formatted,
                    'Qwen2.5-7B-Instruct': 'placeholder',
                    'Qwen2.5-1.5B-Instruct': 'placeholder',
                })

    sub_df = pd.DataFrame(submission_rows)
    return sub_df, vote_stats


def build_submissions(results_df: pd.DataFrame, output_dir: str, timestamp: str):
    """Build majority and plurality vote submissions, log stats."""
    rows_by_qid = _group_by_question(results_df)
    paths = {}

    for label, use_plurality in [('majority', False), ('plurality', True)]:
        sub_df, stats = _build_submission(rows_by_qid, use_plurality=use_plurality)
        path = f"{output_dir}/submission_{label}_{timestamp}.csv"
        sub_df.to_csv(path, index=False)
        paths[label] = path

        logger.info(f"\n{label.upper()} submission: {path} ({len(sub_df)} rows)")
        logger.info(f"  Unanimous: {stats['unanimous']}, Majority(3+): {stats['majority']}, "
                     f"Plurality(2-1-1): {stats['plurality']}, Tie: {stats['tie']}")

        # Answer distribution
        vote_answers = []
        for base_id, rows in rows_by_qid.items():
            answers = [extract_answer(r['Qwen3-32B']) for r in rows]
            answers = [a for a in answers if a]
            if answers:
                vote_answers.append(majority_vote(answers))
        vote_counts = Counter(vote_answers)
        logger.info(f"  Answer distribution ({len(vote_answers)} questions):")
        for ans, cnt in vote_counts.most_common(10):
            logger.info(f"    {ans}: {cnt}")

    return paths


def upload_to_hub(output_dir: str, hf_repo: str, hf_token: str = None):
    """Upload all output files (submissions, checkpoint) to HF Hub dataset repo."""
    from huggingface_hub import HfApi

    api = HfApi(token=hf_token)

    # Create repo if it doesn't exist
    try:
        api.create_repo(repo_id=hf_repo, repo_type="dataset", exist_ok=True, private=True)
    except Exception as e:
        logger.warning(f"Could not create repo {hf_repo}: {e}")

    # Upload all CSV and JSON files in output dir
    uploaded = 0
    for fname in os.listdir(output_dir):
        if fname.endswith('.csv') or fname.endswith('.json'):
            local_path = os.path.join(output_dir, fname)
            try:
                api.upload_file(
                    path_or_fileobj=local_path,
                    path_in_repo=fname,
                    repo_id=hf_repo,
                    repo_type="dataset",
                )
                uploaded += 1
                logger.info(f"  Uploaded: {fname}")
            except Exception as e:
                logger.warning(f"  Failed to upload {fname}: {e}")

    logger.info(f"Uploaded {uploaded} files to https://huggingface.co/datasets/{hf_repo}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="V19 GRPO inference with vLLM (Type A + Type B + Generic prompts)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    parser.add_argument(
        '--model', type=str, required=True,
        help='HuggingFace model ID or local path (merged model)',
    )
    parser.add_argument(
        '--lora', type=str, default=None,
        help='LoRA adapter path (if using base + LoRA)',
    )
    parser.add_argument(
        '--hf-token', type=str, default=None,
        help='HuggingFace token for private models',
    )

    # Data
    parser.add_argument(
        '--test-csv', type=str,
        default='./the-ai-telco-troubleshooting-challenge20251127-8634-8qzscv/phase_2_test.csv',
        help='Path to test CSV',
    )
    parser.add_argument(
        '--output-dir', type=str,
        default='./outputs/inference',
        help='Output directory',
    )

    # Checkpointing
    parser.add_argument(
        '--checkpoint', type=str, default=None,
        help='Checkpoint file path (default: output_dir/checkpoint.json)',
    )
    parser.add_argument(
        '--checkpoint-every', type=int, default=10,
        help='Save checkpoint every N batches',
    )

    # Inference config
    parser.add_argument(
        '--num-gpus', type=int, default=1,
        help='Number of GPUs for tensor parallelism',
    )
    parser.add_argument(
        '--max-model-len', type=int, default=8192,
        help='Maximum context length',
    )
    parser.add_argument(
        '--gpu-memory-utilization', type=float, default=0.95,
        help='GPU memory utilization for vLLM',
    )
    parser.add_argument(
        '--num-generations', type=int, default=4,
        help='Number of generations per question',
    )
    parser.add_argument(
        '--temperature', type=float, default=0.6,
        help='Sampling temperature',
    )
    parser.add_argument(
        '--max-tokens', type=int, default=4096,
        help='Maximum tokens to generate',
    )
    parser.add_argument(
        '--batch-size', type=int, default=32,
        help='Batch size for inference',
    )

    # Upload
    parser.add_argument(
        '--hf-upload', type=str, default=None,
        help='HF dataset repo to upload results (e.g. Phaedrus33/telco-submissions)',
    )

    # Utility
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Run on first 5 samples only',
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("V19 GRPO INFERENCE")
    logger.info("=" * 60)

    num_gpus = setup_gpu()
    if args.num_gpus > num_gpus:
        logger.warning(f"Requested {args.num_gpus} GPUs but only {num_gpus} available")
        args.num_gpus = num_gpus

    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_path = args.checkpoint or f"{args.output_dir}/checkpoint.json"

    # Load test data
    logger.info(f"\nLoading test data: {args.test_csv}")
    test_df = pd.read_csv(args.test_csv)
    logger.info(f"Test samples: {len(test_df)}")

    # Show question type distribution
    type_counts = Counter(get_question_type(q) for q in test_df['question'])
    logger.info(f"Question types: {dict(type_counts)}")

    if args.dry_run:
        test_df = test_df.head(5)
        logger.info("DRY RUN: Using first 5 samples only")

    # Load model
    llm, lora_path = load_model(
        model_path=args.model,
        lora_path=args.lora,
        num_gpus=args.num_gpus,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        hf_token=args.hf_token,
    )

    tokenizer = load_tokenizer(args.model, args.hf_token)

    # Run inference
    results_df = run_inference(
        llm=llm,
        test_df=test_df,
        tokenizer=tokenizer,
        checkpoint_path=checkpoint_path,
        lora_path=lora_path,
        num_generations=args.num_generations,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
        checkpoint_every=args.checkpoint_every,
    )

    # Save raw results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    raw_path = f"{args.output_dir}/submission_raw_{timestamp}.csv"
    results_df.to_csv(raw_path, index=False)
    logger.info(f"Raw submission saved: {raw_path}")

    # Build majority + plurality vote submissions
    vote_paths = build_submissions(results_df, args.output_dir, timestamp)

    # Remap option labels (model trained on C-prefix, test uses varied prefixes)
    logger.info("\nRemapping option labels...")
    remap_submission(raw_path, test_df)
    for label, path in vote_paths.items():
        remap_submission(path, test_df)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("INFERENCE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Questions: {len(test_df)}")
    logger.info(f"Generations per question: {args.num_generations}")
    logger.info(f"Total rows: {len(results_df)}")
    logger.info(f"Raw submission: {raw_path}")
    for label, path in vote_paths.items():
        logger.info(f"{label} submission: {path}")

    # Raw answer distribution
    answers = results_df['Qwen3-32B'].apply(extract_answer)
    answer_counts = answers.value_counts()
    logger.info(f"\nRaw answer distribution:")
    for ans, cnt in answer_counts.head(10).items():
        logger.info(f"  {ans}: {cnt}")

    # Upload to HF Hub
    if args.hf_upload:
        logger.info(f"\nUploading results to HF Hub: {args.hf_upload}")
        upload_to_hub(args.output_dir, args.hf_upload, args.hf_token)


if __name__ == "__main__":
    main()
