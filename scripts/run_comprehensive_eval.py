"""Comprehensive evaluation script covering Test 1, Test 2, and Test 3
for Plant Doctor AI.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, f1_score

# Ensure root directory is on sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data import IMAGE_SIZE
from src.input_validation import (
    CROP_PROMPT_VARIANTS,
    NON_LEAF_PROMPTS,
    SCOPE_THRESHOLD,
    SUPPORTED_CROP_NAMES,
    SUPPORTED_PROMPTS,
    UNSUPPORTED_LEAF_PROMPTS,
    aggregate_scope_logits,
    crop_name_from_prompt,
    decide_input_scope,
    normalize_crop_name,
)
from src.model import load_model
from src.predict import get_inference_transform, load_class_mapping, predict_topk
from src.robustness import synthetic_clutter, variants
from app.streamlit_app import image_quality_checks, validate_plant_image, readable_name

CHECKPOINT_PATH = ROOT_DIR / "artifacts" / "models" / "model_v1.pt"
CLASS_MAP_PATH = ROOT_DIR / "artifacts" / "reports" / "class_to_idx.json"
TEST_DATA_DIR = ROOT_DIR / "artifacts" / "evaluation" / "test_data"
RESULTS_OUTPUT_PATH = ROOT_DIR / "artifacts" / "evaluation" / "comprehensive_eval_results.json"


def evaluate_test_1(model, idx_to_class, class_to_idx, transform, device):
    """Test 1: Supported Held-Out Leaf Image Evaluation & Metric Reporting."""
    print("\n" + "="*70)
    print("RUNNING TEST 1: SUPPORTED HELD-OUT LEAF IMAGE EVALUATION")
    print("="*70)

    supported_dir = TEST_DATA_DIR / "supported"
    all_targets = []
    all_preds = []
    sample_records = []

    for class_folder in sorted(supported_dir.iterdir()):
        if not class_folder.is_dir():
            continue
        class_name = class_folder.name
        if class_name not in class_to_idx:
            continue
        
        target_idx = class_to_idx[class_name]
        image_files = sorted(list(class_folder.glob("*.jpg")) + list(class_folder.glob("*.JPG")) + list(class_folder.glob("*.png")))

        for img_path in image_files:
            image = Image.open(img_path).convert("RGB")
            img_tensor = transform(image).unsqueeze(0).to(device)

            top_preds = predict_topk(
                model=model,
                image_tensor=img_tensor,
                idx_to_class=idx_to_class,
                device=device,
                top_k=5,
            )

            pred_class, confidence = top_preds[0]
            pred_idx = class_to_idx[pred_class]

            all_targets.append(target_idx)
            all_preds.append(pred_idx)

            is_correct = (pred_class == class_name)
            uncertainty_status = "High Confidence" if confidence >= 0.60 else "Uncertain (< 0.60)"

            sample_records.append({
                "file": img_path.name,
                "ground_truth": class_name,
                "predicted": pred_class,
                "confidence": round(confidence, 4),
                "uncertainty_status": uncertainty_status,
                "correct": is_correct,
                "top_predictions": [
                    {"class": c, "confidence": round(s, 4)} for c, s in top_preds[:3]
                ],
            })

    # Metrics computation
    acc = accuracy_score(all_targets, all_preds)
    bal_acc = balanced_accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)

    unique_targets = sorted(list(set(all_targets)))
    target_names = [idx_to_class[i] for i in unique_targets]
    report_dict = classification_report(
        all_targets,
        all_preds,
        labels=unique_targets,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )

    print(f"Total Held-Out Samples Tested: {len(all_targets)}")
    print(f"Overall Accuracy:            {acc * 100:.2f}%")
    print(f"Balanced Accuracy:           {bal_acc * 100:.2f}%")
    print(f"Macro-F1 Score:              {macro_f1 * 100:.2f}%")

    test_1_pass = (acc >= 0.80 and macro_f1 >= 0.75)
    print(f"Test 1 Status:               {'PASS' if test_1_pass else 'FAIL'}")

    return {
        "status": "PASS" if test_1_pass else "FAIL",
        "sample_count": len(all_targets),
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "classification_report": report_dict,
        "sample_records": sample_records,
    }


def evaluate_test_2(model, idx_to_class, class_to_idx, transform, device, seed=42):
    """Test 2: Changed Image Conditions (Robustness Stress Testing)."""
    print("\n" + "="*70)
    print("RUNNING TEST 2: CHANGED IMAGE CONDITIONS & ROBUSTNESS TESTING")
    print("="*70)

    supported_dir = TEST_DATA_DIR / "supported"
    image_paths = []
    labels = []

    for class_folder in sorted(supported_dir.iterdir()):
        if not class_folder.is_dir() or class_folder.name not in class_to_idx:
            continue
        for img_path in sorted(list(class_folder.glob("*.jpg")) + list(class_folder.glob("*.JPG"))):
            image_paths.append(img_path)
            labels.append(class_to_idx[class_folder.name])

    condition_results = {}
    condition_names = [
        "original",
        "brightness_dim_0.55",
        "brightness_bright_1.45",
        "rotation_15_degrees",
        "rotation_30_degrees",
        "gaussian_blur_1.5",
        "synthetic_edge_clutter",
    ]

    for cond in condition_names:
        preds = []
        confidences = []

        for i, img_path in enumerate(image_paths):
            raw_img = Image.open(img_path).convert("RGB")
            all_variants = variants(raw_img, seed=seed + i)
            var_img = all_variants[cond]

            img_tensor = transform(var_img).unsqueeze(0).to(device)
            top_preds = predict_topk(
                model=model,
                image_tensor=img_tensor,
                idx_to_class=idx_to_class,
                device=device,
                top_k=1,
            )
            pred_class, conf = top_preds[0]
            preds.append(class_to_idx[pred_class])
            confidences.append(conf)

        acc = accuracy_score(labels, preds)
        bal_acc = balanced_accuracy_score(labels, preds)
        macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
        mean_conf = float(np.mean(confidences))

        condition_results[cond] = {
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
            "macro_f1": macro_f1,
            "mean_confidence": mean_conf,
        }
        print(f"Condition: {cond:<25} | Acc: {acc*100:6.2f}% | Macro-F1: {macro_f1*100:6.2f}% | Mean Conf: {mean_conf*100:5.2f}%")

    orig_acc = condition_results["original"]["accuracy"]
    orig_f1 = condition_results["original"]["macro_f1"]

    degradations = {}
    for cond, metrics in condition_results.items():
        if cond == "original":
            continue
        degradations[cond] = {
            "accuracy_drop": round(orig_acc - metrics["accuracy"], 4),
            "macro_f1_drop": round(orig_f1 - metrics["macro_f1"], 4),
            "relative_f1_retention": round(metrics["macro_f1"] / (orig_f1 + 1e-9), 4),
        }

    # Test 2 pass if model retains at least 50% performance under moderate perturbations
    avg_f1_retention = np.mean([d["relative_f1_retention"] for d in degradations.values()])
    test_2_pass = (avg_f1_retention >= 0.50)
    print(f"\nAverage Robustness Retention: {avg_f1_retention*100:.2f}%")
    print(f"Test 2 Status:                 {'PASS' if test_2_pass else 'FAIL'}")

    return {
        "status": "PASS" if test_2_pass else "FAIL",
        "condition_metrics": condition_results,
        "performance_degradations": degradations,
        "avg_retention": float(avg_f1_retention),
    }


def evaluate_test_3(model, idx_to_class, transform, device):
    """Test 3: Unsupported or Ambiguous Input & Rejection Rule Verification."""
    print("\n" + "="*70)
    print("RUNNING TEST 3: UNSUPPORTED OR AMBIGUOUS INPUT & REJECTION RULES")
    print("="*70)

    test_cases = []

    # 1. Non-leaf images
    non_leaf_dir = TEST_DATA_DIR / "non_leaf"
    for img_path in sorted([p for p in non_leaf_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")]):
        test_cases.append({
            "category": "Non-Leaf Image",
            "file": img_path.name,
            "path": img_path,
            "expected_rejection": True,
            "expected_status": "non_leaf",
            "transform_func": None,
        })

    # 2. Unsupported plant classes
    unsupported_dir = TEST_DATA_DIR / "unsupported_plants"
    for img_path in sorted([p for p in unsupported_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")]):
        test_cases.append({
            "category": "Unsupported Plant",
            "file": img_path.name,
            "path": img_path,
            "expected_rejection": True,
            "expected_status": "unsupported_plant",
            "transform_func": None,
        })

    # 3. Heavily blurred / ambiguous images (applied to a valid leaf photo)
    supported_sample = list((TEST_DATA_DIR / "supported").glob("**/*.jpg"))[0]
    
    # Heavy blur case (sigma = 12.0)
    test_cases.append({
        "category": "Ambiguous / Severe Blur",
        "file": f"severe_blur_{supported_sample.name}",
        "path": supported_sample,
        "expected_rejection": True,
        "expected_status": "rejection_or_quality_warning",
        "transform_func": lambda im: im.filter(ImageFilter.GaussianBlur(radius=12.0)),
    })

    # Extreme dark underexposure case (brightness = 0.08)
    test_cases.append({
        "category": "Ambiguous / Extreme Darkness",
        "file": f"extreme_dark_{supported_sample.name}",
        "path": supported_sample,
        "expected_rejection": True,
        "expected_status": "rejection_or_quality_warning",
        "transform_func": lambda im: ImageEnhance.Brightness(im).enhance(0.08),
    })

    results = []
    correct_rejections = 0

    for tc in test_cases:
        img = Image.open(tc["path"]).convert("RGB")
        if tc["transform_func"]:
            img = tc["transform_func"](img)

        # 1. Image Quality Checks
        quality_warnings = image_quality_checks(img)
        quality_rejected = False
        quality_reason = ""
        severe_terms = ("low resolution", "very dark", "overexposed", "blurry or lack detail")
        for w in quality_warnings:
            if any(term in w.lower() for term in severe_terms):
                quality_rejected = True
                quality_reason = w
                break

        # 2. CLIP Input-Scope Screening
        try:
            best_desc, best_score, supported_score = validate_plant_image(img)
            scope = decide_input_scope(best_desc, supported_score, best_score)
        except Exception as e:
            best_desc = "error"
            best_score = 0.0
            supported_score = 0.0
            scope = {"accepted": False, "status": "error", "reason": str(e)}

        # 3. Crop Consistency & Confidence Check (if accepted by scope and quality)
        disease_predicted = False
        classifier_top_label = None
        classifier_confidence = None
        rejection_reason = None
        system_decision = None

        if quality_rejected:
            system_decision = "REJECTED_QUALITY"
            rejection_reason = f"Image Quality Guardrail Triggered: {quality_reason}"
        elif not scope["accepted"]:
            system_decision = f"REJECTED_{scope['status'].upper()}"
            rejection_reason = scope["reason"]
        else:
            # Scope accepted, run disease model
            img_tensor = transform(img).unsqueeze(0).to(device)
            top_preds = predict_topk(
                model=model,
                image_tensor=img_tensor,
                idx_to_class=idx_to_class,
                device=device,
                top_k=5,
            )
            classifier_top_label, classifier_confidence = top_preds[0]
            plant, _, _ = readable_name(classifier_top_label)

            clip_crop = crop_name_from_prompt(best_desc)
            if clip_crop and normalize_crop_name(clip_crop) != normalize_crop_name(plant):
                system_decision = "REJECTED_CROP_MISMATCH"
                rejection_reason = f"Crop Mismatch: CLIP screened '{clip_crop}', classifier predicted '{plant}'"
            elif classifier_confidence < 0.60:
                system_decision = "REJECTED_LOW_CONFIDENCE"
                rejection_reason = f"Inconclusive: Model confidence {classifier_confidence:.1%} < 60%"
            else:
                disease_predicted = True
                system_decision = "ACCEPTED_DIAGNOSIS"

        # Check whether rejection rule successfully blocked invalid diagnosis
        rule_passed = (system_decision != "ACCEPTED_DIAGNOSIS")
        if rule_passed:
            correct_rejections += 1

        res_entry = {
            "category": tc["category"],
            "file": tc["file"],
            "expected_rejection": tc["expected_rejection"],
            "system_decision": system_decision,
            "rejection_reason": rejection_reason,
            "scope_status": scope.get("status"),
            "winning_scope_score": round(best_score, 4),
            "supported_crop_score": round(supported_score, 4),
            "quality_warnings": quality_warnings,
            "rule_passed": rule_passed,
        }
        results.append(res_entry)
        status_flag = "PASS" if rule_passed else "FAIL"
        print(f"[{status_flag}] {tc['category']:<28} | File: {tc['file']:<25} | Decision: {system_decision}")

    rejection_rate = correct_rejections / len(test_cases)
    print(f"\nTotal Invalid/Ambiguous Inputs: {len(test_cases)}")
    print(f"Properly Blocked / Flagged:      {correct_rejections} ({rejection_rate*100:.1f}%)")
    test_3_pass = (rejection_rate >= 0.85)
    print(f"Test 3 Status:                   {'PASS' if test_3_pass else 'FAIL'}")

    return {
        "status": "PASS" if test_3_pass else "FAIL",
        "total_cases": len(test_cases),
        "correct_rejections": correct_rejections,
        "rejection_rate": rejection_rate,
        "detailed_results": results,
    }


def main():
    print("Initializing Plant Doctor AI Comprehensive Evaluation Suite...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    class_to_idx, idx_to_class = load_class_mapping(CLASS_MAP_PATH)
    model, class_names, model_device = load_model(CHECKPOINT_PATH, device=device)
    transform = get_inference_transform(img_size=IMAGE_SIZE)

    # 1. Run Test 1
    t1_results = evaluate_test_1(model, idx_to_class, class_to_idx, transform, device)

    # 2. Run Test 2
    t2_results = evaluate_test_2(model, idx_to_class, class_to_idx, transform, device)

    # 3. Run Test 3
    t3_results = evaluate_test_3(model, idx_to_class, transform, device)

    summary = {
        "test_1_supported_heldout": t1_results,
        "test_2_changed_conditions_robustness": t2_results,
        "test_3_unsupported_and_ambiguous_rejection": t3_results,
        "overall_verdict": "PASS" if (t1_results["status"] == "PASS" and t2_results["status"] == "PASS" and t3_results["status"] == "PASS") else "FAIL",
    }

    RESULTS_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*70)
    print(f"EVALUATION COMPLETE. Overall Verdict: {summary['overall_verdict']}")
    print(f"Results saved to: {RESULTS_OUTPUT_PATH}")
    print("="*70)


if __name__ == "__main__":
    main()
