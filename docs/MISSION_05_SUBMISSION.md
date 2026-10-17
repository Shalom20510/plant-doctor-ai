# Mission 05 — Final submission checklist

Use this checklist immediately before recording and submitting. It distinguishes items verified in GitHub from items that still require a real-device/live-app check. Do not mark a pending item complete without evidence.

## Deliverables

| Requirement | Current state | Evidence / next action |
|---|---|---|
| Public source repository | Present | https://github.com/Shalom20510/plant-doctor-ai |
| README, setup, dependencies | Present | README.md and requirements.txt |
| Model/config/class map | Present | `artifacts/models/model_v1.pt`, `artifacts/reports/class_to_idx.json`, `configs/config.yaml` (reference config; app does not currently load it) |
| Input validation and prediction safeguards | Implemented in code | CLIP scope screen, image-quality checks, crop mismatch checks, low-score rejection, top-two margin rejection |
| Automated tests | **PASS in GitHub Actions** | [Latest successful CI run](https://github.com/Shalom20510/plant-doctor-ai/actions/runs/38993480335) |
| Image-based aggregate evaluation | **PENDING** | No independent labeled field-photo benchmark/results are committed. Do not claim accuracy, rejection rates, or robustness metrics without running and recording evaluation. |
| Sample test cases | Script exists; images not committed | `py scripts/download_eval_samples.py` downloads smoke-test samples. PlantVillage images may overlap training data, so they are not guaranteed independent test examples. |
| Live deployment | Link supplied; current status **NOT VERIFIED** | https://plant-doctor-ai-gzqqjtpxbjdwp85ur3fnkh.streamlit.app/ — open it and verify the latest code before recording. |
| Demo video | **PENDING — next action** | Follow [DEMO_RECORDING_SCRIPT.md](DEMO_RECORDING_SCRIPT.md), upload the recording, then paste its share link into the submission form. |
| AI/tool disclosure | Draft below; confirm team has reviewed it | Include the disclosure in submission notes or the required form. |
| Team understanding / jury readiness | **Team action required** | Every presenter must explain the architecture, limits, contributions, and test evidence without reading claims they cannot defend. |

## AI and technology disclosure

**Application:** Plant Doctor AI, a prototype for screening leaf photos and suggesting a possible health-condition class.

**Models:** EfficientNet-B0 classifier with 38 PlantVillage-style classes; CLIP model `openai/clip-vit-base-patch32` is used for image-scope/crop screening, not as the disease classifier.

**Software/tools:** Python, Streamlit, PyTorch, torchvision, Pillow, Hugging Face Transformers, scikit-learn, pytest, GitHub, and Streamlit Community Cloud deployment.

**Generative AI assistance:** ChatGPT assisted with code drafting, debugging, test planning, and documentation during development. The team must review and understand all submitted code and disclose this assistance according to the event's rules. Do not claim that all code was written manually if that is not true.

**Data:** The checkpoint uses 38 classes based on PlantVillage-style leaf categories. The training dataset is not bundled in the repository. The repository's dataset-preparation script documents the source/licensing information; retain attribution and follow the source license. The optional smoke-test downloader also retrieves images from upstream repositories, including Wikimedia Commons for non-leaf/unsupported examples; verify source terms and retain attribution when redistributing any downloaded images.

**Limitations:** The model only recognizes its trained classes. CLIP and softmax scores are not calibrated probabilities. The crop, low-score, image-quality, and top-two-margin checks are heuristics; they can reject correct images or allow incorrect results. Controlled dataset performance may not transfer to real field photos. This is not a confirmed agricultural diagnosis; important treatment decisions should be checked with a qualified agricultural expert.

## Final pre-recording checklist

- [ ] Open the deployed app in a fresh browser tab; confirm no import error or blank screen.
- [ ] Confirm the latest changes are live: Plant type selector, uncalibrated score label, crop mismatch warning, and inconclusive-result behavior.
- [ ] Run `py -m pytest -q` locally if the environment is available; save the terminal output.
- [ ] Prepare a small set of clear images and know the correct labels before recording.
- [ ] Test one supported leaf, one non-leaf photo, one unsupported leaf, and one blurry/dark photo. Record what actually happens; do not force a claim that every case passes.
- [ ] If a chosen crop triggers a mismatch warning, explain that the safeguard prevented a potentially misleading diagnosis.
- [ ] Close personal tabs, hide secrets/notifications, enlarge the browser text, and record a clean screen.
- [ ] Save the final video and test the share link in an incognito/private window.
- [ ] Submit repository URL, live app URL, video URL, and the required AI disclosure.
- [ ] Keep evidence: CI pass, screenshots of observed demo cases, and any actual evaluation output. Never invent missing metrics.

## Links to include in the submission

- Repository: https://github.com/Shalom20510/plant-doctor-ai
- Live app (verify before submission): https://plant-doctor-ai-gzqqjtpxbjdwp85ur3fnkh.streamlit.app/
- Demo video: **ADD YOUR RECORDED VIDEO LINK HERE**
- CI evidence: https://github.com/Shalom20510/plant-doctor-ai/actions/runs/38993480335
