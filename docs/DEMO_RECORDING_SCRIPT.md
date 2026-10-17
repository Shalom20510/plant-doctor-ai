# Plant Doctor AI — Demo recording script

**Target length:** 2–3 minutes. Keep the browser and app ready before pressing Record. Speak clearly and show only behavior you have actually tested.

## Before recording (not part of the video)

1. Open the live app: https://plant-doctor-ai-gzqqjtpxbjdwp85ur3fnkh.streamlit.app/
2. Refresh and verify it reflects the latest GitHub `main` commit. If the new UI is not live, wait for deployment or use the local app; do not present an unverified deployment as working.
3. Prepare four photos: a supported leaf with a known label, a person/object photo, an unsupported plant leaf (for example, banana or fern), and a blurry/dark leaf. Only use the unsupported sample if you know which plant it is.
4. If you can, run `py -m pytest -q` in the project terminal and keep the successful output ready as a separate evidence shot.
5. Turn off notifications and close unrelated windows.

## Recording sequence

### 0:00–0:15 — Problem and solution

Say: “Plant diseases are difficult to identify from leaf appearance alone. Plant Doctor AI is a prototype that screens a leaf photo, predicts a possible class from its trained categories, and gives general care guidance. It also tries to stop and ask for a clearer or supported image when the input is uncertain.”

Show the app title and upload area.

### 0:15–0:40 — Upload and image checks

Upload the clear supported-leaf photo. Show the photo preview and quality-check section. Select **Auto-detect** first; use the manual plant selector only if you know the plant identity.

Say: “The app checks image quality and uses CLIP to screen whether the image resembles a supported crop. The disease classifier is a separate EfficientNet-B0 model.”

### 0:40–1:15 — Prediction and care guidance

Click **Analyze Leaf**. Let the actual result display. Point to the predicted plant/condition, uncalibrated score category, possible alternative matches, and care guidance.

Say: “The classifier predicts among 38 trained classes across 14 crop groups. The score is not a calibrated probability, and this result is not a confirmed diagnosis. The guidance is educational, not a treatment prescription.”

If the result is inconclusive or a crop mismatch, do not hide it. Explain that this is the safeguard working and use another known supported photo only if needed.

### 1:15–1:50 — Rejection behavior

Upload a person/object photo and analyze it. Show the actual rejection. Then try the unsupported plant or blurry/dark photo and show the actual behavior.

Say: “The app is designed to reject non-leaf, unsupported, or unclear inputs instead of always forcing a disease label. These checks are heuristics, so they can still make mistakes.”

Do not claim every unsupported plant is always rejected. Only state the result that appears on screen.

### 1:50–2:15 — Testing and architecture

Show the GitHub repository and the successful Actions workflow briefly.

Say: “The repository includes setup instructions, a model checkpoint and class mapping, input-validation and prediction helpers, evaluation tooling, and automated tests. The CI test suite passes. A separate independent real-world image benchmark is still needed before making accuracy claims.”

### 2:15–2:40 — Limitations and close

Say: “The current model only covers its trained classes, and performance on controlled leaf images may not generalize to field photos. Future work is to evaluate on an independent field-photo dataset, calibrate rejection thresholds, and expand training data for crops and diseases not represented today.”

End on the app or repository page.

## After recording

- Watch the full recording once and check audio, readability, and that no secrets or personal notifications appear.
- Confirm the video link can be opened by judges without requesting your personal account access.
- Add the video URL to `docs/MISSION_05_SUBMISSION.md` and the event submission form.
- Never add unmeasured accuracy percentages or claim the app is clinically/agronomically validated.
