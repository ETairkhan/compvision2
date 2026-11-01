# Assignment 2: household objects + synthetic video frames

Classes: Books, Cups, Pens, Spoons (from Assignment 1).

## Files
- assignment2.ipynb: Colab notebook (frames, training, ablation, stress test, Grad-CAM)
- app.py: Gradio app (upload or webcam, prediction, Grad-CAM, latency, environment)
- PROMPTS.md: how to make the Step-Video-T2V clips
- requirements.txt

## How to run
1. Put in Google Drive: compvision1_cropped (A1 dataset), compvision2_videos/<Class>/*.mp4 (see PROMPTS.md).
2. Open assignment2.ipynb in Colab (GPU runtime), run all cells. Results go to MyDrive/compvision2_results.
3. Copy app.py to that folder, run the last cell. It prints a public link.

Local run: pip install -r requirements.txt, then
MODEL_PATH=resnet18_sota.pth python app.py

## Notes
- Val and test sets contain real photos only. Synthetic frames are only in the training set.
- The split is rebuilt with seed 42 and saved as split.csv; the baseline is retrained on the same split.
- mAP here is the macro average precision of the 4 classes (classification), not detection mAP@0.5.
- All generated frames are synthetic and contain no people.
