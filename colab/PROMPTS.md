# Prompts for Step-Video-T2V (offline generation)

Where: the online version linked in the Step-Video-T2V README, yuewen.cn/videos.
The model itself needs about 72-79 GB GPU memory, so it cannot run on a free Colab T4.
Check on the site if you need an account and what the clip limits are.

Save every clip as mp4 here (about 4 clips per class is enough, 4 frames are taken from each (the 60% center-crop from A1 is applied)):

    MyDrive/compvision2_videos/Books/
    MyDrive/compvision2_videos/Cups/
    MyDrive/compvision2_videos/Pens/
    MyDrive/compvision2_videos/Spoons/

Template: "[camera motion] a [object] [place], [lighting], single object in the center, no people"

Camera motion (change for each clip):
- handheld camera slowly orbits around
- fast pan with motion blur
- camera moves from top-down view to a side view
- slight camera shake, out of focus then in focus

Place and lighting (change for each clip):
- on a wooden table, warm lamp light
- on a plain white desk, bright daylight
- on a floor, dim evening light
- on a kitchen counter, mixed light

Objects: "a closed book", "a ceramic cup", "a ballpoint pen", "a metal spoon".

Example: "Handheld camera slowly orbits around a white ceramic cup on a wooden table, warm lamp light, single object in the center, no people"

Rules:
- No people, hands or faces in the prompts (consent and ethics).
- Look at each clip. Delete clips where the object is wrong or deformed (for example a spoon that looks like a fork).
- Keep the files as they are; the notebook marks every frame as synthetic (syn_ prefix).
