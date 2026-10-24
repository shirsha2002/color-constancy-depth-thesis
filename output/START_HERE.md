# Your first thesis experiment

Shirsha Mitra · Student ID 2023521460130

Your thesis asks whether correcting image color can help a monocular depth model
estimate distance more accurately. This project implements a first controlled
test. It uses an existing pretrained model and real RGB/depth data. It does not
train a new network.

## Read these files in order

1. Open `pdf/Shirsha_Mitra_Thesis_Progress_Report.pdf` for the explanation,
   methods, measured results and selected examples.
2. Open `Shirsha_Mitra_Thesis_Progress.pptx` for the 12-slide professor presentation.
   `Shirsha_Mitra_Thesis_Progress_Slides.pdf` is the portable slide copy.
3. Read `Shirsha_Mitra_Presentation_Notes.txt` aloud while reviewing the slides.
   These notes explain each slide and provide its sources.
4. Read `QQ_Update_English.txt`. Check that you understand every numerical claim
   before sharing the message and presentation with your supervisor.

## What you should be able to explain

**The question.** The objects stay in the same place. Does changing their image's
color affect estimated distance? Can a correction method reduce that error?

**The baseline.** The official Depth Anything V2 Small indoor metric model predicts
a numerical depth value for each pixel. The model receives a single RGB image.
Its output is an estimate in meters.

**The data.** We use NYU Depth V2 photographs with aligned Kinect-derived reference
depth. Eight development images and 32 evaluation images come from separate
scenes in the official train/test split. Missing depth values have been filled by
the original dataset authors. The reference is not a prediction from our model.

**The comparisons.** Every image has 7 conditions: original, warm/cool/green casts
at mild and strong levels. Each condition has 4 processing choices: no correction,
Gray-World, White-Patch and half-strength Gray-World. The full run therefore
contains 40 × 7 × 4 = 1,120 predictions, including 896 evaluation predictions.

**The metrics.** AbsRel expresses absolute error relative to reference distance.
RMSE gives more emphasis to large distance errors and has units of meters.
Delta1 is the fraction of valid pixels whose prediction/reference ratio is within
a factor of 1.25. Lower AbsRel/RMSE and higher Delta1 indicate better predictions.

Illustration only: a prediction of 2.2 m for a reference depth of 2.0 m has
0.2 m absolute error and 0.1 relative error (10%). The experiment uses its actual
reference depths and predictions to calculate the reported averages.

**The fair comparison.** Every method uses the same evaluation images, reference
depth, crop and valid pixels. No fitted scale makes an individual prediction
closer to its reference. Methods and cast gains stay fixed throughout the test.

**The main limitation.** These color casts are synthetic transformations of real
images. They do not recreate every real light source, shadow, mixed illumination
or camera-processing effect. Results from 32 scenes and one depth model are a
pilot. A lower depth error does not by itself prove physically accurate color.

## Questions your professor may ask

- Why did you choose a metric-depth checkpoint? Its output already has meter
  units, allowing direct comparison without fitting a separate depth scale.
- Did you train the depth model? No. We froze the pretrained model and evaluated
  controlled changes to its input.
- What is your own work here? The reproducible experiment pipeline, the paired
  evaluation, correction implementations, strength ablation and analysis. A new
  algorithm contribution remains future work.
- Does correction always help? Use the measured clean-input table and failure
  example to answer. Do not promise that any method helps every scene.
- Why show a failure? It identifies where an illuminant assumption or clipping
  may matter and helps choose the next controlled experiment.
- What next? Test more scenes and a second depth model; isolate clipping effects;
  study real lighting changes; investigate the supervisor's Gray Anchoring paper.

## Reopen and rerun

Open `../ColorConstancyDepth.code-workspace` in VS Code. Select **Terminal → Run
Task**, then choose a task beginning **Thesis:**. The environment and scientific
checks are quick. The full experiment reuses completed predictions when their
configuration fingerprint matches.

To predict your own indoor photograph, run this from the project directory:

```powershell
& .\.venv\Scripts\python.exe scripts/predict_image.py --image path\to\room.jpg --method gray_world_half
```

Look under `results/demo/` for the estimated depth array and comparison image.
You can also press **F5** in VS Code, choose **Estimate depth of an image**, and
enter a photograph path and correction method.

Accuracy requires reference depth, so a personal photograph provides a visual
demonstration rather than a numerical accuracy result.

The GitHub repository is private:
https://github.com/shirsha2002/color-constancy-depth-thesis

Code, configuration, sample IDs, metrics and final materials are versioned.
Credentials, model weights, source data, environments and large prediction caches
stay out of the repository.
