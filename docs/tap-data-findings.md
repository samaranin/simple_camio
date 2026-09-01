# Findings on the collected tap data

Recorded 2026-09-01 while adding the classifier regression baseline. Neither
item is fixed here: both live in `src/detection/pose_detector.py`, which the
thread-safety plan deliberately leaves alone.

## The dataset is almost entirely positive

241 samples across three sessions on 2025-10-27: **237 positive, 4 negative.**

`models/tap_model.json` is therefore trained with almost no counter-examples
and should be expected to over-predict taps. Any accuracy figure quoted from
this data is close to meaningless — a classifier answering "tap" unconditionally
scores 98.3% on it.

Before retraining, collect negatives deliberately: hovering, pointing without
pressing, dragging a finger across zones, and withdrawing the hand mid-press.

## No sample came from the enhanced detector

Every one of the 241 samples carries `metadata.detector == "base"`, even though
`CombinedPoseDetector.__init__` hands the base and enhanced detectors the same
collector object, and `_collect_enhanced_tap_data_positive` /
`_collect_enhanced_tap_data_negative` exist in
`PoseDetectorMPEnhanced`.

**Open question:** can the enhanced collection path fire at all? If it cannot,
training data for the enhanced detector is uncollectable, and the enhanced
detector's classifier features are exercised only at runtime. Worth confirming
against a live session before any work on the classifier.
