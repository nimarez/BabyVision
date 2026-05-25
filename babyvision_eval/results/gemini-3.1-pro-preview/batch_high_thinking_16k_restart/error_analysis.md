# BabyVision Gemini Error Analysis

- Results: `babyvision_eval/results/gemini-3.1-pro-preview/batch_high_thinking_16k_restart/model_results_run_1.json`
- Model: `gemini-3.1-pro-preview`
- Total: 388
- Correct: 171
- Accuracy: 44.07%

## Weakest Subtypes

- Connect the lines: 0.00% (0/19, failures=19)
- 3D Cube Unfold: 8.33% (1/12, failures=11)
- 3D Pattern Completion: 11.11% (2/18, failures=16)
- Find the same: 17.65% (3/17, failures=14)
- Maze: 20.00% (4/20, failures=16)
- Count 3D blocks: 27.27% (6/22, failures=16)
- Find the shadow: 30.43% (7/23, failures=16)
- Overlay Patterns: 35.29% (6/17, failures=11)
- Pattern and Color Completion: 40.00% (8/20, failures=12)
- Rotation Patterns: 40.00% (4/10, failures=6)
- 3D Views: 44.44% (12/27, failures=15)
- 2D Pattern Completion: 45.00% (9/20, failures=11)

## Weakest Dimensions

- 3d_spatial: 31.18% (29/93, failures=64)
- path_tracking: 31.58% (24/76, failures=52)
- coordinate_search: 38.71% (24/62, failures=38)
- rotation_or_shadow: 43.18% (19/44, failures=25)
- pattern_completion: 44.16% (102/231, failures=129)
- fine_detail: 46.08% (94/204, failures=110)
- counting: 53.04% (61/115, failures=54)
- overlay_reconstruction: 54.84% (17/31, failures=14)

## Synthetic Data Targets

- 1. Connect the lines: 0.00% accuracy over 19 tasks. Prompt focus: path_tracking, fine_detail, pattern_completion
- 2. 3D Cube Unfold: 8.33% accuracy over 12 tasks. Prompt focus: 3d_spatial, pattern_completion, fine_detail
- 3. 3D Pattern Completion: 11.11% accuracy over 18 tasks. Prompt focus: 3d_spatial, pattern_completion, fine_detail
- 4. Find the same: 17.65% accuracy over 17 tasks. Prompt focus: fine_detail, pattern_completion, coordinate_search
- 5. Maze: 20.00% accuracy over 20 tasks. Prompt focus: path_tracking, coordinate_search, counting
- 6. Count 3D blocks: 27.27% accuracy over 22 tasks. Prompt focus: counting, 3d_spatial, fine_detail
- 7. Find the shadow: 30.43% accuracy over 23 tasks. Prompt focus: pattern_completion, rotation_or_shadow, coordinate_search
- 8. Overlay Patterns: 35.29% accuracy over 17 tasks. Prompt focus: pattern_completion, overlay_reconstruction, coordinate_search
