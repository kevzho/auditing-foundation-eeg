# BNCI Negative-Transfer Diagnostic

Conclusions are descriptive unless paired statistics are added separately.

## Subjects Dominating Losses

- BNCI2014_001 subject 7, Teacher/student: delta Brier +0.1282, delta accuracy -0.1632, delta ECE -0.0101.
- BNCI2014_001 subject 7, Portfolio: delta Brier +0.1282, delta accuracy -0.1632, delta ECE -0.0101.
- BNCI2014_004 subject 5, Portfolio: delta Brier +0.0736, delta accuracy -0.0531, delta ECE +0.0336.
- BNCI2014_004 subject 5, Augmentation: delta Brier +0.0736, delta accuracy -0.0531, delta ECE +0.0336.
- BNCI2014_004 subject 5, Seed ensemble: delta Brier +0.0521, delta accuracy -0.0344, delta ECE +0.0373.
- BNCI2014_001 subject 7, Seed ensemble: delta Brier +0.0507, delta accuracy -0.0590, delta ECE -0.0054.

## Failure Mode Summary

- BNCI2014_001 Augmentation: accuracy, Brier/calibration, ECE; mean validation-to-heldout Brier rank shift +0.22; mean confidence minus accuracy -0.007.
- BNCI2014_001 Seed ensemble: accuracy, Brier/calibration; mean validation-to-heldout Brier rank shift -1.11; mean confidence minus accuracy -0.003.
- BNCI2014_001 Teacher/student: accuracy, Brier/calibration, ECE; mean validation-to-heldout Brier rank shift +1.00; mean confidence minus accuracy -0.043.
- BNCI2014_001 Portfolio: accuracy, Brier/calibration; mean validation-to-heldout Brier rank shift +1.22; mean confidence minus accuracy +0.006.
- BNCI2014_004 Augmentation: accuracy, Brier/calibration, ECE; mean validation-to-heldout Brier rank shift +0.78; mean confidence minus accuracy +0.024.
- BNCI2014_004 Seed ensemble: accuracy, Brier/calibration; mean validation-to-heldout Brier rank shift -1.00; mean confidence minus accuracy +0.017.
- BNCI2014_004 Teacher/student: no mean loss versus baseline; mean validation-to-heldout Brier rank shift -0.33; mean confidence minus accuracy +0.007.
- BNCI2014_004 Portfolio: accuracy; mean validation-to-heldout Brier rank shift +1.22; mean confidence minus accuracy +0.023.

## Classical Anchor / MDRM-T

MDRM-T + EA rows are not present for the retained BNCI runs, so no classical EA anchor conclusion is supported yet.

## Selection Mismatch

- BNCI2014_001 Portfolio: mean heldout-minus-validation Brier rank shift +1.22.
- BNCI2014_004 Portfolio: mean heldout-minus-validation Brier rank shift +1.22.
- BNCI2014_001 Teacher/student: mean heldout-minus-validation Brier rank shift +1.00.
- BNCI2014_004 Augmentation: mean heldout-minus-validation Brier rank shift +0.78.
