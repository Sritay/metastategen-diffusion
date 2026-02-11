# Release Notes

## v0.2.1 (2026-02-11)

### Features
*   **Optional Active Learning**: Decoupled the training workflow. Users can now run "Normal Training" (Single Model) or the full "Active Learning Loop" independently.
*   **Generalized Training**: The `msgen train` command now supports generalized input data (`.npz` positions + `.pdb` topology), removing the hard dependency on the specific Alanine Dipeptide dataset structure.
*   **Documentation**: Updated usage guides to clarify the distinction between Standard and Active Learning workflows.

### Fixes
*   Refactored `active_learning.py` and `train.py` to share common logic via `workflows/common.py`.
