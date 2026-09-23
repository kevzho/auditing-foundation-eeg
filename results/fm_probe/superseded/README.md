# Superseded artifacts

Moved out of the reporting path on 2026-08-15. `make_fm_audit_report.py` globs
the parent directory only, so nothing here contributes to any table.

## fm_probe_finetune_* (LaBraM, un-tagged, p2/p4)

Retired, not deleted. Its run JSON predates the current format and records only
`{mode, patches, seed}` -- no learning rate, epoch count or freeze depth -- so
it cannot be reproduced faithfully without guessing. It also ran on MPS with
`deterministic_algorithms=false`.

Superseded by `labram_finetune_sel`, which uses the validation-selected recipe
from the sweep rather than the reference default. No claim in the manuscript
depends on these rows; the LaBraM conclusion rests on `labram_finetune_sel`,
`labram_frozen_p2`, `labram_frozen_p4` and `labram_loso`.
