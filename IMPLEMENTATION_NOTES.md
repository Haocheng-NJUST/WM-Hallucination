# Implementation Notes

The package provides paired generation, final-text detector retokenization, context-only fact-span extraction, FPTI/FPAI interventions, and empirical matched TPR calibration.

Watermark configurations are serialized in every generation row. `generated_token_ids` are returned by the generation loop or `model.generate` for audit, while `detector_token_ids` are created by re-encoding the final decoded text with the detector tokenizer. `target_facts` remains an evaluation/output field and is not used by FPTI or FPAI.

DiPmark uses the supplied alpha-reweighting implementation. Unbiased Watermark uses the supplied delta-reweighting strategy and reconstructs `p_t/q_t` from the generation prompt plus final-text completion IDs. TextSeal uses the supplied dual-key Gumbel-max PRF, and MorphMark uses the supplied adaptive green-mass reweighting. Source revisions are recorded in each method configuration and in the README.
