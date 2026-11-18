# Grounding and caveats

- Full public v9.1: SAADE_AUA_v9_1_REPEAT_PADDING_FULL_FOR_REVIEW.zip; 876,330 segment-scenario-model scores.
- Hidden v9.2: SAADE_AUA_v9_2_FINAL_HIDDEN_ONCE_FOR_REVIEW.zip; 8,694 fixed-protocol scores.
- All main quantities recomputed from archived completed summary / scenario-metrics tables, or cross-checked against them.
- 4,885 usable canonical recordings is **not** identical to 4,816 public canonical test recordings (69 excluded for hidden-overlap control).
- Inference units repeated within parent audio; do not treat 885,024 scores as independent observations.
- Pair-bootstrap CIs conditional on one voice per validation/hidden split; no inference across independent TTS generators.
- Comparison zero- vs repeat-padding includes threshold recalibration; not a purely isolated padding effect.
- Canonical AUA class and source confounded; report source-conditioned recals only.
- Hugging Face ATV-UA repo https://huggingface.co/datasets/adam-i/ATV-UA is user-supplied public dataset reference. Its README and revision were not available to verify in the present environment; reconcile the public revision/manifest with the local AUA catalogue before manuscript submission.
