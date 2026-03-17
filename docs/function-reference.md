# Function Reference

This document is the canonical natural-language reference for functions and methods in this repository.

Use it when:

- explaining what a function does
- checking intended inputs and outputs before editing
- keeping experiment and watermark behavior understandable without reading every implementation detail

Keep entries aligned with code. If code and this document diverge, update the document in the same change.

## `experiments/run.py`

### `run_all_experiments(config)`

Loads the experiment module named by `config["name"]`, reads the parameter sweep from `config["parameters"]`, merges each parameter set with `config["common"]`, and calls the module-level `run(...)` function once per merged config.

This is the central dispatcher for config-driven experiment execution. It assumes each runner module follows the existing `experiments.runs.<name>.run(config)` convention.

It also ensures the default artifact root `artifacts/output/` exists before runner execution.

### `main(argv=None)`

CLI entrypoint for `python -m experiments.run`. It requires `--config <path>` and forwards the loaded YAML config to `run_all_experiments(...)`.

## `experiments/utils/config.py`

### `load_config(experiment=None, path=None)`

Loads a YAML config into a Python dictionary. If `path` is provided, it loads that exact file. Otherwise it resolves `experiment` to `experiments/configs/<experiment>.yaml`.

This is the shared config loader used by both CLI entrypoints and notebooks.

## `experiments/utils/runtime.py`

### `apply_seed(seed)`

Applies a seed consistently across Python, NumPy, and PyTorch. It also sets deterministic CuDNN behavior when CUDA is available.

Use this before any sampling-heavy experiment logic so experiment results are reproducible across runs.

### `resolve_device(device_index)`

Resolves the configured device string or index into a `torch.device`, preferring CUDA when available, and prints the device chosen for the run.

### `_load_pretrained_model_and_tokenizer(model_class, model_name, device, dtype, log_prefix)`

Shared internal loader for Hugging Face model/tokenizer pairs. It loads the tokenizer, instantiates the requested model class with the chosen dtype, moves the model to the target device, and prints a short label.

### `load_model_and_tokenizer(model_name, device)`

Loads a causal language model and tokenizer from Hugging Face, moves the model to the requested device, and normalizes generation config to avoid sampling parameters that conflict with some watermarking schemes.

This is the standard loader for experiments that generate continuations with decoder-only models.

### `load_seq2seq_model_and_tokenizer(model_name, device)`

Loads a sequence-to-sequence translation model and tokenizer, chooses a dtype based on the device, moves the model to the target device, and switches it to eval mode.

This is used by translation and round-trip attack experiments.

### `build_runtime(config, dataset_max_length=50)`

Shared runtime bootstrap for generation experiments. It applies the seed, resolves the target device, loads the causal language model and tokenizer, loads the cached C4 prompts, and returns `(device, model, tokenizer, data)`.

## `experiments/utils/data.py`

### `translate_texts(texts, model, tokenizer, device, batch_size=8, max_source_length=256, max_target_length=256)`

Translates a list of texts in batches using a seq2seq model. Inputs are tokenized with padding and truncation, generation is run under `torch.no_grad()`, and decoded strings are returned in the original order.

### `GeneratedSample`

Small container for a successfully generated example. It stores the prompt token ids, continuation token ids, and decoded continuation text so callers do not have to rebuild those views separately.

### `generate_valid_sample(model, tokenizer, inputs, max_length, device, wat=None, max_attempts=10)`

Shared generation helper that retries up to `max_attempts` times until a continuation reaches the requested length. It returns a `GeneratedSample` when successful and `None` when no valid sample is found.

### `save_text_results(rows, output_path)`

Saves structured text outputs to disk. The input is a list of row dictionaries. The serialization format is chosen from the output path suffix:

- `.parquet` for Parquet
- `.jsonl` for JSON Lines
- anything else for CSV

The function creates the parent directory if needed and returns the resolved output path object.

### `count_conditional_ngrams(token_sequences, order)`

Counts empirical next-token transitions for a token-level `order`-gram estimator. It returns per-context counts, per-context next-token counters, and the total number of usable transitions.

The function treats each token sequence independently and does not create contexts across sequence boundaries.

### `estimate_conditional_entropy(token_sequences, order)`

Estimates the empirical conditional entropy \hat h_k = H(Y_t \mid Y_{t-k:t-1}) from token sequences using base-2 logarithms, so the returned entropy is measured in bits.

It raises a clear error if the provided sequences are too short to supply any transitions for the requested order.

### `load_c4(tokenizer, dataset_size, max_length=50)`

Loads and caches a tokenized subset of the C4 `realnewslike` training split under `artifacts/cache/`. If the cache for the tokenizer, dataset size, and max length exists, it reuses it. Otherwise it streams examples, filters for long enough texts, tokenizes them, and pickles the result.

This is the default text source for most generation experiments in the repo.

## `experiments/utils/watermarks.py`

### `load_default_dip_watermarker(vocab_size, device, seed=42)`

Constructs a `DipWatermarker` with the repo’s default DiP config for the given vocabulary size and device. The seed is used to derive the deterministic hash key that drives DiP’s context-dependent permutation.

### `load_default_kgw_watermarker(vocab_size, device, seed=42)`

Constructs a `KGWWatermarker` with the repo’s default KGW hyperparameters and a seeded device-aware generator.

### `load_default_unbiased_watermarker(vocab_size, device, seed=42)`

Constructs an `UnbiasedWatermarker` with the repo’s default unbiased watermark config for the given vocabulary size and device.

### `load_default_exp_watermarker(vocab_size, device, lam=256, seed=42)`

Constructs an `ExpWatermarker` with the repo’s default EXP hyperparameters, including the default block size and seeded generator.

### `load_default_wepa_watermarker(vocab_size, device, lam=256, degree=1, bits=None, seed=42)`

Constructs a `WepaWatermarker` with the standard defaults used across the paper experiments, optionally changing the automaton degree, bit precision, or sequence length parameter `lam`.

### `load_unwatermarked_watermarker()`

Returns a lightweight wrapper object that exposes a `generate(...)` method but performs ordinary model sampling without applying a watermark.

This is used as a baseline in experiments such as perplexity comparisons.

### `get_wat_name(wat)`

Maps a watermarker instance to the short experiment name used in logs, output names, and WandB runs. It includes extra suffixes for WEPA degree, bit precision, and non-default `lam` values.

### `UnwatermarkedWatermarker.generate(model, inputs, max_new_tokens)`

Generates a continuation from the base language model without watermarking. It simply forwards to `model.generate(...)` with `do_sample=True`.

## `experiments/utils/metrics.py`

### `calc_roc_auc_score(pos_scores, neg_scores)`

Computes ROC AUC from separate positive and negative score lists. It returns `max(roc_auc, 1 - roc_auc)` so the metric remains orientation-agnostic if a detector score is inverted relative to expectation.

### `calc_tpr_at_fpr(pos_scores, neg_scores, fpr_threshold=0.01)`

Builds an ROC curve from positive and negative scores and returns the largest true positive rate observed while the false positive rate stays at or below the requested threshold.

## `experiments/utils/logging.py`

### `init_wandb(project, name, config)`

Thin wrapper around `wandb.init(...)`. It creates and returns a WandB run using the provided project name, run name, and config dictionary, and writes local WandB files under `artifacts/wandb/`.

### `log_wandb_run(experiment, run_name, wandb_config, metrics, tables=None)`

Shared WandB lifecycle helper. It opens a run, logs scalar metrics and optional tables, closes the run, and prints a short summary of the logged config and metrics.

## `experiments/utils/execution.py`

### `run_experiment(experiment, run_name, wat, model, tokenizer, data, max_length, device, evaluate_scores=True, upperbound=False)`

Shared evaluation loop for several sweep-style experiments. It:

- generates watermarked continuations
- computes p-values for each continuation
- optionally computes detector scores for positive and negative samples
- derives summary metrics such as p-value statistics, ROC AUC, and TPR at fixed FPR
- logs the results to WandB tables and scalars

This function centralizes the repeated evaluation pattern used by varying-length, varying-bits, and varying-lambda experiments.

## `experiments/utils/__init__.py`

Compatibility package surface. It preserves the old `experiments.utils` import path by re-exporting implementations from the focused modules above.

## `experiments/runs/diversity.py`

### `_build_watermarkers(model, device, seed)`

Builds the default generator set used by the diversity experiment, including WEPA with `degree=2` and `bits=6`, plus EXP, KGW, unbiased, and the unwatermarked baseline.

### `_resolve_watermarker_names(watermarkers, configured_names=None)`

Resolves the subset of diversity generators to evaluate. It preserves config order, removes duplicates, and raises an explicit error for unsupported names.

### `_generate_prompt_samples(model, tokenizer, prompt, device, wat, max_length, samples_per_prompt)`

Generates multiple valid continuations for one fixed prompt and returns both continuation token sequences and decoded texts.

### `_save_sample_metrics(output_dir, model_name, wat_name, order, rows)`

Saves per-sample entropy estimates for one `(model, generator, k)` combination as a CSV artifact.

### `_select_longest_prompts(data, selected_prompt_count, candidate_pool_size)`

Selects the longest tokenized prompts from the first `candidate_pool_size` loaded prompt candidates. Ties are broken by original prompt order.

### `run(config)`

Runs the diversity experiment by generating many continuations per fixed prompt, estimating empirical token-level conditional entropies for each requested `k`, and logging both aggregate metrics and per-sample tables.

The runner uses prompt-local counting only: contexts are never formed across continuation boundaries or across prompts. It can also over-sample an initial prompt pool and then keep only the longest prompt subset via `prompt_candidate_pool_size`.

## `experiments/runs/corruption.py`

### `run(config)`

Runs corruption-robustness experiments for a chosen corruption type such as substitution, deletion, or insertion. It loads the model and prompts, constructs several watermarkers, applies each corruption level to generated sequences, evaluates p-values and detector statistics, and logs metrics per corruption fraction.

This runner includes corruption-specific negative sample generation rather than using the generic shared experiment loop. The corruption-fraction sweeps are config-driven via `common.corruption_fractions`.

## `experiments/runs/efficiency.py`

### `run(config)`

Measures detection-time efficiency for several watermark schemes. It repeatedly evaluates p-values on cached token sequences, times the runs, and reports elapsed time for optimized and unoptimized detector paths.

The goal is runtime comparison rather than quality or robustness evaluation.

## `experiments/runs/ppl.py`

### `run(config)`

Runs perplexity comparisons across watermarked and unwatermarked generators. It generates continuations, computes continuation-only perplexity under the base model, summarizes the distribution, and logs results to WandB.

This runner is focused on fluency cost rather than detector performance.

## `experiments/runs/translation.py`

### `TranslationPipeline`

Typed container for the round-trip translation models and tokenizers used by the translation attack experiment.

### `_load_translation_pipeline(language, device)`

Loads the forward and backward translation models for a pivot language such as French or Russian and returns them in a `TranslationPipeline`.

### `_generate_samples(model, tokenizer, data, device, max_length, wat=None)`

Generates continuations from the base model, optionally through a supplied watermarker. It returns three aligned lists:

- decoded generated texts
- generated continuation token ids
- prompt token ids

This prepares the examples later attacked by round-trip translation.

### `_round_trip_texts(texts, pipeline, device, batch_size, max_source_length, max_target_length)`

Runs round-trip translation through the forward and backward models stored in the provided `TranslationPipeline`.

This simulates paraphrastic attacks that may disrupt watermark evidence while preserving meaning.

### `_retokenize_texts(tokenizer, texts, max_length)`

Converts attacked strings back into 1D token tensors without adding special tokens, truncated to the requested max length.

### `_evaluate_attacked_samples(wat, model, tokenizer, attacked_texts, prompt_token_ids, max_retokenized_length, n_samples, upperbound)`

Retokenizes round-tripped texts and evaluates them under the supplied detector. It returns:

- attacked-sample p-values
- positive detector scores for the attacked samples
- attacked token lengths

The unbiased detector path includes the prompt prefix required by that method.

### `_save_examples(output_dir, model_name, wat_name, language, max_length, original_texts, pivot_texts, round_trip_texts, attacked_lengths, n_examples)`

Builds a small table of example translation attacks and writes it to disk as a CSV artifact for manual inspection.

### `_resolve_sharding(config)`

Resolves `shard_index` and `num_shards` for the translation runner. It accepts config values and allows environment overrides via `WEPA_TRANSLATION_SHARD_INDEX` and `WEPA_TRANSLATION_NUM_SHARDS`.

### `_build_jobs(languages, wat_names, max_lengths)`

Builds the deterministic `(language, watermark, max_length)` job grid used by the translation runner before shard partitioning.

### `_resolve_watermarker_names(watermarkers, configured_names=None)`

Resolves the subset of translation watermarkers to evaluate. When `configured_names` is omitted, it keeps the full default set. When names are provided, it preserves their order, removes duplicates, and raises an explicit error if any requested watermark is unsupported.

### `run(config)`

Runs the translation-based attack experiment end to end. It generates samples, applies round-trip translation attacks, retokenizes attacked outputs, evaluates detector degradation, and saves both metrics and examples.

Unless the config explicitly sets `upperbound: true`, translation detection uses the standard non-upperbound p-value path. In that default mode, the runner also defaults `n_samples` to `10000`.

The translation runner can also shard the `(language, watermark, max_length)` job grid across multiple workers. By default it uses `shard_index=0` and `num_shards=1`, which preserves single-process behavior. A config may additionally restrict the evaluated watermark subset via `watermarkers: [...]`.

## `experiments/runs/varying_bits.py`

### `run(config)`

Sweeps over WEPA bit-width settings and automaton degrees. For each combination it builds a WEPA watermarker and delegates the actual evaluation to `experiments.utils.run_experiment(...)`.

The degree list and bit-width sweep are read from config when provided.

## `experiments/runs/varying_lambda.py`

### `run(config)`

Sweeps over `lam`, the key sequence or automaton size parameter, for EXP and WEPA-style methods. It constructs the appropriate watermarker for each setting and runs the shared experiment loop.

The degree list and `lam` sweep are read from config when provided.

## `experiments/runs/varying_length.py`

### `run(config)`

Evaluates watermark detection as a function of generated continuation length for several watermark families. It builds a fixed set of watermarkers, sweeps a list of target generation lengths, and calls the shared evaluation loop at each length.

The continuation-length sweep is read from config when provided.

## `experiments/runs/varying_length_long.py`

### `run(config)`

Long-sequence variant of the varying-length experiment. It focuses on larger continuation lengths and uses the shared evaluation loop with an upper-bound detection mode for some methods.

The long-sequence length sweep is read from config when provided.

## `src/watermarker/base.py`

### `Watermarker.test_statistic(token_ids)`

Abstract detector interface. Concrete watermark implementations must define how to score a token sequence.

The returned statistic is the raw detector score used by downstream AUC and threshold-based metrics.

### `Watermarker.scores(neg_input_ids_list, pos_input_ids_list)`

Applies `test_statistic(...)` to negative and positive token lists, concatenates the resulting scores, and returns both the score array and the corresponding label array.

This helper standardizes score/label construction for generic evaluation code.

## `src/watermarker/dip.py`

### `DIPConfig.__init__(vocab_size, device=torch.device("cpu"), alpha=0.5, gamma=0.5, hash_key=b"42", prefix_length=5, ignore_history_generation=False, ignore_history_detection=False, z_threshold=4.0)`

Stores the repo-native DiP hyperparameters needed for generation and detection, including the hash key used to derive context-dependent permutations.

### `DIPUtils.__init__(config)`

Initializes DiP helper state, including the context-history cache and a mode flag distinguishing generation from detection.

### `DIPUtils._get_rng_seed(context_code)`

Hashes a context representation with the private key to derive the deterministic random seed used for the current context.

### `DIPUtils._extract_context_code(context)`

Extracts the prefix context bytes used to seed the cipher, either from the full context or from the configured prefix window.

### `DIPUtils.from_random(rng, vocab_size, device)`

Samples a token permutation from one generator or a batch of generators. This is used to build the shuffled vocabulary view required by DiP.

### `DIPUtils.reweight_logits(shuffle, p_logits)`

Applies the DiP probability reweighting transform in shuffled-token space and maps the result back to the original vocabulary order.

### `DIPUtils.get_seed_for_cipher(input_ids)`

Builds the per-example “seen before” mask and seed values for the current batch of contexts.

### `DIPUtils._get_green_token_quantile(input_ids, current_token)`

Computes the quantile-like position of the current token under the DiP greenlist construction.

### `DIPUtils._get_dip_score(input_ids)`

Computes per-token DiP detector scores over a full token sequence.

### `DIPUtils.score_sequence(input_ids)`

Aggregates DiP token-level scores over a full sequence and returns the z-style detector statistic plus token-level flags.

### `DIPLogitsProcessor.__init__(config, utils)`

Stores the DiP config and utilities needed to modify logits during generation.

### `DIPLogitsProcessor._apply_watermark(input_ids, scores)`

Applies DiP’s reweighting rule to a batch of logits conditioned on the current contexts.

### `DIPLogitsProcessor.__call__(input_ids, scores)`

Hugging Face logits-processor interface that injects DiP watermarking into generation.

### `DipWatermarker.__init__(config)`

Builds the repo-native DiP watermarker, including helper utilities and the Hugging Face logits processor used during generation.

### `DipWatermarker.generate(model, inputs, max_new_tokens)`

Generates a watermarked continuation from tokenized model inputs using the DiP logits processor.

### `DipWatermarker.test_statistic(token_ids)`

Computes the DiP z-style detector statistic for a token sequence. Larger values indicate stronger watermark evidence.

### `DipWatermarker.p_value(token_ids)`

Converts the DiP detector statistic into a one-sided Gaussian-tail p-value.

### `DIP`

Compatibility alias for `DipWatermarker`.

## `src/watermarker/exp.py`

### `_distance_edit_jit(token_ids, cost, gamma)`

Numba-compiled dynamic program for the EXP detector’s edit-style distance computation. It evaluates how well a token sequence aligns with the secret structure under insertion/deletion-style penalties.

### `_test_statistic_jit(token_ids, cost, block_size, gamma)`

Numba-compiled helper for the EXP detector statistic. It slides a fixed-size token block across the sequence, evaluates every cyclic cost-matrix offset through `_distance_edit_jit(...)`, and returns the minimum distance observed.

### `ExpWatermarker.__init__(lam, vocab_size, block_size=None, gamma=0, shift=True, generator=None, device=torch.device("cpu"))`

Initializes the EXP watermarker. Internally it reuses a WEPA instance to generate the secret substates and cost matrix, then exposes EXP-specific detection methods over that structure.

### `ExpWatermarker.generate(model, inputs, max_new_tokens)`

Generates watermarked text by delegating to the internal WEPA generator. EXP in this repo reuses the same generation mechanism and differs mainly in detection.

### `ExpWatermarker.test_statistic(token_ids)`

Computes the EXP detector statistic by converting the tensor to a NumPy array and delegating the sliding-window distance search to `_test_statistic_jit(...)`.

Lower values indicate stronger agreement with the watermark template.

### `ExpWatermarker.z_score(token_ids, n_samples=20)`

Estimates a z-score for the observed detector statistic by comparing it to statistics from randomized surrogate samples.

### `ExpWatermarker.p_value(token_ids, n_samples=20)`

Estimates a p-value for the observed detector statistic using Monte Carlo samples from the null distribution.

### `ExpWatermarker.p_value_unoptimized(token_ids, n_samples=20)`

Reference implementation of the EXP p-value estimator without the optimized shortcuts used in the default detector path.

### `ExpWatermarker.scores(neg_input_ids_list, pos_input_ids_list, n_samples=20)`

Evaluates EXP p-values for negative and positive example sets and returns scores and labels for downstream metrics.

## `src/watermarker/kgw.py`

### `KGWLogitsProcessor.__init__(vocab_size, green_ratio, delta, generator=None, device=torch.device("cpu"))`

Initializes the KGW logits processor, including the vocabulary size, greenlist ratio, score shift `delta`, and seeded generator.

### `KGWLogitsProcessor._random_permutation(context_code)`

Generates the context-dependent permutation used to derive KGW greenlists.

### `KGWLogitsProcessor._get_greenset_ids_cached(context_code)`

Cache-aware helper that returns green token ids for a previously seen context.

### `KGWLogitsProcessor._get_greenset_ids(context_code)`

Builds the green token set for a context, either by lookup or by generating a fresh permutation.

### `KGWLogitsProcessor.__call__(input_ids, scores)`

Applies the KGW logit bias to green tokens for the current context before sampling.

### `KGWWatermarker.__init__(key, vocab_size, green_ratio, delta, generator=None, device=torch.device("cpu"))`

Constructs the KGW watermarker and the corresponding logits processor.

### `KGWWatermarker._random_permutation(context_code)`

Internal context-based permutation helper mirroring the logits processor’s greenlist construction.

### `KGWWatermarker._get_green_ids_cached(context_code)`

Cache-aware helper for detector-side greenlist lookup.

### `KGWWatermarker._get_green_ids_set(context_code)`

Detector-side construction of the green token set associated with a context.

### `KGWWatermarker._sample_token(probs)`

Samples a token from the next-token probability distribution after the greenlist bias has been applied.

### `KGWWatermarker.generate(model, inputs, max_new_tokens)`

Generates a watermarked continuation using the KGW logits processor.

### `KGWWatermarker.test_statistic(token_ids)`

Computes the KGW detector statistic, typically the count- or z-style evidence that observed tokens fell into the expected greenlists more often than chance.

### `KGWWatermarker.p_value(token_ids, n_samples=20)`

Converts the KGW detector statistic into a p-value-like significance measure.

### `KGWWatermarker.z_score(token_ids)`

Computes the KGW z-score used as a detector summary.

### `KGWWatermarker.scores(neg_input_ids_list, pos_input_ids_list)`

Returns detector scores and labels for negative and positive sample sets.

## `src/watermarker/topp.py`

### `TopPSampler.__init__(device=torch.device("cpu"))`

Initializes a simple top-p sampler helper.

### `TopPSampler._top_p_sampling(logits, p=0.9)`

Applies nucleus sampling to a vector of logits and returns a sampled token id from the truncated distribution.

### `TopPSampler.generate(model, input_ids, max_length, p=0.9)`

Generates a continuation token by token using repeated forward passes and top-p sampling.

This helper is independent of the watermark detectors and is mainly a plain sampling utility.

## `src/watermarker/unbiased.py`

### `TransformersConfig.__init__(model, tokenizer, vocab_size=None, device='cuda', *args, **kwargs)`

Stores the model, tokenizer, effective vocabulary size, and device information needed by the unbiased watermark implementation.

### `BaseConfig.__init__(vocab_size, device=torch.device("cpu"), *args, **kwargs)`

Builds the default unbiased watermark config dictionary, merges overrides, stores device metadata, and delegates algorithm-specific field extraction to `initialize_parameters()`.

### `BaseConfig.initialize_parameters()`

Abstract hook for subclasses to load config values into explicit attributes.

### `BaseConfig.algorithm_name`

Abstract property returning the algorithm name.

### `BaseWatermark.__init__(algorithm_config, *args, **kwargs)`

Placeholder initializer for the imported unbiased watermark implementation.

### `BaseWatermark.generate_watermarked_text(prompt, *args, **kwargs)`

Placeholder generation interface for the imported unbiased watermark implementation.

### `BaseWatermark.p_value(text, return_dict=True, *args, **kwargs)`

Placeholder p-value interface for the imported unbiased watermark implementation.

### `WatermarkStrategy.from_random(rng, vocab_size)`

Abstract strategy method for sampling the random object used by a watermarking scheme from a generator.

### `WatermarkStrategy.reweight_logits(shuffle, p_logits, alpha)`

Abstract strategy method for converting the random object and next-token logits into watermark-modified logits.

### `DeltaStrategy.from_random(rng, vocab_size)`

Samples the scalar random variable used by delta-style unbiased watermarking.

### `DeltaStrategy.reweight_logits(u, p_logits)`

Converts next-token logits into a one-token support distribution determined by the sampled scalar `u`.

### `GammaStrategy.from_random(rng, vocab_size)`

Samples a full vocabulary permutation for gamma-style unbiased watermarking.

### `GammaStrategy.reweight_logits(shuffle, p_logits, alpha=0.5)`

Applies the gamma-style unbiased reweighting transform in shuffled-token space and maps the result back to the original vocabulary order.

### `UnbiasedConfig.initialize_parameters()`

Copies unbiased-watermark hyperparameters such as watermark type, grid size, key, prefix length, p-value threshold, and history behavior from the config dictionary to the config object.

### `UnbiasedConfig.algorithm_name`

Returns the algorithm name for the unbiased watermark implementation.

### `UnbiasedUtils.__init__(config, *args, **kwargs)`

Initializes helper state for the unbiased watermark, including the selected reweighting strategy, RNG, and context history.

### `UnbiasedUtils._get_rng_seed(context_code)`

Hashes the current context into the deterministic seed used to sample the next watermark random object.

### `UnbiasedUtils._extract_context_code(context)`

Extracts the context bytes that determine the watermark seed for the current position.

### `UnbiasedUtils.get_seed_for_cipher(input_ids)`

Returns per-example history flags and deterministic seeds for the current batch of contexts.

### `UnbiasedUtils._apply_watermark(input_ids, scores)`

Applies the unbiased watermark transform to a batch of next-token logits.

### `UnbiasedUtils._safe_minus(a, b)`

Numerically safe subtraction helper used in likelihood-ratio calculations.

### `UnbiasedUtils._from_grid(dist_ps)`

Maps detector values onto the finite grid used by the discrete null approximation.

### `UnbiasedUtils._get_max_llr(dist_ps, dist_qs)`

Computes the maximum log-likelihood ratio used in detection.

### `UnbiasedUtils._score_llr(dist_ps, dist_qs)`

Computes the sequence-level likelihood-ratio score for the unbiased detector.

### `UnbiasedUtils.score_sequence(input_ids, p_logits, q_logits)`

Aggregates per-position detector evidence over a full sequence.

### `UnbiasedLogitsProcessor.__init__(config, utils, *args, **kwargs)`

Stores config and helper state for the unbiased logits processor.

### `UnbiasedLogitsProcessor.__call__(input_ids, scores)`

Hugging Face logits-processor hook that watermarks generation under the unbiased scheme.

### `UnbiasedWatermarker.__init__(config, *args, **kwargs)`

Builds the full unbiased watermark object, including helper utilities and logits processor state.

### `UnbiasedWatermarker.generate(model, inputs, max_new_tokens)`

Generates watermarked continuations under the unbiased watermarking scheme.

### `UnbiasedWatermarker.p_value(token_ids, model)`

Computes the significance of a token sequence under the unbiased detector, using the base model when required by the algorithm.

### `UnbiasedWatermarker.scores(neg_input_ids_list, pos_input_ids_list, model)`

Evaluates detector scores for negative and positive example sets under the unbiased watermark detector.

### `UnbiasedWatermarker.test_statistic(model, token_ids)`

Computes the raw unbiased detector statistic for a sequence.

## `src/watermarker/wepa.py`

### `_distance_edit_jit(token_ids, degree, cost, gamma_d, gamma_i)`

Numba-compiled edit-distance-style dynamic program used by the WEPA detector. It evaluates how well a token sequence aligns with the automaton-defined secret substates while accounting for substitutions, deletions, and insertions.

### `WepaLogitsProcessor.__init__(lam, vocab_size, substates, degree=1, bits=None, generator=None, device=torch.device("cpu"))`

Initializes the WEPA logits processor with the automaton size, vocabulary size, substates, optional bit-precision setting, and random generator state.

### `WepaLogitsProcessor.reset_state()`

Randomly initializes the current automaton state before generation begins.

### `WepaLogitsProcessor.__call__(input_ids, scores)`

Transforms next-token logits into a one-token support distribution determined by the current automaton state and then advances the state according to the WEPA transition rule.

### `WepaWatermarker.__init__(lam, vocab_size, degree=1, bits=None, gamma_d=1, gamma_i=2, generator=None, device=torch.device("cpu"))`

Constructs the WEPA watermarker, samples the secret substates, builds the detector cost matrix, validates hyperparameters, and creates the logits processor used during generation.

### `WepaWatermarker.distance_edit(token_ids)`

Runs the WEPA dynamic program on a token sequence and returns the minimum edit-style distance to the automaton structure.

### `WepaWatermarker.generate(model, inputs, max_new_tokens)`

Generates a watermarked continuation with batch size 1 using the WEPA logits processor.

### `WepaWatermarker.test_statistic(token_ids)`

Returns the detector statistic for a token sequence. In this implementation it is the negative normalized edit distance, so larger values indicate stronger watermark evidence.

### `WepaWatermarker.z_score(token_ids, n_samples=20)`

Estimates a z-score for the observed WEPA detector statistic from Monte Carlo null samples.

### `WepaWatermarker.p_value(token_ids, n_samples=20, upperbound=False)`

Estimates a p-value for the observed WEPA detector statistic. The optional `upperbound` path uses a coarser estimate for long-sequence experiments.

### `WepaWatermarker.scores(neg_input_ids_list, pos_input_ids_list, n_samples=20)`

Evaluates WEPA p-values for negative and positive sample sets and returns scores and labels.
