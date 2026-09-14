## MLPerf-Mobile v6.0 LLM Benchmark

MLPerf-Mobile v6.0 adopted 2 models for the LLM Benchmark, with a maximum input-prompt length of 2048 tokens:

1. Llama 3.1 8B Instruct: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
2. Llama 3.2 3B Instruct: https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct

Only Post-Training Quantization (PTQ) is allowed for quantization of the LLM models for benchmarking purposes. The calibration dataset approved for use with PTQ is the WikiText dataset: 
https://huggingface.co/datasets/Salesforce/wikitext/tree/main/wikitext-2-raw-v1  

For Accuracy and Performance measurements, MLPerf Mobile v6.0 LLM benchmark adopted two datasets: 
1. Tiny-MMLU using the FewShot-2 protocol, with 100 input_formatted style promnpts having at most 5 shots, where example shots are truncated from the preformatted input prompts, such that the input prompt length does not exceed 2048 tokens. The Tiny-MMLU dataset can be downloaded from here https://huggingface.co/datasets/tinyBenchmarks/tinyMMLU/tree/main/all.
2. Tiny-IFEval-33: The MLPerf Mobile Working Group carefully selected 33 prompts from the IFEval dataset https://huggingface.co/datasets/google/IFEval/blob/main/README.md, to be a representative dataset. The Tiny_IFEval-33 dataset is balanced in multiple aspects: From the 33 prompts, 13 promnpts have 1 instruction each, 10 have 2 instructions each, and the remainig 10 prompts have 3 instructions each. They are also selected such that each type of instruction has at least 2 instances across all promnpts, and the instructions are also representative of all the different instruction categories. The Tiny-IFEval-33 dataset can be found here [IFEval33](/dataset/ifeval_33.jsonl).