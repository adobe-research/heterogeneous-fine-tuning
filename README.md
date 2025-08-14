#  A Framework for Fine-Tuning LLMs using Heterogeneous Feedback 

This repository contains the code and data associated with our RANLPL 2025 paper [A Framework for Fine-Tuning LLMs using Heterogeneous Feedback](https://arxiv.org/abs/2408.02861) by Ryan Aponte, Ryan A. Rossi, Shunan Guo, Franck Dernoncourt, Tong Yu, Xiang Chen, Subrata Mitra, and Nedim Lipka.

Fine-tuning was performed on 8xA100-80GB and Python 3.7 was used. Fine-tuning was performed with [Stack-LLaMA](https://github.com/huggingface/trl/tree/main/examples/research_projects/stack_llama/scripts) and the entire process took under 24 hours per model.

## Explanation of Directories

1. 7B_huggingface - the weights for LLaMA in Huggingface format
2. evaluation - contains scripts to get results and directories for results
3. finetune_llama - fine-tuned model weights
4. generative_task - generative task in Appendix E. 3
5. instruction_following_eval - Script to generate dataset for IFEval.

## Citation
If you use this repository, please cite our [paper](https://arxiv.org/abs/2408.02861):
```bibtex
@misc{aponte2024frameworkfinetuningllmsusing,
      title={A Framework for Fine-Tuning LLMs using Heterogeneous Feedback}, 
      author={Ryan Aponte and Ryan A. Rossi and Shunan Guo and Franck Dernoncourt and Tong Yu and Xiang Chen and Subrata Mitra and Nedim Lipka},
      year={2024},
      eprint={2408.02861},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2408.02861}, 
}
```

## License

The evaluation code and needle set data is licensed under the [Adobe Research License](LICENSE.md). The license prohibits commercial use and allows non-commercial research use.
