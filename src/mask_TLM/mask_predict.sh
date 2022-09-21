#!/usr/bin/env bash

dir="/home/bli/syntactic_sensitivity/data"

python nnlm.py ../pretrained_TLM/fr-TM/26_8/  --test_file $dir/agreement/subj-v/subj-que-v_aux.txt \
                            --target_file $dir/agreement/subj-v/subj-que-v_aux.tab \
                            --json_file $dir/mask_intervention/subj-v_masking_ids/mask_context_ids.json \
                            --out $dir/mask_intervention/subj-v_after_intervention/mask-context_transformer_26_8.pred