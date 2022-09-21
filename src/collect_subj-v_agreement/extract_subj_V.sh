#!/usr/bin/env bash


lang="French"
test="/data/bli/parsed_corpora/fr_gutenberg_spacy_conllu/archives/subj-v/subj-que-v.aux/gutenberg_subj-que-v_candidates.spacy-conll"
#test="/home/bli/synStructure/data/treebanks/French/test2.conllu"
treebank="/home/bli/synStructure/data/treebanks/$lang"
lm_data="/home/bli/synStructure/data/lm_data_prep/data/wiki/French"
output="/home/bli/synStructure/data/agreement/$lang/subj-verb/subj_relv_verb/subj_aux_pp-adj" #
# extract obj pp agreement
python extract_subj-pp_agreement.py --treebank $test \
                            --paradigms $treebank/fr_subj_V_full_paradigms_min1.txt \
                            --vocab $treebank/full_corpus/fr_lm_vocab.txt\
                            --output $output \
                            --lm_data $lm_data \




<<c
treebank="/data/bli/parsed_corpora/fr_gutenberg_spacy_conllu/archives"
paradigms="/home/bli/synStructure/data/treebanks/French/fr_subj_V_full_paradigms_min1.txt"
vocab="/home/bli/TACL/src/jean-zay-lm/fr-TM/28.2/tokcodes"
trainDIR="/home/bli/synStructure/data/lm_data_prep/data/wiki/French"
outPref="/home/bli/synStructure/data/agreement/French/subj-verb/subj_relv_verb/fix5_subj-que-v_aux"
#outPref="/home/bli/TACL/data/agreement/French/subj-verb/test_aux"

python extract_subj-que_aux_fix5.py  --treebank $treebank/subj-v/subj_unk_v/subj_unk_v-aux_gutenberg_candidates.spacy-conll \
                             --vocab $vocab \
                             --paradigms $paradigms \
                             --lm_data $trainDIR \
                             --output $outPref
c



