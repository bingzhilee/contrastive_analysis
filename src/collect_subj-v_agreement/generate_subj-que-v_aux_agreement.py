# -*- coding: utf-8 -*-

""" generate French subject *que(obj)* verb/aux agreement test set and collect relevant statistics """

import depTree
import json
import argparse
import pandas as pd
from collections import Counter,defaultdict
from generate_utils import  match_features,ltm_to_word,get_alt_form,read_paradigms, load_vocab, vocab_freqs,is_good_form
import random

def inside(tree, a):
    # return tuple (nodes, l, r), nodes is the context, l is the cue and r is the target
    if a.child.index < a.head.index:
        nodes = tree.nodes[a.child.index: a.head.index - 1]
        l = a.child
        r = a.head
    else:
        nodes = tree.nodes[a.head.index: a.child.index - 1]
        l = a.head
        r = a.child
    return nodes, l, r
def alt_numeral_morph(morph):
    if "Number=Plur" in morph:
        morph_alt = morph.replace("Plur", "Sing")
        return "plur", morph_alt
    elif "Number=Sing" in morph:
        morph_alt = morph.replace("Sing", "Plur")
        return "sing", morph_alt
def plurality(morph):

    if "Number=Plur" in morph:
        return "plur"
    elif "Number=Sing" in morph:
        return "sing"
    else:
        return "none"


def extract_sent_features(t,nodes,l,r,vocab):
    """ Extracting some features of the construction and the sentence for data analysis """
    r_idx = int(r.index)
    l_idx = int(l.index)
    # problem of numbers : '3 500' is one node in conll, but two tokens for lm
    prefix = " ".join(n.word.replace(' ','') for n in t.nodes[:r_idx-1])
    prefix_lm_list = prefix.split()
    len_prefix = len(prefix_lm_list)
    n_unk = len([w for w in prefix_lm_list if w not in vocab])
    correct_number = plurality(r.morph)
    context_attr_noun_ids = [str(n.index) for n in nodes if n.pos=="NOUN" and plurality(n.morph)!=correct_number]
    str_context_noun_ids = "_".join(context_attr_noun_ids)
    #compute the cls noun number
    context_noun_num = [plurality(n.morph) for n in nodes if
                        n.pos in ["NOUN", "PROPN"] and plurality(n.morph) != "none"]
    cls_noun_num = context_noun_num[-1] if context_noun_num else "none"

    #compute the cls number on the left of target verb
    context_num = [plurality(n.morph) for n in nodes if plurality(n.morph) != "none"]
    cls_token_num = context_num[-1] if context_num else "none"

    # calculate the major number before the target verb in the sentence
    prefix_number_values = [plurality(n.morph) for n in t.nodes[:r.index-1]]
    prefix_number = [v for v in prefix_number_values if v!="none"]
    prefix_number.reverse()# if #sing == #plur, take the closer number to target verb
    com_num = Counter(prefix_number).most_common(1)[0][0]

    # compute the first noun number
    noun_nodes = [n for n in t.nodes[:l.index-1] if n.pos in ["NOUN","PROPN"]]
    noun_num = [plurality(n.morph) for n in noun_nodes if plurality(n.morph) != "none"]
    fst_noun_num = noun_num[0] if noun_num else "none"

    # get the left closest noun of the left closet 'que'/'qui' to target verbe
    que_list = [n for n in t.nodes[:r.index - 1] if n.word.lower() in ["que","qu'"]]

    if que_list:
        closest_que_id = que_list[-1].index
        que_nouns_num = [plurality(n.morph) for n in t.nodes[:closest_que_id - 1] if
                         n.pos in ["NOUN", "PROPN"] and plurality(n.morph) != "none"]
        que_n_num = que_nouns_num[-1] if que_nouns_num else "none"
    else:
        que_n_num = "none"
        print(" ".join([n.word for n in l+nodes+r]))

    # PUNCT
    sent_poses = [n.pos for n in t.nodes]
    if sent_poses[l.index-2]=="PUNCT" or (sent_poses[l.index-2]=="DET" and sent_poses[l.index-3]=="PUNCT"):
        punct_before_nsubj = True
    else:
        punct_before_nsubj = False
    if sent_poses [l.index]=="PUNCT":
        punct_after_nsubj = True
    else:
        punct_after_nsubj = False
    if sent_poses [r.index-2]=="PUNCT":
        punct_before_verb = True
    else:
        punct_before_verb = False

    len_context = r_idx - l_idx

    # compute the number of attrs
    attrs = [nb for nb in context_noun_num if nb != correct_number] if context_noun_num else []
    soft_attrs = len(attrs)

    if len(attrs) == len(context_noun_num):
        homo_attrs = len(attrs)

    else:
        homo_attrs = 0


    return str_context_noun_ids,correct_number,cls_noun_num,cls_token_num,fst_noun_num,com_num,que_n_num,\
           punct_before_nsubj,punct_after_nsubj,punct_before_verb,soft_attrs,homo_attrs,len_prefix,len_context,n_unk


def match_subj_v_agreement(l, r, ltm_paradigms, vocab):
    if (not "Number" in l.morph) or (not "Number" in r.morph) or plurality(l.morph) != plurality(r.morph):
        return False

    # check if the form and alt_form of verbs are in the vocab
    r_alt = get_alt_form(r.lemma, r.pos, r.morph, ltm_paradigms)
    l_alt = get_alt_form(l.lemma, l.pos, l.morph, ltm_paradigms)
    if not r_alt or not l_alt:
        return False
    # if the noun is invariable or the verb is invariable or the altertive verb is not in lm vocab
    if l.word == l_alt or r.word == r_alt or r_alt not in vocab:
        return False

    # r is wrongly capitalized or the wrong pair: soit eussent
    if not r_alt[0] == r.word[0] and r.word not in ["a", "ont", "est", "sont",
                                                    "suis", "sommes", "vais", "allons", "vas", "allez"]:
        #print("r_alt = ", r_alt, "r_word = ", r.word)
        return False
    return True


def compute_mask_ids(l, r, context_nodes, tree):
    # compute que(obj) id: leftmost "que","obj"
    context_w_dep = [(n.word.lower(), n.dep_label) for n in
                     tree.nodes[l.index:r.index - 1]]
    context_dep = [n.dep_label for n in
                   tree.nodes[l.index:r.index - 1]]
    context_idx = [n.index for n in
                   tree.nodes[l.index:r.index - 1]]
    context_word_obj = [n.word.lower() for n in context_nodes if n.dep_label == "obj"]
    if "que" in context_word_obj:
        que_id = context_idx[context_w_dep.index(("que", "obj"))]
    elif "qu'" in context_word_obj:
        que_id = context_idx[context_w_dep.index(("qu'", "obj"))]
    # if no que-obj in the context, skip
    else:
        que_id = None

    # compute if rel_nsubj is inversed
    # index of the head of the relative clause(VERB: direct child of nsubj)
    if "acl:relcl" not in context_dep:
        rel_nsubj_inversed = "none"

    else:
        rel_verb_id = context_idx[context_dep.index("acl:relcl")]

        rel_verb_nsubj = [n for n in context_nodes if
                          n.head_id == rel_verb_id and n.dep_label in ["nsubj", "nsubj:pass", "nsubj:caus"]]
        if rel_verb_nsubj:
            rel_nsubj_id = rel_verb_nsubj[0].index
            if rel_nsubj_id > rel_verb_id:
                rel_nsubj_inversed = True

            else:
                rel_nsubj_inversed = False
        else:
            rel_nsubj_inversed = "none"

    # compute nsubj modifiers' ids (det, adj with the same number) for masking intervention
    child_l_det_ids = [str(n.index) for n in tree.nodes[:l.index - 1] if
                       n.head_id == l.index and n.dep_label == "det" and plurality(n.morph) == plurality(
                           l.morph)]
    child_l_adj_4w_ids = [str(n.index) for n in tree.nodes[l.index - 3:l.index + 2] if
                          n.head_id == l.index and n.dep_label == "amod" and plurality(n.morph) == plurality(
                              l.morph)]
    det_nsubj_idx = "none"
    amod_nsubj_idx = "none"
    if child_l_det_ids:
        det_nsubj_idx = "_".join(child_l_det_ids)
    if child_l_adj_4w_ids:
        amod_nsubj_idx = "_".join(child_l_adj_4w_ids)

    return que_id,rel_nsubj_inversed,det_nsubj_idx,amod_nsubj_idx

collective_words = ["plupart","moitié","bande", "dizaine", "ensemble", "foule", "horde", "millier",
                    "multitude", "majorité", "nombre", "ribambelle", "totalité", "troupeau"]

def is_content_word(pos):
    return pos in ["ADJ", "NOUN", "VERB", "PROPN", "NUM", "ADV"]

def generate_sentence(t,l,r, paradigms,ltm_paradigms, vocab):
    output = []
    output_alt = []
    form, form_alt = r.word,get_alt_form(r.lemma, r.pos, r.morph, ltm_paradigms)
    cue_form, cue_form_alt = l.word,get_alt_form(l.lemma, l.pos, l.morph, ltm_paradigms)

    for i in range(r.index):
        if i == l.index-1:
            random_forms_cue = choose_random_forms(ltm_paradigms, vocab, l.pos, l.morph, n_samples=1,
                                               gold_word=l.word)
            #print(random_forms_cue)
            if random_forms_cue:
                _, cue_form, cue_form_alt = random_forms_cue[0]
            output.append(cue_form)
            output_alt.append(cue_form)
            continue

        if i == r.index -1 and r.pos !="AUX":

            random_forms_target = choose_random_forms(ltm_paradigms, vocab, r.pos, r.morph, n_samples=1,
                                               gold_word=r.word)
            #print(random_forms_target)
            if random_forms_target:
                _, form, form_alt = random_forms_target[0]
            output.append(form)
            output_alt.append(form_alt)
            continue
        elif i == r.index -1 and r.pos =="AUX":
            output.append(form)
            output_alt.append(form_alt)
            continue
        if t.nodes[i].pos == "ADV" and t.nodes[i].word == "n'":
            output.append(t.nodes[i].word)
            output_alt.append(t.nodes[i].word)
            continue


        substitutes = []
        n = t.nodes[i]
        # substituting content words
        if is_content_word(n.pos):
            for word in paradigms:
                if word == n.word:
                    continue
                # matching capitalization and vowel
                # replace the word starts with majuscule and vowel with the same morphological word
                if  not match_features(word,n.word):
                    continue

                #d["is"] == ('be', 'AUX', 'Number=Sing|Mood=Ind|Tense=Pres|VerbForm=Fin', 100)
                tag_set = set([p[1] for p in paradigms[word]]) # different POS of the word to substitute
                # use words with unambiguous POS
                if len(tag_set) == 1 and tag_set.pop() == n.pos: # .pop() return and remove the last element of set
                    for _, _, morph, freq in paradigms[word]:
                        if n.morph == morph and int(freq) > 1 and word in vocab:
                            substitutes.append(word)

            if len(substitutes) == 0:
                output.append(n.word)
                output_alt.append(n.word)
            else:
                w = random.choice(substitutes)
                output.append(w)
                output_alt.append(w)
        else:
            output.append(n.word)
            output_alt.append(n.word)
    return " ".join(output)," ".join(output_alt),form, form_alt

def choose_random_forms(ltm_paradigms, vocab, gold_pos, morph, n_samples=10, gold_word=None):
    candidates = set()
    for lemma in ltm_paradigms:
        poses = list(ltm_paradigms[lemma].keys())

        if len(set(poses)) == 1 and poses.pop() == gold_pos:
            form = ltm_paradigms[lemma][gold_pos][morph]
            _, morph_alt = alt_numeral_morph(morph)
            form_alt = ltm_paradigms[lemma][gold_pos][morph_alt]
            if form == form_alt: # sing and plur candidate word have the same form
                continue

            if not is_good_form(gold_word, form, morph, lemma, gold_pos, vocab, ltm_paradigms):
                continue

            candidates.add((lemma, form, form_alt))

    if len(candidates) > n_samples:
        return random.sample(list(candidates), n_samples)
    else:
        return random.sample(list(candidates), len(candidates))

def collect_agreement(trees,  paradigms, feature_list, vocab,constr_size):
    output = []
    whole_constr = set()
    constr_id = 0
    ltm_paradigms = ltm_to_word(paradigms)  # best_paradigms_lemmas['be']['Aux'][morph]=='is'

    for tree in trees:
        for a in tree.arcs:
            if a.length() > 3:
                context_nodes, l, r = inside(tree, a)

                if l.word.lower() in collective_words:
                    continue
                # make sure there is no "conjunction nominal subject"
                conj_pos = [n.pos for n in context_nodes if n.head_id == l.index and n.dep_label == "conj"]
                if "NOUN" in conj_pos or "PROPN" in conj_pos :
                    continue

                in_vocab = all([n.word in vocab for n in context_nodes + [l, r]])
                if not in_vocab:
                    continue

                if l.pos not in ["NOUN"] or a.dep_label not in ["nsubj"]:
                    continue

                child_r = [n for n in context_nodes if
                           n.head_id == r.index and n.pos == "AUX" and plurality(n.morph)!="none"]

                if r.pos =="VERB" and "VerbForm=Fin" in r.morph and match_subj_v_agreement(l,r,ltm_paradigms,vocab):
                    pattern = "subj_rel_verb"
                elif  r.pos in ["NOUN", "ADJ", "VERB","PRON","PROPN","ADV"] and len(child_r) == 1 \
                        and match_subj_v_agreement(l, child_r[0], ltm_paradigms,vocab) and child_r[0].index-l.index > 3:# livre que voici sera
                    pattern = "subj_rel_aux"
                    r = child_r[0]
                    context_nodes = tree.nodes[l.index:child_r[0].index-1]
                else:
                    continue

                que_index, rel_nsubj_inversed, det_nsubj_idx, amod_nsubj_idx = compute_mask_ids(l, r, context_nodes,tree)
                if que_index:
                    que_id = que_index
                else:
                    continue

                # rm duplicate contexts
                constr = " ".join(str(n.word) for n in [l] + context_nodes + [r])
                if constr in whole_constr:
                    continue
                else:
                    whole_constr.add(constr)
                    print(constr_id,"0:", constr)
                    pos = " ".join([n.pos for n in tree.nodes])


                str_context_noun_ids, correct_number, cls_noun_num, cls_token_num, fst_noun_num, com_num, que_n_num, \
                punct_before_nsubj, punct_after_nsubj, punct_before_verb, soft_attrs, homo_attrs, len_prefix, len_context, n_unk = extract_sent_features(
                    tree, context_nodes, l, r, vocab)

                for i in range(1, constr_size):
                    new_sent_p, new_sent_alt_p, form, form_alt = generate_sentence(tree, l, r, paradigms, ltm_paradigms,vocab)
                    sent_suffix = " ".join(n.word for n in tree.nodes[r.index:])
                    new_sent = new_sent_p + " " + sent_suffix
                    new_sent_alt = new_sent_alt_p + " " + sent_suffix
                    print(i," ".join(new_sent_p.split()[l.index-1:r.index]))

                    output.append((pattern, constr_id,i, "correct", form, str_context_noun_ids,correct_number, cls_noun_num, cls_token_num, fst_noun_num,
                                   com_num, que_n_num, punct_before_nsubj, punct_after_nsubj, punct_before_verb, soft_attrs,
                                   homo_attrs, len_prefix, len_context, n_unk,que_id,det_nsubj_idx,amod_nsubj_idx,rel_nsubj_inversed, new_sent+" <eos>",pos))
                    # we don't analyse the sentences features of wrong examples, only the columns form_alt and new_sent_alt are useful

                    output.append((pattern, constr_id,i, "wrong", form_alt,str_context_noun_ids, correct_number, cls_noun_num, cls_token_num, fst_noun_num,
                                   com_num, que_n_num, punct_before_nsubj, punct_after_nsubj, punct_before_verb, soft_attrs,
                                   homo_attrs, len_prefix, len_context, n_unk,que_id,det_nsubj_idx,amod_nsubj_idx,rel_nsubj_inversed, new_sent_alt+" <eos>",pos))

                constr_id += 1
                print()


    return output




def main():
    parser = argparse.ArgumentParser(description='Generating sentences based on patterns')

    parser.add_argument('--treebank', type=str, required=True, help='input file (in a CONLL column format)')
    parser.add_argument('--paradigms', type=str, required=True,
                        help="the dictionary of tokens and their morphological annotations")
    parser.add_argument('--vocab', type=str, required=True, help='(LM) Vocabulary to generate words from')
    parser.add_argument('--lm_data', type=str, required=False, help="path to LM data to estimate word frequencies")
    parser.add_argument('--output', type=str, required=True, help="prefix for generated text and annotation data")
 
    args = parser.parse_args()

    print("* Loading trees")
    # trees is a list of DependencyTree objets
    trees = depTree.load_trees_from_conll(args.treebank)
    print("# {0} parsed trees loaded".format(len(trees)))

    paradigms = read_paradigms(args.paradigms)
    vocab = load_vocab(args.vocab)
    data = collect_agreement(trees,paradigms,["Number"],vocab,4)
    print("# In total, {0} agreement sentences collected".format(int(len(data) / 2)))

    data = pd.DataFrame(data, columns=["pattern","constr_id","sent_id", "class", "form","str_context_noun_ids", "correct_number","cls_noun_num",
                                   "cls_token_num", "fst_noun_num","com_num",  "que_n_num",
                                   "punct_before_nsubj", "punct_after_nsubj", "punct_before_verb", "soft_attrs","homo_attrs",
                                   "len_prefix", "len_context", "n_unk","que_id","det_nsubj_idx","amod_nsubj_idx","rel_nsubj_inversed","sent","pos"])

    if args.lm_data:
        freq_dict = vocab_freqs(args.lm_data + "/train.txt", vocab)
        data["freq"] = data["form"].map(freq_dict)
        fields = ["pattern","constr_id","sent_id", "class", "form", "str_context_noun_ids", "correct_number","cls_noun_num",
                                   "cls_token_num", "fst_noun_num","com_num", "que_n_num",
                                   "punct_before_nsubj", "punct_after_nsubj", "punct_before_verb", "soft_attrs","homo_attrs",
                                   "len_prefix", "len_context", "freq","n_unk","que_id","det_nsubj_idx","amod_nsubj_idx","rel_nsubj_inversed","sent","pos"]
    else:
        fields = ["pattern","constr_id","sent_id", "class", "form", "str_context_noun_ids", "correct_number","cls_noun_num",
                                   "cls_token_num", "fst_noun_num","com_num",  "que_n_num",
                                   "punct_before_nsubj", "punct_after_nsubj", "punct_before_verb", "soft_attrs","homo_attrs",
                                   "len_prefix", "len_context", "n_unk","que_id","det_nsubj_idx","amod_nsubj_idx","rel_nsubj_inversed","sent","pos"]

    data[fields].to_csv(args.output  + ".tab", sep="\t", index=False)
    with open(args.output + ".txt","w") as txt:
        txt.write("\n".join(data['sent'].tolist()))



if __name__ == "__main__":
    main()
