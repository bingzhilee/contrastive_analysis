# -*- coding: utf-8 -*-

""" extract French subject *que(obj)* verb/aux agreement test set and collect relevant statistics
 specifically for fix pattern with 5 tokens between subj and v/aux: nsubj ADP NOUN que PRON VERB target-v
 extract candidates show 882 examples with UNK tokens between

 Remark:
 - don't need any other statistics info, only the len_prefix, len_context
 - only need correct txt, maybe tab?
 - It does not require that the alt form in vocab
 


 """

import depTree
import json
import argparse
import pandas as pd
from collections import Counter,defaultdict
from agreement_utils import grep_complex_rel_pattern,grep_obj_rel_pattern, grep_subj_rel_pattern,match_features,ltm_to_word,get_alt_form,read_paradigms, load_vocab, vocab_freqs

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

def plurality(morph):

    if "Number=Plur" in morph:
        return "plur"
    elif "Number=Sing" in morph:
        return "sing"
    else:
        return "none"


def match_subj_v_agreement(l, r, ltm_paradigms, vocab):
    if (not "Number" in l.morph) or (not "Number" in r.morph) or plurality(l.morph) != plurality(r.morph):
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
        if ('que', 'obj') not in context_w_dep:
            breakpoint()
        que_id = context_idx[context_w_dep.index(("que", "obj"))]
    elif "qu'" in context_word_obj:
        que_id = context_idx[context_w_dep.index(("qu'", "obj"))]
    # if no que-obj in the context, skip
    else:
        que_id = None

    # compute if rel_nsubj is inversed
    # index of the head of the relative clause(VERB: direct child of nsubj)
    if "acl:relcl" not in context_dep:
        # print("que-obj present, but no acl:relcl dependency label: ")
        rel_nsubj_inversed = "none"
        # case cop: "comte , tout ambitieux qu' il était , pourrait"
        # ['punct', 'advmod', 'amod', 'obj', 'nsubj', 'cop', 'punct']
    else:
        rel_verb_id = context_idx[context_dep.index("acl:relcl")]
        # nsubj:caus, motifs que le roi fait valoir
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
    # ses longs cheveux noirs, que j
    if child_l_det_ids:
        det_nsubj_idx = "_".join(child_l_det_ids)
    if child_l_adj_4w_ids:
        amod_nsubj_idx = "_".join(child_l_adj_4w_ids)

    return que_id,rel_nsubj_inversed,det_nsubj_idx,amod_nsubj_idx

def fix_pattern(tree,l,r,que_id,vocab):
    nsubj_id = l.index  # ud index
    v_id = r.index  # ud index
    v = r.word
    nsubj = l.word
    que = tree.nodes[que_id - 1]# que_id is ud index
    sent = [n.word.replace(" ", "") for n in tree.nodes]
    if que_id - nsubj_id > 2: # at least 2 tokens between them
        # nodes/pos/words between nsubj and que
        nsubj2que_nodes = [n for n in tree.nodes[nsubj_id: que_id - 1] if n.pos not in ['PUNCT', 'ADJ', 'ADV', "X","SYM","INTJ"]]
        nsubj2que_pos = " ".join([n.pos for n in nsubj2que_nodes])
        #nsubj2que_words = " ".join([n.word for n in nsubj2que_nodes])
        
        # one intervened noun betqeen nsubj and que
        if nsubj2que_pos in ['ADP NOUN', 'ADP PROPN']:
            # nodes between que and verb
            que2v_nodes = [n for n in tree.nodes[que_id:v_id] if n.pos not in ['PUNCT', 'ADJ', 'ADV','X',"SYM","INTJ"]]
            que2v_pos = [n.pos for n in que2v_nodes]
            que2v_words = [n.word for n in que2v_nodes]
            
            # nsubj ADP NOUN que PRON VERB target-v
            if " ".join(que2v_pos[:2]) == 'PRON VERB' and not " ".join(
                    que2v_pos[:3]) == 'PRON VERB VERB' and not " ".join(
                que2v_pos[:4]) == 'PRON VERB ADP VERB':

                # new list of nodes from nsubj to verb(included)
                nodes_nsubj_v = [l] + nsubj2que_nodes+ [que] + que2v_nodes[:2] + [r]
                nodes_before_nsubj = tree.nodes[:nsubj_id-1]
                nodes_after_v = tree.nodes[v_id:]
                new_tree = nodes_before_nsubj + nodes_nsubj_v + nodes_after_v
                new_sent = " ".join([n.word.replace(" ", "") for n in new_tree])
                len_pref = len(nodes_before_nsubj + nodes_nsubj_v)-1
                len_cont = len(nodes_nsubj_v)-1
                len2_nsubj = len(nodes_before_nsubj)
                constr = " ".join([n.word for n in nodes_nsubj_v])

                in_vocab = all([n.word in vocab for n in nodes_nsubj_v])
                if not in_vocab:
                    return False

                # debugging
                if len2_nsubj + len_cont != len_pref:
                    print("len2_nsubj + len_cont != len_pref")
                    breakpoint()
                if new_sent.split()[len_pref] != r.word:
                    print("new_sent.split()[len_pref] != r.word")
                    breakpoint()
                if new_sent.split()[len2_nsubj] != l.word:
                    print("new_sent.split()[len2_nsubj] != l.word")
                    breakpoint()

                # computing cls_token_num
                context_num = [plurality(n.morph) for n in nodes_nsubj_v[:-1] if plurality(n.morph) != "none"]
                cls_token_num = context_num[-1] if context_num else "none"
                # computing attrs
                intervened_noun = nsubj2que_nodes[1]
                attr_num = plurality(intervened_noun.morph) if plurality(intervened_noun.morph) else "none"
                if attr_num != "none":
                    if attr_num != plurality(l.morph):
                        attr = True
                    else:
                        attr = False
                else:
                    attr = "none"
                # computing que python id in the new_sent
                que_py_id = len2_nsubj + 3
                # new pos
                new_poses = " ".join([n.pos for n in new_tree])
                #breakpoint()
                return constr,new_sent,len2_nsubj,len_cont,len_pref, cls_token_num, attr, que_py_id, new_poses

            else:
                return False





collective_words = ["plupart","moitié","bande", "dizaine", "ensemble", "foule", "horde", "millier",
                    "multitude", "majorité", "nombre", "ribambelle", "totalité", "troupeau"]

def collect_agreement(trees,  paradigms, vocab):
    output = []
    whole_constr = set()
    constr_id = 0
    ltm_paradigms = ltm_to_word(paradigms)  # best_paradigms_lemmas['be']['Aux'][morph]=='is'

    for tree in trees:
        for a in tree.arcs:
            if a.length() > 3:
                context_nodes, l, r = inside(tree, a)

                context_w_dep = [(n.word.lower(), n.dep_label) for n in
                                 tree.nodes[l.index:r.index - 1]]


                # we don't consider the word "plupart" as subject
                # # toujours annoté singulier, mais verbe s’accorde avec le dernier substantif
                # la plupart des hommes se souviennent … la plupart du monde ne se soucie
                if l.word.lower() in collective_words:
                    continue
                # make sure there is no "conjunction nominal subject"
                conj_pos = [n.pos for n in context_nodes if n.head_id == l.index and n.dep_label == "conj"]
                if "NOUN" in conj_pos or "PROPN" in conj_pos :
                    continue

                if l.word not in vocab or r.word not in vocab:
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
                    #if match_subj_v_agreement(l,r,ltm_paradigms,vocab) and ("que","obj") in context_w_dep or ("qu'","obj") in context_w_dep:
                    #    breakpoint()
                    continue



                que_index, rel_nsubj_inversed, det_nsubj_idx, amod_nsubj_idx = compute_mask_ids(l, r, context_nodes,tree)
                if que_index:
                    que_id = que_index
                else:
                    continue

                is_fixedPattern = fix_pattern(tree,l,r,que_id,vocab)
                if is_fixedPattern:
                    constr, new_sent, len2_nsubj, len_context, len_prefix, cls_token_num, attr, que_id, new_poses = is_fixedPattern
                else:
                    continue

                # rm duplicate contexts
                #constr = " ".join(str(n.word) for n in [l] + context_nodes + [r])
                if constr in whole_constr:
                    continue
                else:
                    whole_constr.add(constr)


                form = r.word
                correct_number = plurality(r.morph)

                output.append((pattern, constr_id, form, correct_number, cls_token_num, attr,
                               len2_nsubj, len_prefix, len_context, que_id, rel_nsubj_inversed, new_sent+" <eos>",new_poses))
                constr_id += 1


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
    # needed for original UD treebanks (e.g. French) which contain spans, e.g. 10-11
    # annotating mutlimorphemic words as several nodes in the tree
    # for t in trees:
    #    t.remerge_segmented_morphemes()
    
    paradigms = read_paradigms(args.paradigms)
    #d["is"] == ('be','AUX','Number=Sing|Mood=Ind|Tense=Pres|VerbForm=Fin',100)
    #print(paradigms["fait"])
    vocab = load_vocab(args.vocab)
    data = collect_agreement(trees,paradigms,vocab)

    print("# In total, {0} agreement sentences collected".format(int(len(data))))

    data = pd.DataFrame(data, columns=["pattern","constr_id", "form", "correct_number",
                                   "cls_token_num", "attr", "len2_nsubj",
                                   "len_prefix", "len_context", "que_id","rel_nsubj_inversed","sent","pos"])

    if args.lm_data:
        freq_dict = vocab_freqs(args.lm_data + "/train.txt", vocab)
        data["freq"] = data["form"].map(freq_dict)
        fields = ["pattern","constr_id", "form", "correct_number",
                                   "cls_token_num", "attr", "len2_nsubj",
                                   "len_prefix", "len_context", "que_id","freq", "rel_nsubj_inversed","sent","pos"]


    else:
        fields = ["pattern", "constr_id", "form", "correct_number",
                  "cls_token_num", "attr", "len2_nsubj",
                  "len_prefix", "len_context", "que_id",  "rel_nsubj_inversed", "sent", "pos"]

    data[fields].to_csv(args.output  + ".tab", sep="\t", index=False)
    with open(args.output + ".txt","w") as txt:
        txt.write("\n".join(data['sent'].tolist()))



if __name__ == "__main__":
    main()
