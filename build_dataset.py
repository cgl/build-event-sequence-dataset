import os
import re
from nltk.tokenize.util import regexp_span_tokenize


def read_relations(line, events_doc, corefs_doc, afters_doc, parents_doc):
    _, lid, event_ids = line.strip().split("\t")
    if line.startswith("@After"):
        afters_doc[lid] = event_ids.split(",")
    elif line.startswith("@Coreference"):
        corefs_doc[lid] = event_ids.split(",")
        for e_id in corefs_doc[lid]:
            events_doc[e_id]["coref"] = lid
    elif line.startswith("@Subevent"):
        parents_doc[lid] = event_ids.split(",")
    else:
        pass


def add_corefs_to_single_events(events, corefs):
    for doc_id in events.keys():
        index = len(corefs[doc_id])
        for event_id in events[doc_id].keys():
            if "coref" not in events[doc_id][event_id]:
                events[doc_id][event_id]["coref"] = "C%d" %index
                corefs[doc_id]["C%d" %index] = [event_id]
                index += 1


# brat_conversion 1b386c986f9d06fd0a0dda70c3b8ade9 E194	145,154	sentences Justice_Sentence Actual
def read_annotations(ann_file_tbf):
    events, corefs, afters, parents = {}, {}, {}, {}
    with open(ann_file_tbf) as ann_file:
        for line in ann_file:
            if line.startswith("#B"):
                doc_id = line.strip().split(" ")[-1]
                events[doc_id] = {}
                corefs[doc_id] = {}
                afters[doc_id] = {}
                parents[doc_id] = {}
            elif line.startswith("@"):
                read_relations(line, events[doc_id], corefs[doc_id], afters[doc_id],parents[doc_id])
            elif line.startswith("b"):
                _, _, event_id, offsets, nugget, event_type, realis = line.strip().split("\t")
                events[doc_id][event_id] = {"offsets": offsets,
                                            "nugget": nugget,
                                            "event_type": event_type,
                                            "realis": realis}
            else:
                pass
    add_corefs_to_single_events(events, corefs)
    return events, corefs, afters, parents


def use_context_words(ann_file_tbf, dataset="_train", window_size=4):
    """
    Builds two files one with positive examples and one with negative examples.
    Each event pair surrounded by context words lays in one line seperated with
    whitespace. Eg: c1 c2 e1 c3 c4 c5 c6 e2 c7 c8
    """

    events, corefs, afters, parents = read_annotations(ann_file_tbf)

    data_folder = os.path.join("data", "LDC2016E130_V5", "data", "all")
    positives, negatives = [], []
    for doc_id in events:
        for event_id in events[doc_id]:
            for ind, to_event_id in enumerate(events[doc_id]):
                if event_id == to_event_id:
                    continue
                linked_event_ids = [event_id, to_event_id]
                is_positive = linked_event_ids in afters[doc_id].values()
                if not is_positive and ind % 30 != 0:
                    continue
                with open(os.path.join(data_folder, doc_id+".txt")) as file:
                    text = file.read()
                    replacements = [(" author=", "_author="),
                                    (" datetime=", "_datetime="),
                                    (" id=", "_id="),
                                    (" alt=", "_alt="),
                                    ("doc id", "doc_id"),
                                    ("img src", "img_src"),
                                    ("a href", "a_href"),
                                    ("\n" , " "),
                                    #(" <", "_<"),
                                    #("> ", ">_"),
                    ]
                    for r in replacements:
                        text = text.replace(r[0], r[1])

                token_list = list(regexp_span_tokenize(text, r'\s'))
                ctx_word_list = [] # [doc_id,event_id, to_event_id]
                for i in range(2):
                    e_id = linked_event_ids[i]
                    event_offsets = tuple([int(a) for a in events[doc_id][e_id]["offsets"].split(",")])
                    try:
                        nugget_ind = token_list.index(event_offsets)
                    except ValueError:
                        try:
                            new_nugget_ind = [ind for ind,off in enumerate(token_list) if
                                              off[0] == event_offsets[0] or
                                              off[1] == event_offsets[1] or
                                              off[0]-1 == event_offsets[0] or
                                              off[0]+1 == event_offsets[0]
                                              ][0]
                            nugget_ind = new_nugget_ind
                        except IndexError:
                            print(is_positive, doc_id, events[doc_id][e_id]["nugget"],
                                  text[event_offsets[0]-5:event_offsets[1]+5])
                            if is_positive:
                                import ipdb; ipdb.set_trace()
                            continue
                    # found the nugget in the tokenized text
                    # index of the nugget in token list is nugget_ind
                    if i == 0:
                        ctx_word_list.append(doc_id)
                        ctx_word_list.append(event_id)
                        ctx_word_list.append(to_event_id)
                    for t_ind in range(nugget_ind-window_size, nugget_ind + window_size + 1):
                        if 0 > t_ind or t_ind >= len(token_list):
                            context_word = "pad"
                        else:
                            context_word = text[token_list[t_ind][0]:token_list[t_ind][1]]
                            # remove <post> <a href.. > etc
                            context_word = re.sub(r"<.*>", "", context_word)
                        ctx_word_list.append(context_word.strip('"\',.:“”'))

                if is_positive:
                    positives.append(" ".join(ctx_word_list))
                else:
                    negatives.append(" ".join(ctx_word_list))
    with open("seq_positives%s_%s.txt" % (dataset, window_size), "w") as file:
        file.write("\n".join(positives))
    with open("seq_negatives%s_%s.txt" % (dataset, window_size), "w") as file:
        file.write("\n".join(negatives[:len(positives)*2]))
    with open("seq_negatives%s_%s_all.txt" % (dataset, window_size), "w") as file:
        file.write("\n".join(negatives))


if __name__ == "__main__":
    training = "LDC2016E130_test.tbf"
    test = "LDC2016E130_test.tbf"
    evaluation = "Sequence_2017_test.tbf"
    training_plus_test = "Sequence_2017_training.tbf"
    ann_file_tbf = os.path.join("data", evaluation)
    use_context_words(ann_file_tbf, dataset="_eval")
    ann_file_tbf = os.path.join("data", test)
    use_context_words(ann_file_tbf, dataset="_test")
    ann_file_tbf = os.path.join("data", training)
    use_context_words(ann_file_tbf)
