def evaluate_ed_InKBF1(gold_entities, candidates_list, pred_ids, num_candidates):
    num_mentions_with_candidates = 0
    num_mentions = 0
    matched = 0
        
    for gold_entity, candidates, pred_idx, num_candidate in zip(gold_entities, candidates_list, pred_ids, num_candidates):
        num_mentions += 1
        
        if num_candidate == 0:
            assert candidates == []
            continue

        num_mentions_with_candidates += 1

        pred_entity = candidates[pred_idx]
        if gold_entity == pred_entity:
            matched += 1

    precision = 100. * matched / num_mentions_with_candidates
    recall = 100. * matched / num_mentions
    f1 = 2.0 * precision * recall / (precision + recall)

    return precision, recall, f1
