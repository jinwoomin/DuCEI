import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
import torch.nn.functional as F
from data.vocab import additional_tokens


class DotProductScoringEDModel(nn.Module):
    def __init__(self, args, device):
        super(DotProductScoringEDModel, self).__init__()
        self.args = args
        self.device = device

        self.hidden_size = args.hidden_size

        config = AutoConfig.from_pretrained(args.lm_name, attention_window=args.attention_window_size)
        config.attention_probs_dropout_prob = args.dropout_rate
        config.hidden_dropout_prob = args.dropout_rate
        
        self.lm_encoder = AutoModel.from_pretrained(args.lm_name, config=config)
        self.lm_encoder.resize_token_embeddings(config.vocab_size + len(additional_tokens), mean_resizing=args.mean_resizing)
            
        lm_hidden_size = self.lm_encoder.config.hidden_size

        self.mention_mlp = nn.Linear(lm_hidden_size * 2, self.hidden_size)
        self.candidates_mlp = nn.Linear(lm_hidden_size * 2, self.hidden_size)

    def forward(self, 
                input_ids, 
                token_type_ids, 
                attention_mask, 
                global_attention_mask, 
                mention_spans,
                candidates_spans, 
                candidates_mask, 
                gold_entity_positions=None):

        lm_outputs = self.lm_encoder(input_ids=input_ids,
                                     token_type_ids=token_type_ids,
                                     attention_mask=attention_mask,
                                     global_attention_mask=global_attention_mask)
        
        lm_output = lm_outputs.last_hidden_state
        # lm_output [batch, seq_len, lm_hidden_size]

        batch, _, lm_hidden_size = lm_output.size()

        mention_rep = lm_output.gather(1, mention_spans.unsqueeze(-1).expand(-1, -1, lm_hidden_size))
        # mention_spans [batch, 2] => [batch, 2, 1] => [batch, 2, lm_hidden_size]
        # mention_rep [batch, 2, lm_hidden_size]

        mention_rep = mention_rep.view(batch, -1)
        # mention_rep [batch, 2, lm_hidden_size] => [batch, 2 * lm_hidden_size]

        mention_rep = self.mention_mlp(mention_rep).unsqueeze(1)
        # mention_rep [batch, 2 * lm_hidden_size] => [batch, hidden_size] => [batch, 1, hidden_size]

        max_num_candidates = candidates_spans.size(1)

        candidates_spans = candidates_spans.view(batch, -1)
        # candidates_spans [batch, max_num_candidates * 2]

        candidates_rep = lm_output.gather(1, candidates_spans.unsqueeze(-1).expand(-1, -1, lm_hidden_size))
        # candidates_spans [batch, max_num_candidates * 2] => [batch, max_num_candidates * 2, 1]
        #                            => [batch, max_num_candidates * 2, lm_hidden_size]
        # candidates_rep [batch, max_num_candidates * 2, lm_hidden_size]

        candidates_rep = candidates_rep.view(batch, max_num_candidates, 2, -1).view(batch, max_num_candidates, -1)
        # candidates_rep [batch, max_num_candidates * 2, lm_hidden_size] => [batch, max_num_candidates, 2, lm_hidden_size]
        #                   => [batch, max_num_candidates, 2 * lm_hidden_size]

        candidates_rep = self.candidates_mlp(candidates_rep)
        # candidates_rep [batch, max_num_candidates, 2 * lm_hidden_size] => [batch, max_num_candidates, hidden_size]

        logits = torch.bmm(mention_rep, candidates_rep.transpose(1, 2)).squeeze(1)
        # logits [batch, 1, max_num_candidates] => [batch, max_num_candidates]

        logits.masked_fill_(candidates_mask.eq(0), float('-inf'))

        if gold_entity_positions == None:  # Evaluation
            pred_ids = torch.argmax(logits, dim=-1).tolist()
            # pred_ids [batch]

            probs = torch.softmax(logits, dim=-1).tolist()
            num_candidates = candidates_mask.sum(1).tolist()
            # num_candidates [batch]
            
            return pred_ids, probs, num_candidates

        # Training
        loss = F.cross_entropy(logits, gold_entity_positions)
        return loss


class CandidateClassificationEDModel(nn.Module):
    def __init__(self, args, device):
        super(CandidateClassificationEDModel, self).__init__()
        self.args = args
        self.device = device

        config = AutoConfig.from_pretrained(args.lm_name, attention_window=args.attention_window_size)
        config.attention_probs_dropout_prob = args.dropout_rate
        config.hidden_dropout_prob = args.dropout_rate

        self.lm_encoder = AutoModel.from_pretrained(args.lm_name, config=config)
        self.lm_encoder.resize_token_embeddings(config.vocab_size + len(additional_tokens), mean_resizing=args.mean_resizing)

        lm_hidden_size = self.lm_encoder.config.hidden_size

        self.classifier = nn.Linear(lm_hidden_size * 4 if self.args.concat_mention else lm_hidden_size * 2, 1)

    def forward(self, 
                input_ids, 
                token_type_ids, 
                attention_mask, 
                global_attention_mask, 
                mention_spans,
                candidates_spans, 
                candidates_mask, 
                gold_entity_positions=None):

        lm_outputs = self.lm_encoder(input_ids=input_ids,
                                     token_type_ids=token_type_ids,
                                     attention_mask=attention_mask,
                                     global_attention_mask=global_attention_mask)
        lm_output = lm_outputs.last_hidden_state
        # lm_output [batch, seq_len, lm_hidden_size]

        batch, _, lm_hidden_size = lm_output.size()

        max_num_candidates = candidates_spans.size(1)

        candidates_spans = candidates_spans.view(batch, -1)
        # candidates_spans [batch, max_num_candidates * 2]

        candidates_rep = lm_output.gather(1, candidates_spans.unsqueeze(-1).expand(-1, -1, lm_hidden_size))
        # candidates_spans [batch, max_num_candidates * 2] => [batch, max_num_candidates * 2, 1]
        #                            => [batch, max_num_candidates * 2, lm_hidden_size]
        # candidates_rep [batch, max_num_candidates * 2, lm_hidden_size]

        candidates_rep = candidates_rep.view(batch, max_num_candidates, 2, -1).view(batch, max_num_candidates, -1)
        # candidates_rep [batch, max_num_candidates * 2, lm_hidden_size] => [batch, max_num_candidates, 2, lm_hidden_size]
        #                   => [batch, max_num_candidates, 2 * lm_hidden_size]

        if self.args.concat_mention:
            mention_rep = lm_output.gather(1, mention_spans.unsqueeze(-1).expand(-1, -1, lm_hidden_size))
            # mention_spans [batch, 2] => [batch, 2, 1] => [batch, 2, lm_hidden_size]
            # mention_rep [batch, 2, lm_hidden_size]

            mention_rep = mention_rep.view(batch, -1).unsqueeze(1)
            # mention_rep [batch, 2, lm_hidden_size] => [batch, 2 * lm_hidden_size] => [batch, 1, 2 * lm_hidden_size]
            
            mention_rep = mention_rep.repeat(1, max_num_candidates, 1)
            # mention_rep [batch, 1, hidden_size] => [batch, max_num_candidates,2 * lm_hidden_size]

            combined_mlp = torch.cat([mention_rep, candidates_rep], dim=-1)
            logits = self.classifier(combined_mlp).squeeze(-1)
        else:
            logits = self.classifier(candidates_rep).squeeze(-1)
        # logits [batch, max_num_candidates, 1] => [batch, max_num_candidates]

        if gold_entity_positions == None:  # evaluation
            logits.masked_fill_(candidates_mask.eq(0), float('-inf'))

            if self.args.scoring == "sigmoid":
                probs = torch.sigmoid(logits)
            elif self.args.scoring == "softmax":
                probs = torch.softmax(logits, dim=-1)

            pred_ids = torch.argmax(probs, dim=-1).tolist()
            # pred_ids [batch]

            probs = probs.tolist()

            num_candidates = candidates_mask.sum(1).tolist()
            # num_candidates [batch]

            return pred_ids, probs, num_candidates

        # Training
        if self.args.scoring == "sigmoid":
            labels = torch.full(candidates_mask.size(), 0).type_as(logits)
            labels.scatter_(1, gold_entity_positions.unsqueeze(-1), 1)
            # labels [batch, max_num_candidates]

            if self.args.label_smooth > 0.0:
                epsilon = self.args.label_smooth
                # answer position 1 - epsilon + (epsilon / max_num_candidates)
                # remain position epsilon / max_num_candidates
                
                num_candidates = candidates_mask.sum(dim=1, keepdim=True)
                # num_candidates [batch, 1]
                
                labels = labels * (1 - epsilon) + (epsilon / num_candidates)
            
            loss = F.binary_cross_entropy_with_logits(logits.view(-1), labels.view(-1), reduction="none")
            loss = loss.masked_select(candidates_mask.view(-1).bool()).sum()
            loss = loss / candidates_mask.sum().type_as(logits)

        elif self.args.scoring == "softmax":
            logits.masked_fill_(candidates_mask.eq(0), float('-inf'))
            if self.args.label_smooth > 0.0:
                eps = self.args.label_smooth
                
                log_probs = F.log_softmax(logits, dim=-1)
                log_probs = log_probs.masked_fill(candidates_mask.eq(0), 0.0)
                # [batch, max_num_candidates]
                
                num_candidates = candidates_mask.sum(dim=1, keepdim=True)
                # num_candidates [batch, 1]

                smooth_target = torch.full_like(log_probs, eps)
                smooth_target = smooth_target / num_candidates
                # smooth_target = (self.args.label_smooth / num_candidates).expand_as(log_probs)
                # smooth_target [batch, max_num_candidates]

                smooth_target.scatter_(1, gold_entity_positions.unsqueeze(-1), 1 - eps + (eps / num_candidates))
                # scatter_addition = (1 - self.args.label_smooth + (self.args.label_smooth / num_candidates))
                # scatter_addition [batch, 1]
                # smooth_target = smooth_target.scatter(-1, gold_entity_positions.unsqueeze(-1), scatter_addition)
                
                smooth_target = smooth_target * candidates_mask
                # log_probs = log_probs * candidates_mask
                
                per_sample_loss = -(log_probs * smooth_target).sum(dim=-1) / candidates_mask.sum(dim=-1)
                loss = per_sample_loss.mean()

                # loss = -torch.sum(log_probs * smooth_target) / torch.sum(candidates_mask)
                # losses = -(log_probs * smooth_target)
                # losses [batch, max_num_candidates]
                # loss = losses.masked_select(candidates_mask.bool()).mean()
            else:
                loss = F.cross_entropy(logits, gold_entity_positions)

        assert loss is not None

        return loss


class ExtractiveQAEDModel(nn.Module):
    def __init__(self, args, device):
        from transformers.models.longformer.modeling_longformer import LongformerForQuestionAnswering

        super(ExtractiveQAEDModel, self).__init__()
        self.args = args
        self.device = device

        self.hidden_size = args.hidden_size

        config = AutoConfig.from_pretrained(args.lm_name, attention_window=args.attention_window_size)
        config.attention_probs_dropout_prob = args.dropout_rate
        config.hidden_dropout_prob = args.dropout_rate
        
        self.lm_encoder = LongformerForQuestionAnswering.from_pretrained(args.lm_name, config=config)
        self.lm_encoder.resize_token_embeddings(config.vocab_size + len(additional_tokens), mean_resizing=args.mean_resizing)
                
    def forward(self, 
                input_ids, 
                token_type_ids, 
                attention_mask, 
                global_attention_mask, 
                mention_spans,
                candidates_spans, 
                candidates_mask, 
                gold_entity_positions=None):

        start_positions, end_positions = None, None
        if gold_entity_positions is not None:
            batch = gold_entity_positions.size(0)

            indices = gold_entity_positions.view(-1, 1, 1).expand(-1, -1, candidates_spans.size(-1))
            # indices [batch, 1, 1] => [batch, 1, 2]

            gold_entity_span_positions = candidates_spans.gather(1, indices).view(batch, -1)
            # gold_entity_span_positions [batch, 1, 2] => [batch, 2]

            start_positions, end_positions = gold_entity_span_positions.split(1, dim=-1)
            # start_positions, end_positions [batch, 1]

            start_positions = start_positions.squeeze(-1)
            end_positions = end_positions.squeeze(-1)
            # start_positions, end_positions [batch, 1] => [batch]

        lm_outputs = self.lm_encoder(input_ids=input_ids,
                                     token_type_ids=token_type_ids,
                                     attention_mask=attention_mask,
                                     global_attention_mask=global_attention_mask,
                                     start_positions=start_positions,
                                     end_positions=end_positions)

        if gold_entity_positions == None:  # evaluation:
            start_logits, end_logits = lm_outputs.start_logits, lm_outputs.end_logits
            # start_logits, end_logits [batch, seq_len]

            start_probs = torch.softmax(start_logits, dim=-1)
            end_probs = torch.softmax(end_logits, dim=-1)
            # start_probs, end_probs [batch, seq_len]

            candidates_start_positions = candidates_spans[:, :, 0]
            candidates_end_positions = candidates_spans[:, :, 1]
            # candidates_start_positions, candidates_end_positions [batch, max_num_candidates]

            candidates_start_probs = start_probs.gather(1, candidates_start_positions)
            candidates_end_probs = end_probs.gather(1, candidates_end_positions)
            # candidates_start_probs, candidates_end_probs [batch, max_num_candidates]

            candidates_joint_probs = candidates_start_probs * candidates_end_probs
            # candidates_joint_probs [batch, max_num_candidates]
            
            candidates_joint_probs.masked_fill_(candidates_mask.eq(0), 0.0)  # .masked_fill_(candidates_mask.eq(0), -100000.0)
            pred_ids = torch.argmax(candidates_joint_probs, dim=-1).tolist()  # pred_ids [batch]
            
            probs = candidates_joint_probs.tolist()
            
            num_candidates = candidates_mask.sum(1).tolist()  # num_candidates [batch]
            
            return pred_ids, probs, num_candidates

        return lm_outputs.loss
