class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07, unbiased=False):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.unbiased = unbiased

    def forward(self, features, labels=None, mask=None, k=0, weight=None, target_mask=None, target_index=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute weight mask
        if weight is not None:
            weight_mask = torch.ones_like(labels).float()
            for i in range(len(weight)):
                weight_mask[labels == i] = weight[i]
            weight_mask = weight_mask.repeat(anchor_count, 1).squeeze()

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask

        if self.unbiased:
            # only use negative with different label and itself
            exp_logits = exp_logits * (1 - mask)
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + torch.exp(logits))
        else:
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # KCL
        if k > 0:
            # randomly choose K positives
            mask_pos_view = torch.eye(batch_size, dtype=torch.float32).to(device)
            mask_pos_view = mask_pos_view.repeat(anchor_count, contrast_count)
            mask_pos_view = mask_pos_view * logits_mask

            # for i in range(mask.size(0)):
            #     same_class_idx = torch.nonzero(mask[i])
            #
            #     if k > same_class_idx.size(0):
            #         same_class_idx_sampled = same_class_idx
            #     else:
            #         idx = random.sample(range(same_class_idx.size(0)), k)
            #         same_class_idx_sampled = same_class_idx[idx]
            #
            #     mask_pos_view[i][same_class_idx_sampled] = 1

            # all_pos_idxs = mask.view(-1).nonzero().view(-1)
            # num_pos_per_anchor = mask.sum(1)
            # num_pos_cum = num_pos_per_anchor.cumsum(0).roll(1)
            # num_pos_cum[0] = 0
            # rand = torch.rand([mask.size(0), k], device=mask.device)
            # idxs = ((rand * num_pos_per_anchor.view(-1, 1)).floor() + num_pos_cum.view(-1, 1)).long()
            # sampled_pos_idxs = all_pos_idxs[idxs.view(-1)]
            # mask_pos_view.view(-1)[sampled_pos_idxs] = 1

            mask_copy = mask.clone()
            # only sample from same class but not augmentation view
            mask_copy = mask_copy * (1 - mask_pos_view)
            mask_copy = mask_copy
            # add all targets to the mask
            if target_index is not None:
                target_mask_all = target_index.unsqueeze(0).repeat(mask.size(0), 2)
                assert target_mask_all.size(0) == mask.size(0) and target_mask_all.size(1) == mask.size(1)
                mask_pos_view[(mask_copy * target_mask_all).nonzero(as_tuple=True)] = 1
                mask_copy[target_mask_all.nonzero(as_tuple=True)] = 0

            # sample K from batch other than targets
            for i in range(k):
                all_pos_idxs = mask_copy.view(-1).nonzero().view(-1)
                num_pos_per_anchor = mask_copy.sum(1)
                num_pos_cum = num_pos_per_anchor.cumsum(0).roll(1)
                num_pos_cum[0] = 0
                rand = torch.rand(mask_copy.size(0), device=mask_copy.device)
                idxs = ((rand * num_pos_per_anchor).floor() + num_pos_cum).long()
                idxs = idxs[num_pos_per_anchor.nonzero().view(-1)]
                sampled_pos_idxs = all_pos_idxs[idxs.view(-1)]
                mask_pos_view.view(-1)[sampled_pos_idxs] = 1
                mask_copy.view(-1)[sampled_pos_idxs] = 0

            mean_log_prob_pos = (mask_pos_view * log_prob).sum(1) / mask_pos_view.sum(1)
        else:
            # compute mean of log-likelihood over positive
            mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        if weight is not None:
            mean_log_prob_pos = weight_mask * mean_log_prob_pos

        # loss
        if target_mask is not None:
            # mask out the loss with target as anchor
            target_mask = target_mask.repeat(anchor_count)
            mean_log_prob_pos = mean_log_prob_pos * target_mask
            loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
            loss = loss.view(anchor_count, batch_size).sum() / target_mask.sum()
        else:
            loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
            loss = loss.view(anchor_count, batch_size).mean()

        return loss