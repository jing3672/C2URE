"""Client-local randomized causal transport on the actually trained graph.

Unlike server-anchor probes, these interventions are executed on each client's
current local graph, so a data-poisoning trigger is inside the intervention
support.  Only a fixed-length causal response summary is returned to the
server; no graph, target class, trigger template, or clean counterpart is used.
"""

import copy
import itertools
import math

import numpy as np
import torch
import torch.nn.functional as F


def _outputs(model, x, adj):
    logits = model(x, adj)
    centered = logits - logits.mean(dim=1, keepdim=True)
    energy = -torch.logsumexp(logits, dim=1)
    return centered, energy


def _local_graph_worlds(x, adj, budget, seed):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    gate = torch.rand(x.shape[0], generator=generator, dtype=x.dtype)
    gate = gate.to(x.device)
    gate = gate * (float(budget) / gate.mean().clamp(min=1e-6))
    gate = gate.clamp(max=1.0)
    permutation = torch.randperm(
        x.shape[0], generator=generator).to(x.device)
    x_do = x + gate[:, None] * (x[permutation] - x)
    keep = (1.0 - gate[:, None]) * (1.0 - gate[None, :])
    adj_do = adj * keep
    return ((x, adj), (x, adj_do), (x_do, adj), (x_do, adj_do))


@torch.no_grad()
def causal_node_transport(reference_model, client_model, x, adj, trials=4,
                          budget=0.2, seed=0, device="cpu"):
    """Return node-level graph-mediated update effects and CF teacher logits.

    The estimand is the change in the client update's centered-logit effect
    after a randomized joint intervention on structure and attributes:

        tau_v = E_w ||U_v(do(A, X; w)) - U_v(G)||_2,
        U_v(g) = Z_v(theta_i, g) - Z_v(theta_{t-1}, g).

    No class is singled out.  The counterfactual teacher is the previous
    global model averaged over the randomized intervention worlds.
    """
    reference_model = reference_model.to(device).eval()
    client_model = client_model.to(device).eval()
    x = x.to(device)
    adj = adj.to(device)
    factual_ref, _ = _outputs(reference_model, x, adj)
    factual_client, _ = _outputs(client_model, x, adj)
    factual_update = factual_client - factual_ref
    effects = []
    cf_probabilities = []
    for trial in range(max(int(trials), 2)):
        worlds = _local_graph_worlds(
            x, adj, budget, int(seed) + trial * 104729)
        # Joint do(A, X) is world 3.  Randomization identifies the average
        # graph-mediated part of applying the local parameter update.
        x_do, adj_do = worlds[3]
        ref_do, _ = _outputs(reference_model, x_do, adj_do)
        client_do, _ = _outputs(client_model, x_do, adj_do)
        do_update = client_do - ref_do
        effects.append((do_update - factual_update).norm(dim=1))
        cf_probabilities.append(F.softmax(reference_model(x_do, adj_do), dim=1))
    effect = torch.stack(effects).mean(dim=0)
    factual_magnitude = factual_update.norm(dim=1)
    probability_worlds = torch.stack(cf_probabilities)
    teacher_probability = probability_worlds.mean(dim=0)
    # Jensen-Shannon dispersion across intervention worlds.  Small values
    # mean the teacher's semantic prediction survives randomized do(G).
    teacher_js = (
        probability_worlds
        * (probability_worlds.clamp(min=1e-8).log()
           - teacher_probability.clamp(min=1e-8).log().unsqueeze(0))
    ).sum(dim=2).mean(dim=0)
    return effect, factual_magnitude, teacher_probability, teacher_js


@torch.no_grad()
def causal_node_transport_vectors(reference_model, client_model, x, adj,
                                  trials=4, budget=0.2, seed=0,
                                  device="cpu"):
    """Return the signed node-level do(G) x do(theta) response.

    Unlike :func:`causal_node_transport`, this estimator does not collapse the
    class-centred response to a norm.  Its direction can therefore expose a
    shared, previously unknown outcome shift without naming a target class.
    The accompanying reliability is one when intervention worlds agree on
    the response direction and approaches zero when they cancel.
    """
    reference_model = reference_model.to(device).eval()
    client_model = client_model.to(device).eval()
    x = x.to(device)
    adj = adj.to(device)
    factual_ref, _ = _outputs(reference_model, x, adj)
    factual_client, _ = _outputs(client_model, x, adj)
    factual_update = factual_client - factual_ref
    responses = []
    do_updates = []
    for trial in range(max(int(trials), 2)):
        x_do, adj_do = _local_graph_worlds(
            x, adj, budget, int(seed) + trial * 104729)[3]
        ref_do, _ = _outputs(reference_model, x_do, adj_do)
        client_do, _ = _outputs(client_model, x_do, adj_do)
        do_update = client_do - ref_do
        responses.append(factual_update - do_update)
        do_updates.append(do_update)
    response_worlds = torch.stack(responses)
    response = response_worlds.mean(dim=0)
    reliability = (response.norm(dim=1)
                   / response_worlds.norm(dim=2).mean(dim=0).clamp(min=1e-8))
    main_effect = 0.5 * (factual_update + torch.stack(do_updates).mean(dim=0))
    return response, main_effect, reliability


def _robust_positive_z(values):
    values = values.detach().float()
    center = values.median()
    mad = (values - center).abs().median()
    floor = 1e-3 * values.abs().median().clamp(min=1e-6)
    scale = torch.maximum(1.4826 * mad, floor)
    return torch.relu((values - center) / scale.clamp(min=1e-8))


def repair_client_by_causal_transport(reference_model, client_model, client,
                                      epochs=5, lr=0.01, trials=4,
                                      budget=0.2, node_z=2.5,
                                      distill_weight=1.0, seed=0,
                                      device="cpu"):
    """Construct a target-free counterfactual local update.

    Nodes are selected only when both applying the client update and changing
    the graph world expose an unusually large transport residual.  Their
    possibly poisoned labels are not trusted; the repaired update transports
    the previous global model's interventional soft prediction instead.  The
    remaining local labels retain the client's non-IID task information.
    """
    effect, magnitude, teacher_prob, teacher_js = causal_node_transport(
        reference_model, client_model, client.x, client.adj, trials=trials,
        budget=budget, seed=seed, device=device)

    train_idx = torch.as_tensor(client.train_idx, dtype=torch.long,
                                device=device)
    y = client.y.to(device)
    num_classes = teacher_prob.shape[1]
    confidence, cf_label = teacher_prob.max(dim=1)
    observed_probability = teacher_prob.gather(1, y[:, None]).squeeze(1)
    # A label is counterfactually inadmissible only when the randomized graph
    # worlds agree on another class with at least twice chance confidence.
    # Stability is defined relative to this client's own intervention-world
    # distribution, not a target class or a clean-graph reference.
    confidence_floor = min(0.5, 2.0 / max(int(num_classes), 2))
    train_js = teacher_js[train_idx]
    stability_ceiling = torch.quantile(train_js, 0.75)
    suspicious = ((cf_label != y)
                  & (confidence >= confidence_floor)
                  & (teacher_js <= stability_ceiling)
                  & (observed_probability < (1.0 / num_classes)))
    stable_idx = train_idx[~suspicious[train_idx]]
    suspect_idx = train_idx[suspicious[train_idx]]
    x = client.x.to(device)
    adj = client.adj.to(device)
    teacher_prob = teacher_prob.detach()
    repaired = copy.deepcopy(reference_model).to(device)
    optimizer = torch.optim.Adam(repaired.parameters(), lr=lr,
                                 weight_decay=5e-4)
    for epoch in range(max(int(epochs), 1)):
        repaired.train()
        logits = repaired(x, adj)
        losses = []
        if stable_idx.numel():
            losses.append(F.cross_entropy(logits[stable_idx], y[stable_idx]))
        if suspect_idx.numel():
            losses.append(float(distill_weight) * F.kl_div(
                F.log_softmax(logits[suspect_idx], dim=1),
                teacher_prob[suspect_idx], reduction="batchmean"))
        if not losses:
            break
        loss = sum(losses)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if hasattr(repaired, "_feats"):
        repaired._feats = None

    attack_idx = getattr(client, "attack_idx", None)
    overlap = None
    if attack_idx is not None and len(attack_idx):
        truth = torch.zeros_like(suspicious)
        truth[torch.as_tensor(attack_idx, dtype=torch.long,
                              device=device)] = True
        overlap = {
            "attack_node_recall_diagnostic": float(
                (suspicious & truth).sum() / truth.sum().clamp(min=1)),
            "selected_node_precision_diagnostic": float(
                (suspicious & truth).sum() / suspicious.sum().clamp(min=1)),
        }
    minimum_evidence = max(3, int(np.ceil(0.05 * train_idx.numel())))
    repair_ready = bool(suspect_idx.numel() >= minimum_evidence)
    diag = {
        "selected_nodes": int(suspicious.sum()),
        "selected_fraction": float(suspicious.float().mean()),
        "stable_train_nodes": int(stable_idx.numel()),
        "counterfactual_train_nodes": int(suspect_idx.numel()),
        "legacy_node_z_unused": float(node_z),
        "teacher_confidence_floor": float(confidence_floor),
        "teacher_stability_js_ceiling": float(stability_ceiling),
        "mean_train_teacher_confidence": float(confidence[train_idx].mean()),
        "repair_ready": repair_ready,
        "target_class_used": False,
        "clean_graph_used": False,
        "energy_used": False,
        "attack_identity_used_for_repair": False,
        "estimand": (
            "interventional label admissibility under E_do(A,X)[p_theta_t]"),
    }
    if overlap is not None:
        diag.update(overlap)
        diag["attack_identity_used_only_for_diagnostic_overlap"] = True
    return repaired, diag


def train_client_with_invariant_causal_gradients(
        reference_model, client, epochs=5, lr=0.01, worlds=4, budget=0.2,
        sign_agreement=0.5, seed=0, device="cpu", client_model=None,
        local_label_rescue=False, rescue_transport_z=1.5,
        clean_start_label_reconstruction=False,
        clean_start_reconstruction_epochs=2,
        continuous_causal_label_weight=False,
        clean_prototype_bundle=None, prototype_margin=0.0,
        prototype_policy=None, prototype_require_policy=False):
    """Train using only gradients stable across randomized do(A, X) worlds.

    For a causally suspicious client, the observed labels are accepted only
    where they agree with the previous global model's interventional
    consensus.  Other nodes use the target-free consensus distribution.
    Gradients are calculated independently in multiple graph worlds and a
    coordinate is transported to the server only if its direction is stable.
    """
    reference_model = reference_model.to(device).eval()
    x = client.x.to(device)
    adj = client.adj.to(device)
    y = client.y.to(device)
    train_idx = torch.as_tensor(client.train_idx, dtype=torch.long,
                                device=device)
    world_count = max(int(worlds), 3)

    with torch.no_grad():
        teacher_worlds = []
        teacher_hidden_worlds = []
        graph_worlds = []
        for trial in range(world_count):
            graph_world = _local_graph_worlds(
                x, adj, budget, int(seed) + trial * 104729)[3]
            graph_worlds.append(graph_world)
            teacher_logits = reference_model(*graph_world)
            teacher_worlds.append(F.softmax(teacher_logits, dim=1))
            hidden = (reference_model._feats[0]
                      if getattr(reference_model, "_feats", None) else None)
            teacher_hidden_worlds.append(
                F.normalize(hidden, dim=1) if hidden is not None else None)
        teacher_world_prob = torch.stack(teacher_worlds)
        teacher_prob = teacher_world_prob.mean(dim=0)
        teacher_confidence, teacher_label = teacher_prob.max(dim=1)
        world_confidence, world_label = teacher_world_prob.max(dim=2)
        label_stable = (world_label == teacher_label.unsqueeze(0)).all(dim=0)
        minimum_world_confidence = world_confidence.min(dim=0).values
        confidence_floor = min(0.5, 2.0 / max(teacher_prob.shape[1], 2))
        # Poisoning may append unlabeled trigger nodes to x/adj while client.y
        # still contains labels only for the original nodes.  Build the trust
        # mask on that common, label-supported prefix and never let appended
        # trigger nodes enter the supervised repair objective.
        label_count = min(int(teacher_label.numel()), int(y.numel()))
        valid_train_idx = train_idx[
            (train_idx >= 0) & (train_idx < label_count)]
        trusted = torch.zeros_like(teacher_label, dtype=torch.bool)
        teacher_supported = (
            (teacher_label[:label_count] == y[:label_count])
            & label_stable[:label_count]
            & (minimum_world_confidence[:label_count]
               >= confidence_floor))
        label_targets = y[:label_count].reshape(1, -1, 1).expand(
            world_count, -1, 1)
        teacher_label_probability = teacher_world_prob[
            :, :label_count].gather(2, label_targets).squeeze(2).mean(dim=0)
        teacher_label_agreement = (
            world_label[:, :label_count]
            == y[:label_count].unsqueeze(0)).float().mean(dim=0)
        teacher_support_score = (
            teacher_label_probability * teacher_label_agreement)
        rescued = torch.zeros(label_count, dtype=torch.bool, device=device)
        reconstructed = torch.zeros(
            label_count, dtype=torch.bool, device=device)
        prototype_rescued = torch.zeros(
            label_count, dtype=torch.bool, device=device)
        prototype_classes = []
        prototype_support_score = torch.zeros(
            label_count, dtype=torch.float32, device=device)
        reconstruction_support_score = torch.zeros(
            label_count, dtype=torch.float32, device=device)
        prototype_policy_available = prototype_policy is not None
        rescue_transport = None
        if (clean_prototype_bundle is not None
                and (not prototype_require_policy
                     or prototype_policy_available)
                and all(value is not None for value in teacher_hidden_worlds)):
            clean_prototypes, prototype_class_ids = clean_prototype_bundle
            clean_prototypes = F.normalize(
                clean_prototypes.to(device=device, dtype=torch.float32), dim=1)
            prototype_class_ids = torch.as_tensor(
                prototype_class_ids, dtype=torch.long, device=device)
            prototype_classes = [
                int(value) for value in prototype_class_ids.detach().cpu()]
            if (clean_prototypes.ndim == 2 and clean_prototypes.shape[0]
                    and clean_prototypes.shape[1]
                    == teacher_hidden_worlds[0].shape[1]):
                prototype_predictions = []
                prototype_margins = []
                prototype_hidden_worlds = list(teacher_hidden_worlds)
                if prototype_policy_available:
                    prototype_policy = prototype_policy.to(device).eval()
                    policy_x, policy_adj, _ = prototype_policy(x, adj)
                    reference_model(policy_x, policy_adj)
                    policy_hidden = (
                        reference_model._feats[0]
                        if getattr(reference_model, "_feats", None) else None)
                    if policy_hidden is not None:
                        prototype_hidden_worlds.append(
                            F.normalize(policy_hidden, dim=1))
                for hidden in prototype_hidden_worlds:
                    similarities = hidden[:label_count] @ clean_prototypes.T
                    top_count = min(2, int(similarities.shape[1]))
                    top_values, top_indices = torch.topk(
                        similarities, k=top_count, dim=1)
                    prototype_predictions.append(
                        prototype_class_ids[top_indices[:, 0]])
                    if top_count > 1:
                        prototype_margins.append(
                            top_values[:, 0] - top_values[:, 1])
                    else:
                        prototype_margins.append(torch.full_like(
                            top_values[:, 0], float("inf")))
                prototype_predictions = torch.stack(prototype_predictions)
                prototype_margins = torch.stack(prototype_margins)
                prototype_support_score = (
                    (prototype_predictions
                     == y[:label_count].unsqueeze(0)).float().mean(dim=0)
                    * torch.sigmoid(
                        4.0 * prototype_margins.min(dim=0).values))
                # A local label is rescued only when its clean-anchor semantic
                # class is unchanged in every randomized graph/feature world.
                # This is stricter than ordinary nearest-prototype matching and
                # remains independent of the attack target and trigger form.
                prototype_rescued = (
                    (prototype_predictions
                     == y[:label_count].unsqueeze(0)).all(dim=0)
                    & (prototype_margins.min(dim=0).values
                       >= float(prototype_margin)))
                prototype_rescued &= ~teacher_supported
        if bool(local_label_rescue) and client_model is not None:
            client_model = client_model.to(device).eval()
            factual_reference, _ = _outputs(reference_model, x, adj)
            factual_client, _ = _outputs(client_model, x, adj)
            factual_update = factual_client - factual_reference
            client_prob_worlds = []
            transport_worlds = []
            for graph_world in graph_worlds:
                reference_world, _ = _outputs(
                    reference_model, *graph_world)
                client_world, _ = _outputs(client_model, *graph_world)
                client_prob_worlds.append(F.softmax(
                    client_model(*graph_world), dim=1))
                transport_worlds.append(
                    (client_world - reference_world - factual_update)
                    .norm(dim=1))
            client_prob_worlds = torch.stack(client_prob_worlds)
            client_confidence, client_label = client_prob_worlds.max(dim=2)
            client_label_stable = (
                client_label[:, :label_count]
                == y[:label_count].unsqueeze(0)
            ).all(dim=0)
            client_confidence_stable = (
                client_confidence[:, :label_count].min(dim=0).values
                >= confidence_floor)
            rescue_transport = torch.stack(transport_worlds).mean(dim=0)
            rescue_transport_score = _robust_positive_z(
                rescue_transport[:label_count])
            # Only rescue labels that the global teacher does not already
            # support.  Stability under randomized graph/feature worlds and
            # low do(theta)xdo(A,X) transport jointly distinguish a genuine
            # Non-IID label from a graph-mediated shortcut.
            rescued = (
                ~teacher_supported
                & client_label_stable
                & client_confidence_stable
                & (rescue_transport_score <= float(rescue_transport_z)))
        reconstruction_transport = None
        if (bool(clean_start_label_reconstruction)
                and valid_train_idx.numel()):
            # Rebuild a client contribution from the last trusted global
            # model; never fine-tune or inspect the uploaded client model.
            # A local label that the global teacher does not yet know is
            # admitted only when a clean-start shadow relearns it in every
            # held-out do(A, X) world and its update-by-world transport is a
            # robust inlier.  This recovers rare Non-IID task information
            # without reopening the original parameter path.
            clean_start = copy.deepcopy(reference_model).to(device).train()
            clean_optimizer = torch.optim.Adam(
                clean_start.parameters(), lr=lr, weight_decay=5e-4)
            reconstruction_epochs = max(
                int(clean_start_reconstruction_epochs), 1)
            for reconstruction_epoch in range(reconstruction_epochs):
                # Label selection lives under the surrounding no-grad audit;
                # temporarily re-enable autograd only for this clean-start
                # shadow.  No gradient reaches the uploaded client model.
                with torch.enable_grad():
                    clean_optimizer.zero_grad()
                    world_losses = []
                    for trial in range(world_count):
                        reconstruction_world = _local_graph_worlds(
                            x, adj, budget,
                            int(seed) + 7000003
                            + reconstruction_epoch * 1000003
                            + trial * 104729)[3]
                        reconstruction_logits = clean_start(
                            *reconstruction_world)
                        world_losses.append(F.cross_entropy(
                            reconstruction_logits[valid_train_idx],
                            y[valid_train_idx]))
                    (sum(world_losses) / len(world_losses)).backward()
                    clean_optimizer.step()

            clean_start.eval()
            reconstruction_prob_worlds = []
            reconstruction_transport_worlds = []
            factual_reference, _ = _outputs(reference_model, x, adj)
            factual_reconstruction, _ = _outputs(clean_start, x, adj)
            factual_reconstruction_update = (
                factual_reconstruction - factual_reference)
            with torch.no_grad():
                for trial in range(world_count):
                    heldout_world = _local_graph_worlds(
                        x, adj, budget,
                        int(seed) + 17000033 + trial * 104729)[3]
                    reference_world, _ = _outputs(
                        reference_model, *heldout_world)
                    reconstruction_world, _ = _outputs(
                        clean_start, *heldout_world)
                    reconstruction_prob_worlds.append(F.softmax(
                        clean_start(*heldout_world), dim=1))
                    reconstruction_transport_worlds.append(
                        (reconstruction_world - reference_world
                         - factual_reconstruction_update).norm(dim=1))
            reconstruction_prob_worlds = torch.stack(
                reconstruction_prob_worlds)
            reconstruction_confidence, reconstruction_label = (
                reconstruction_prob_worlds.max(dim=2))
            reconstruction_label_stable = (
                reconstruction_label[:, :label_count]
                == y[:label_count].unsqueeze(0)).all(dim=0)
            reconstruction_confidence_stable = (
                reconstruction_confidence[:, :label_count]
                .min(dim=0).values >= confidence_floor)
            reconstruction_transport = torch.stack(
                reconstruction_transport_worlds).mean(dim=0)
            reconstruction_transport_score = _robust_positive_z(
                reconstruction_transport[:label_count])
            reconstruction_label_probability = (
                reconstruction_prob_worlds[:, :label_count].gather(
                    2, label_targets).squeeze(2).mean(dim=0))
            reconstruction_label_agreement = (
                reconstruction_label[:, :label_count]
                == y[:label_count].unsqueeze(0)).float().mean(dim=0)
            reconstruction_support_score = (
                reconstruction_label_probability
                * reconstruction_label_agreement
                * torch.exp(-reconstruction_transport_score.clamp(max=10.0)))
            reconstruction_candidates = (
                ~teacher_supported
                & reconstruction_label_stable
                & reconstruction_confidence_stable
                & (reconstruction_transport_score
                   <= float(rescue_transport_z)))

            # A malicious client can append many labels to one destination
            # class.  Bound newly reconstructed support per class by a
            # target-agnostic balanced-share budget and keep the lowest
            # transport candidates.  Existing teacher-supported labels are
            # never subject to this cap.
            class_count = max(int(teacher_prob.shape[1]), 1)
            per_class_cap = max(
                1, int(np.ceil(2.0 * valid_train_idx.numel()
                               / class_count)))
            for class_id in torch.unique(y[valid_train_idx]).tolist():
                class_candidates = valid_train_idx[
                    reconstruction_candidates[valid_train_idx]
                    & (y[valid_train_idx] == int(class_id))]
                if class_candidates.numel() > per_class_cap:
                    ordering = torch.argsort(
                        reconstruction_transport_score[class_candidates])
                    class_candidates = class_candidates[
                        ordering[:per_class_cap]]
                reconstructed[class_candidates] = True
        trusted[:label_count] = (
            teacher_supported | rescued | reconstructed | prototype_rescued)
        trusted_idx = valid_train_idx[trusted[valid_train_idx]]
        cf_idx = valid_train_idx[~trusted[valid_train_idx]]
        causal_label_weight = trusted.float()
        if bool(continuous_causal_label_weight):
            # Nodes that fail a hard causal certificate are not discarded.
            # Their local label receives at most half of the supervision,
            # with the remainder distilled from the interventional teacher.
            # Thus uncertain local evidence can preserve Non-IID information
            # but can never dominate the clean counterfactual target.
            soft_evidence = torch.maximum(
                teacher_support_score,
                torch.maximum(
                    reconstruction_support_score,
                    prototype_support_score)).clamp(min=0.0, max=1.0)
            valid_label_mask = torch.zeros(
                label_count, dtype=torch.bool, device=device)
            valid_label_mask[valid_train_idx] = True
            partial = (~trusted[:label_count]) & valid_label_mask
            label_prefix_weight = causal_label_weight[:label_count]
            label_prefix_weight[partial] = (
                0.5 * soft_evidence[partial])
            causal_label_weight[:label_count] = label_prefix_weight

    repaired = copy.deepcopy(reference_model).to(device)
    optimizer = torch.optim.Adam(repaired.parameters(), lr=lr,
                                 weight_decay=5e-4)
    agreement_history = []
    for epoch in range(max(int(epochs), 1)):
        repaired.train()
        epoch_worlds = [(x, adj)]
        for trial in range(world_count - 1):
            epoch_worlds.append(_local_graph_worlds(
                x, adj, budget,
                int(seed) + epoch * 1000003 + trial * 104729)[3])
        gradients_by_parameter = [[] for _ in repaired.parameters()]
        for x_world, adj_world in epoch_worlds:
            logits = repaired(x_world, adj_world)
            losses = []
            if bool(continuous_causal_label_weight):
                active_idx = valid_train_idx
                if active_idx.numel():
                    active_weight = causal_label_weight[active_idx]
                    local_ce = F.cross_entropy(
                        logits[active_idx], y[active_idx], reduction="none")
                    # Normalize trusted local-label mass inside each observed
                    # class.  A coalition cannot dominate merely by adding
                    # many labels to one destination class, while genuine
                    # rare Non-IID classes retain equal optimization voice.
                    balanced_local = torch.zeros_like(active_weight)
                    present_classes = torch.unique(y[active_idx])
                    for class_id in present_classes.tolist():
                        class_mask = y[active_idx] == int(class_id)
                        class_mass = active_weight[class_mask].sum()
                        if float(class_mass) > 1e-8:
                            balanced_local[class_mask] = (
                                active_weight[class_mask] / class_mass)
                    if float(balanced_local.sum()) > 1e-8:
                        losses.append(
                            (balanced_local * local_ce).sum()
                            / max(int(present_classes.numel()), 1))
                    distill_weight = 1.0 - active_weight
                    if float(distill_weight.sum()) > 1e-8:
                        node_kl = F.kl_div(
                            F.log_softmax(logits[active_idx], dim=1),
                            teacher_prob[active_idx], reduction="none").sum(dim=1)
                        losses.append(
                            (distill_weight * node_kl).sum()
                            / distill_weight.sum().clamp_min(1e-8))
            else:
                if trusted_idx.numel():
                    losses.append(F.cross_entropy(
                        logits[trusted_idx], y[trusted_idx]))
                if cf_idx.numel():
                    losses.append(F.kl_div(
                        F.log_softmax(logits[cf_idx], dim=1),
                        teacher_prob[cf_idx], reduction="batchmean"))
            if not losses:
                continue
            gradients = torch.autograd.grad(
                sum(losses), tuple(repaired.parameters()),
                retain_graph=False, create_graph=False)
            for bucket, gradient in zip(gradients_by_parameter, gradients):
                bucket.append(gradient.detach())
        optimizer.zero_grad()
        retained = total = 0
        for parameter, gradient_worlds in zip(
                repaired.parameters(), gradients_by_parameter):
            stacked = torch.stack(gradient_worlds)
            agreement = stacked.sign().float().mean(dim=0).abs()
            stable = agreement >= float(sign_agreement)
            parameter.grad = stacked.mean(dim=0) * stable
            retained += int(stable.sum())
            total += stable.numel()
        optimizer.step()
        agreement_history.append(retained / max(total, 1))
    if hasattr(repaired, "_feats"):
        repaired._feats = None

    attack_idx = getattr(client, "attack_idx", None)
    overlap = {}
    attack_tensor = torch.empty(0, dtype=torch.long, device=device)
    if attack_idx is not None and len(attack_idx):
        attack_tensor = torch.as_tensor(
            attack_idx, dtype=torch.long, device=device)
        attack_tensor = attack_tensor[
            (attack_tensor >= 0) & (attack_tensor < label_count)]
    if attack_idx is not None and len(attack_tensor):
        overlap = {
            "attack_nodes_using_counterfactual_diagnostic": float(
                (~trusted[attack_tensor]).float().mean()),
            "attack_identity_used_only_for_diagnostic_overlap": True,
        }
        if bool(local_label_rescue):
            overlap["attack_nodes_rescued_diagnostic"] = float(
                rescued[attack_tensor].float().mean())
        if clean_prototype_bundle is not None:
            overlap["attack_nodes_prototype_rescued_diagnostic"] = float(
                prototype_rescued[attack_tensor].float().mean())
    diag = {
        "repair_ready": True,
        "trusted_train_nodes": int(trusted_idx.numel()),
        "counterfactual_train_nodes": int(cf_idx.numel()),
        "mean_gradient_coordinate_retention": float(np.mean(
            agreement_history)),
        "sign_agreement_threshold": float(sign_agreement),
        "world_count": world_count,
        "confidence_floor": float(confidence_floor),
        "all_world_label_stability_required": True,
        "local_label_rescue_enabled": bool(local_label_rescue),
        "rescued_train_nodes": int(
            rescued[valid_train_idx].sum()) if rescued.numel() else 0,
        "rescue_transport_z": float(rescue_transport_z),
        "clean_start_label_reconstruction_enabled": bool(
            clean_start_label_reconstruction),
        "clean_start_reconstruction_epochs": int(
            max(int(clean_start_reconstruction_epochs), 1)),
        "continuous_causal_label_weight_enabled": bool(
            continuous_causal_label_weight),
        "mean_causal_label_weight": float(
            causal_label_weight[valid_train_idx].mean())
        if valid_train_idx.numel() else 0.0,
        "partial_causal_label_nodes": int((
            (causal_label_weight[valid_train_idx] > 0.0)
            & (causal_label_weight[valid_train_idx] < 1.0)).sum())
        if valid_train_idx.numel() else 0,
        "clean_start_reconstructed_train_nodes": int(
            reconstructed[valid_train_idx].sum())
        if reconstructed.numel() else 0,
        "clean_start_uses_uploaded_client_model": False,
        "prototype_rescue_enabled": bool(
            clean_prototype_bundle is not None),
        "prototype_rescued_train_nodes": int(
            prototype_rescued[valid_train_idx].sum())
        if prototype_rescued.numel() else 0,
        "prototype_margin": float(prototype_margin),
        "prototype_classes": prototype_classes,
        "prototype_policy_required": bool(prototype_require_policy),
        "prototype_policy_available": bool(prototype_policy_available),
        "target_class_used": False,
        "clean_graph_used": False,
        "energy_used": False,
        "attack_identity_used_for_repair": False,
        "estimand": "cross-do(A,X)-world invariant local gradient",
    }
    diag.update(overlap)
    return repaired, diag


def _summarize(values):
    values = values.detach().float().reshape(-1)
    quantiles = torch.quantile(
        values, torch.tensor([0.50, 0.75, 0.90, 0.95, 0.99],
                             device=values.device))
    return torch.cat([
        quantiles,
        values.mean().reshape(1),
        values.std(unbiased=False).reshape(1),
        values.max().reshape(1),
    ])


@torch.no_grad()
def client_local_causal_profile(reference_model, client_model, x, adj,
                                trials=4, budget=0.2, seed=0,
                                device="cpu", outcome="logits"):
    """Estimate local do(theta) x do(G) response distribution.

    World order is factual, do(A), do(X), do(A,X).  The profile includes the
    client-update-by-joint-graph contrast and the third-order A-by-X contrast,
    for both target-free energy and centered logits.
    """
    reference_model = reference_model.to(device).eval()
    client_model = client_model.to(device).eval()
    x = x.to(device)
    adj = adj.to(device)
    trial_profiles = []
    for trial in range(max(int(trials), 2)):
        worlds = _local_graph_worlds(
            x, adj, budget, int(seed) + trial * 104729)
        ref_outputs = [_outputs(reference_model, *world) for world in worlds]
        client_outputs = [_outputs(client_model, *world) for world in worlds]

        # Update effect U_w = Y(theta_i, G_w) - Y(theta_0, G_w).
        update_logits = [client_outputs[w][0] - ref_outputs[w][0]
                         for w in range(4)]
        update_energy = [client_outputs[w][1] - ref_outputs[w][1]
                         for w in range(4)]

        # Delta_G Delta_theta: joint graph intervention versus factual graph.
        joint_logits = update_logits[3] - update_logits[0]
        joint_energy = update_energy[3] - update_energy[0]
        # Delta_A Delta_X Delta_theta: non-additive attributed-graph pathway.
        interaction_logits = (update_logits[3] - update_logits[1]
                              - update_logits[2] + update_logits[0])
        interaction_energy = (update_energy[3] - update_energy[1]
                              - update_energy[2] + update_energy[0])
        components = [
            _summarize(joint_logits.norm(dim=1)),
            _summarize(interaction_logits.norm(dim=1)),
        ]
        if outcome == "hybrid":
            components.extend([
                _summarize(joint_energy.abs()),
                _summarize(interaction_energy.abs()),
            ])
        elif outcome != "logits":
            raise ValueError(f"unknown local causal outcome: {outcome}")
        trial_profiles.append(torch.cat(components))
    return torch.stack(trial_profiles)


def _robust_profile_scores(profiles):
    profiles = np.asarray(profiles, dtype=np.float64)
    center = np.median(profiles, axis=0, keepdims=True)
    mad = np.median(np.abs(profiles - center), axis=0, keepdims=True)
    # A relative floor prevents nearly constant coordinates from dominating.
    magnitude = np.median(np.abs(center), axis=1, keepdims=True)
    scale = np.maximum(1.4826 * mad, 1e-3 * np.maximum(magnitude, 1e-6))
    standardized = (profiles - center) / scale
    return np.sqrt(np.mean(standardized ** 2, axis=1)), standardized


def _rank_reliability(left, right):
    left_rank = np.argsort(np.argsort(left, kind="mergesort"),
                           kind="mergesort")
    right_rank = np.argsort(np.argsort(right, kind="mergesort"),
                            kind="mergesort")
    if left_rank.std() == 0.0 or right_rank.std() == 0.0:
        return 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _auc(scores, malicious_count):
    scores = np.asarray(scores, dtype=np.float64)
    m = int(malicious_count)
    if m <= 0 or m >= len(scores):
        return None
    wins = 0.0
    total = 0
    for positive in scores[:m]:
        for negative in scores[m:]:
            wins += float(positive > negative) + 0.5 * float(positive == negative)
            total += 1
    return float(wins / max(total, 1))


@torch.no_grad()
def compute_local_causal_scores(reference_model, client_models, clients,
                                trials=4, budget=0.2, seed=0,
                                device="cpu", outcome="logits"):
    """Compute profiles and robust scores without using attack identities."""
    per_client_trials = []
    for client, model in zip(clients, client_models):
        per_client_trials.append(client_local_causal_profile(
            reference_model, model, client.x, client.adj,
            trials=trials, budget=budget,
            seed=int(seed) + int(client.cid) * 1009, device=device,
            outcome=outcome).cpu())
    tensor = torch.stack(per_client_trials).numpy()
    profiles = tensor.mean(axis=1)
    scores, standardized = _robust_profile_scores(profiles)
    half = max(tensor.shape[1] // 2, 1)
    left_scores, _ = _robust_profile_scores(tensor[:, :half].mean(axis=1))
    right_scores, _ = _robust_profile_scores(tensor[:, half:].mean(axis=1))
    reliability = _rank_reliability(left_scores, right_scores)
    return scores, profiles, standardized, reliability


@torch.no_grad()
def audit_local_causal_energy(reference_model, client_models, clients,
                              trials=4, budget=0.2, seed=0, device="cpu",
                              malicious_count=0, outcome="logits"):
    """Return JSON-safe target-free local causal response diagnostics."""
    scores, profiles, standardized, reliability = compute_local_causal_scores(
        reference_model, client_models, clients, trials=trials,
        budget=budget, seed=seed, device=device, outcome=outcome)
    ranked = np.argsort(scores)[::-1]
    return {
        "name": "client-local-randomized-causal-transport-auditor",
        "estimand": (
            "Delta_G Delta_theta and Delta_A Delta_X Delta_theta "
            "on each current local graph"),
        "target_class_used": False,
        "trigger_template_used": False,
        "clean_graph_used": False,
        "full_graph_uploaded": False,
        "aggregation_changed": False,
        "execution_assumption": (
            "clients execute the prescribed randomized audit and report "
            "the fixed-length response summary honestly"),
        "trials": int(max(int(trials), 2)),
        "budget": float(budget),
        "profile_dimension": int(profiles.shape[1]),
        "outcome": outcome,
        "energy_used": bool(outcome == "hybrid"),
        "split_half_rank_reliability": reliability,
        "scores": [float(v) for v in scores],
        "ranked_clients": [int(v) for v in ranked],
        "standardized_profiles": standardized.tolist(),
        "diagnostic_auc": _auc(scores, malicious_count),
        "malicious_identity_used_for_scoring": False,
        "malicious_identity_used_only_for_diagnostic_auc": bool(
            malicious_count > 0),
    }


def _model_with_parameter_groups(reference_model, client_model, groups):
    """Apply only the named client-update groups to the reference model."""
    candidate = copy.deepcopy(reference_model)
    reference_state = reference_model.state_dict()
    client_state = client_model.state_dict()
    candidate_state = candidate.state_dict()
    groups = set(groups)
    for key, reference_value in reference_state.items():
        group = key.split(".", 1)[0]
        if group in groups and torch.is_floating_point(reference_value):
            candidate_state[key] = client_state[key].detach().to(
                device=reference_value.device, dtype=reference_value.dtype)
        else:
            candidate_state[key] = reference_value.detach().clone()
    candidate.load_state_dict(candidate_state)
    if hasattr(candidate, "_feats"):
        candidate._feats = None
    return candidate


@torch.no_grad()
def audit_layerwise_causal_transport(reference_model, client_models, clients,
                                     trials=4, budget=0.2, seed=0,
                                     device="cpu", malicious_count=0,
                                     outcome="logits"):
    """Exact parameter-group Shapley audit of do(G,X)-dependent updates.

    For a subset S of parameter groups, v_i(S) is the norm of the target-free
    Delta_G Delta_theta / Delta_A Delta_X Delta_theta profile obtained under
    do(theta = theta_ref + Delta_theta_i[S]).  Exact Shapley values allocate
    the full response across groups without assuming that GCN layers act
    additively.  This function is diagnostic only and never changes training.
    """
    reference_state = reference_model.state_dict()
    group_names = sorted({
        key.split(".", 1)[0] for key, value in reference_state.items()
        if torch.is_floating_point(value)
    })
    if not group_names:
        raise ValueError("layerwise causal audit found no floating groups")
    if len(group_names) > 4:
        raise ValueError(
            "exact layerwise causal audit supports at most four groups")

    subsets = []
    for size in range(len(group_names) + 1):
        subsets.extend(itertools.combinations(group_names, size))
    client_records = []
    shapley_matrix = []
    response_matrix = []
    factorial_n = math.factorial(len(group_names))
    for client, client_model in zip(clients, client_models):
        values = {}
        for subset in subsets:
            if not subset:
                values[subset] = 0.0
                continue
            candidate = _model_with_parameter_groups(
                reference_model, client_model, subset)
            profile = client_local_causal_profile(
                reference_model, candidate, client.x, client.adj,
                trials=trials, budget=budget,
                seed=int(seed) + int(client.cid) * 1009,
                device=device, outcome=outcome).mean(dim=0)
            values[subset] = float(profile.norm().item())

        shapley = []
        for group in group_names:
            others = [name for name in group_names if name != group]
            contribution = 0.0
            for size in range(len(others) + 1):
                for subset in itertools.combinations(others, size):
                    with_group = tuple(sorted((*subset, group)))
                    subset = tuple(sorted(subset))
                    coefficient = (
                        math.factorial(size)
                        * math.factorial(len(group_names) - size - 1)
                        / factorial_n)
                    contribution += coefficient * (
                        values[with_group] - values[subset])
            shapley.append(float(contribution))
        absolute = np.abs(np.asarray(shapley, dtype=np.float64))
        shares = absolute / max(float(absolute.sum()), 1e-12)
        response_matrix.append([values[(name,)] for name in group_names])
        shapley_matrix.append(shapley)
        client_records.append({
            "client": int(client.cid),
            "subset_response": {
                "+".join(subset) if subset else "reference": float(value)
                for subset, value in values.items()
            },
            "shapley": dict(zip(group_names, shapley)),
            "absolute_share": dict(zip(group_names, shares.tolist())),
            "dominant_group": group_names[int(np.argmax(shares))],
            "dominant_share": float(shares.max()),
            "full_response": float(values[tuple(group_names)]),
        })

    shapley_matrix = np.asarray(shapley_matrix, dtype=np.float64)
    absolute_shapley = np.abs(shapley_matrix)
    group_auc = {
        group: _auc(absolute_shapley[:, column], malicious_count)
        for column, group in enumerate(group_names)
    }
    dominant_shares = np.asarray(
        [record["dominant_share"] for record in client_records])
    m = int(malicious_count)
    return {
        "name": "layerwise-do-theta-by-do-gx-shapley-audit",
        "estimand": (
            "Shapley_l ||Delta_G Delta_theta[S] and "
            "Delta_A Delta_X Delta_theta[S]||"),
        "parameter_groups": group_names,
        "clients": client_records,
        "group_diagnostic_auc": group_auc,
        "mean_dominant_share_all": float(dominant_shares.mean()),
        "mean_dominant_share_malicious_diagnostic": (
            float(dominant_shares[:m].mean()) if 0 < m < len(clients)
            else None),
        "mean_dominant_share_benign_diagnostic": (
            float(dominant_shares[m:].mean()) if 0 < m < len(clients)
            else None),
        "target_class_used": False,
        "trigger_template_used": False,
        "clean_graph_used": False,
        "full_graph_uploaded": False,
        "aggregation_changed": False,
        "malicious_identity_used_for_scoring": False,
        "malicious_identity_used_only_for_diagnostic_summary": bool(
            0 < m < len(clients)),
        "outcome": outcome,
        "trials": int(max(int(trials), 2)),
        "budget": float(budget),
    }
