"""C2URE core functions extracted verbatim from the experiment implementation."""

import copy
import itertools
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from local_causal_energy import causal_node_transport_vectors, client_local_causal_profile, compute_local_causal_scores, repair_client_by_causal_transport, train_client_with_invariant_causal_gradients

def train_node_model(model, x, adj, train_idx, y, epochs, lr, device="cpu",
                     optimizer_name="adam", val_idx=None,
                     select_best_on_val=False, training_seed=None):
    model = copy.deepcopy(model).to(device)
    model.train()
    if optimizer_name == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    elif optimizer_name == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=lr,
                              weight_decay=5e-4)
    else:
        raise ValueError(f"unknown local optimizer: {optimizer_name}")
    x = x.to(device)
    adj = adj.to(device)
    y = y.to(device)
    train_idx = torch.as_tensor(train_idx, dtype=torch.long, device=device)
    val_idx_t = None
    if val_idx is not None:
        val_idx_t = torch.as_tensor(
            val_idx, dtype=torch.long, device=device).reshape(-1)
    best_acc = -1.0
    best_state = None
    for epoch_idx in range(epochs):
        if training_seed is not None:
            epoch_seed = int(training_seed) + int(epoch_idx)
            random.seed(epoch_seed)
            np.random.seed(epoch_seed)
            torch.manual_seed(epoch_seed)
            torch.cuda.manual_seed_all(epoch_seed)
        model.train()
        opt.zero_grad()
        logits = model(x, adj)
        loss = F.cross_entropy(logits[train_idx], y[train_idx])
        loss.backward()
        opt.step()
        if (select_best_on_val and val_idx_t is not None
                and val_idx_t.numel() > 0):
            model.eval()
            with torch.no_grad():
                val_logits = model(x, adj)
                val_acc = float((
                    val_logits[val_idx_t].argmax(1) == y[val_idx_t]
                ).float().mean().item())
            if val_acc > best_acc:
                best_acc = val_acc
                best_state = copy.deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)
    if hasattr(model, "_feats"):
        model._feats = None
    if hasattr(model, "_causal_gate"):
        model._causal_gate = None
    return model


def _local_class_weights(labels, train_idx):
    """Inverse-frequency weights using only client-owned labels."""
    train_labels = labels[train_idx]
    class_count = int(labels.max().item()) + 1
    counts = torch.bincount(train_labels, minlength=class_count).float()
    return _class_weights_from_counts(counts), counts


def _class_weights_from_counts(counts):
    counts = counts.float()
    represented = counts > 0
    weights = torch.zeros_like(counts)
    if represented.any():
        weights[represented] = counts[represented].reciprocal()
        weights[represented] /= weights[represented].mean().clamp(min=1e-8)
    return weights


@torch.no_grad()
def _trusted_hidden_prototypes(reference_model, probe, device="cpu",
                               prototypes_per_class=1):
    """Build deterministic multi-modal class prototypes on trusted anchors.

    A single class mean can erase legitimate Non-IID sub-modes.  For more
    than one prototype per class, deterministic cosine k-means represents
    those modes without consuming client identity, attack labels, targets,
    trigger templates, or test nodes.
    """
    if probe is None:
        return None
    x, adj, labels, probe_idx = probe
    model = copy.deepcopy(reference_model).to(device).eval()
    x, adj, labels = x.to(device), adj.to(device), labels.to(device)
    probe_idx = torch.as_tensor(
        probe_idx, dtype=torch.long, device=device).reshape(-1)
    if probe_idx.numel() == 0:
        return None
    model(x, adj)
    hidden = model._feats[0] if getattr(model, "_feats", None) else None
    if hidden is None:
        return None
    prototypes, class_ids = [], []
    requested = max(int(prototypes_per_class), 1)
    for class_id in torch.unique(labels[probe_idx]).tolist():
        class_idx = probe_idx[labels[probe_idx] == int(class_id)]
        if not class_idx.numel():
            continue
        class_hidden = F.normalize(hidden[class_idx], dim=1)
        cluster_count = min(requested, int(class_hidden.shape[0]))
        if cluster_count == 1:
            class_prototypes = F.normalize(
                class_hidden.mean(dim=0), dim=0).unsqueeze(0)
        else:
            class_mean = F.normalize(class_hidden.mean(dim=0), dim=0)
            first = int(torch.argmax(class_hidden @ class_mean).item())
            centers = [class_hidden[first]]
            for _ in range(1, cluster_count):
                similarities = class_hidden @ torch.stack(centers).T
                nearest = similarities.max(dim=1).values
                centers.append(class_hidden[int(torch.argmin(nearest).item())])
            class_prototypes = torch.stack(centers)
            for _ in range(5):
                assignment = torch.argmax(
                    class_hidden @ class_prototypes.T, dim=1)
                updated = []
                for cluster_id in range(cluster_count):
                    members = class_hidden[assignment == cluster_id]
                    updated.append(
                        F.normalize(members.mean(dim=0), dim=0)
                        if members.numel() else class_prototypes[cluster_id])
                updated = torch.stack(updated)
                if torch.allclose(updated, class_prototypes, atol=1e-6):
                    class_prototypes = updated
                    break
                class_prototypes = updated
        prototypes.extend(list(class_prototypes))
        class_ids.extend([int(class_id)] * int(class_prototypes.shape[0]))
    if not prototypes:
        return None
    return torch.stack(prototypes), torch.as_tensor(
        class_ids, dtype=torch.long, device=device)


def train_trust_recovery_local(
        global_model, client, epochs, lr, optimizer_name="adam", device="cpu",
        training_seed=None, class_weights_override=None,
        balance_counts_override=None, augmentation=False,
        augmentation_edge_drop=0.20, augmentation_feature_mask=0.10,
        augmentation_weight=0.50, augmentation_consistency=0.25,
        sam=False, sam_rho=0.05):
    """Train one LCCT-retained client with aggregate-prior rebalancing.

    The optional class weights come from count vectors submitted by the
    LCCT-retained clients.  No server graph, server label, attack target, or
    simulator malicious identity is consumed.
    """
    model = copy.deepcopy(global_model).to(device)
    model.train()
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=5e-4)
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, weight_decay=5e-4)
    else:
        raise ValueError(f"unknown local optimizer: {optimizer_name}")

    x = client.x.to(device)
    adj = client.adj.to(device)
    labels = client.y.to(device)
    train_idx = torch.as_tensor(
        client.train_idx, dtype=torch.long, device=device).reshape(-1)
    class_weights, local_class_counts = _local_class_weights(
        labels, train_idx)
    balance_counts = local_class_counts
    if class_weights_override is not None:
        class_weights = class_weights_override.to(
            device=device, dtype=torch.float32)
        if balance_counts_override is not None:
            balance_counts = balance_counts_override.to(
                device=device, dtype=torch.float32)

    step_count = max(int(epochs), 1)
    for epoch_idx in range(int(epochs)):
        if training_seed is not None:
            epoch_seed = int(training_seed) + int(epoch_idx)
            random.seed(epoch_seed)
            np.random.seed(epoch_seed)
            torch.manual_seed(epoch_seed)
            torch.cuda.manual_seed_all(epoch_seed)
        optimizer.zero_grad()
        logits = model(x, adj)
        loss = F.cross_entropy(
            logits[train_idx], labels[train_idx], weight=class_weights)
        if augmentation:
            x_view = x
            if float(augmentation_feature_mask) > 0.0:
                feature_keep = (
                    torch.rand_like(x) >= float(augmentation_feature_mask))
                x_view = x * feature_keep.to(x.dtype)
            adj_view = adj.clone()
            if float(augmentation_edge_drop) > 0.0:
                upper = torch.triu(adj_view != 0, diagonal=1)
                row, col = torch.where(upper)
                if row.numel():
                    remove = (torch.rand(
                        row.numel(), device=adj.device)
                        < float(augmentation_edge_drop))
                    row, col = row[remove], col[remove]
                    adj_view[row, col] = 0
                    adj_view[col, row] = 0
            view_logits = model(x_view, adj_view)
            view_ce = F.cross_entropy(
                view_logits[train_idx], labels[train_idx],
                weight=class_weights)
            clean_probability = F.softmax(
                logits[train_idx].detach(), dim=1)
            consistency = F.kl_div(
                F.log_softmax(view_logits[train_idx], dim=1),
                clean_probability, reduction="batchmean")
            loss = (loss
                    + float(augmentation_weight) * view_ce
                    + float(augmentation_consistency) * consistency)
        loss.backward()
        perturbations = []
        if sam:
            grad_norm = torch.norm(torch.stack([
                parameter.grad.norm(p=2)
                for parameter in model.parameters()
                if parameter.grad is not None
            ]), p=2).clamp(min=1e-12)
            scale = float(sam_rho) / grad_norm
            with torch.no_grad():
                for parameter in model.parameters():
                    if parameter.grad is None:
                        continue
                    perturbation = parameter.grad * scale
                    parameter.add_(perturbation)
                    perturbations.append((parameter, perturbation))
            optimizer.zero_grad()
            sharp_logits = model(x, adj)
            sharp_loss = F.cross_entropy(
                sharp_logits[train_idx], labels[train_idx],
                weight=class_weights)
            sharp_loss.backward()
            with torch.no_grad():
                for parameter, perturbation in perturbations:
                    parameter.sub_(perturbation)
        optimizer.step()
    if hasattr(model, "_feats"):
        model._feats = None
    if hasattr(model, "_causal_gate"):
        model._causal_gate = None
    diagnostics = {
        "client": int(client.cid),
        "local_class_counts": [
            float(value) for value in local_class_counts.cpu()],
        "balance_class_counts": [
            float(value) for value in balance_counts.cpu()],
        "class_weights": [float(value) for value in class_weights.cpu()],
        "class_balance_rule": "inverse_retained_aggregate_frequency",
        "local_steps": int(step_count),
        "augmentation_enabled": bool(augmentation),
        "augmentation_edge_drop": float(augmentation_edge_drop),
        "augmentation_feature_mask": float(augmentation_feature_mask),
        "augmentation_weight": float(augmentation_weight),
        "augmentation_consistency": float(augmentation_consistency),
        "sam_enabled": bool(sam),
        "sam_rho": float(sam_rho),
    }
    return model, diagnostics


def _class_balanced_cross_entropy(logits, labels):
    """Cross entropy with equal total mass for every represented class."""
    labels = labels.reshape(-1)
    classes, counts = torch.unique(labels, return_counts=True)
    sample_weights = torch.zeros_like(labels, dtype=logits.dtype)
    for class_id, count in zip(classes, counts):
        sample_weights[labels == class_id] = 1.0 / count.to(logits.dtype)
    sample_weights = sample_weights / sample_weights.mean().clamp(min=1e-12)
    losses = F.cross_entropy(logits, labels, reduction="none")
    return (losses * sample_weights).mean()


def adaptive_server_utility_repair(
        model, x, adj, y, train_idx, monitor_idx=None, anchor_state=None,
        max_steps=50, patience=6, min_delta=1e-4, lr=0.001,
        w_align=1.0, max_class_drop=0.05,
        selection_mode="balanced-risk", device="cpu"):
    """Recover Non-IID utility with monitored, rollback-safe calibration.

    The trusted anchor set is split into a class-balanced calibration subset
    and a disjoint monitor subset. Optimization is selected by held-out
    class-balanced risk and stops after convergence; ``max_steps`` is only a
    safety cap, not a protocol-dependent epoch prescription. Suspicious
    client updates are never reintroduced.
    """
    repaired = copy.deepcopy(model).to(device)
    x = x.to(device)
    adj = adj.to(device)
    y = y.to(device)
    train_idx = torch.as_tensor(
        train_idx, dtype=torch.long, device=device).reshape(-1)
    monitor_idx = torch.as_tensor(
        [] if monitor_idx is None else monitor_idx,
        dtype=torch.long, device=device).reshape(-1)
    if train_idx.numel() == 0:
        return repaired, {"applied": False, "reason": "no_training_anchors"}
    if monitor_idx.numel() == 0:
        # Fail closed when an external monitor set is unavailable: reserve
        # one deterministic example per represented class where possible.
        reserved = []
        retained = []
        for class_id in torch.unique(y[train_idx]).tolist():
            members = train_idx[y[train_idx] == int(class_id)]
            if members.numel() > 1:
                reserved.append(members[-1:])
                retained.append(members[:-1])
            else:
                retained.append(members)
        if reserved:
            monitor_idx = torch.cat(reserved)
            train_idx = torch.cat(retained)
    if monitor_idx.numel() == 0:
        return repaired, {
            "applied": False,
            "reason": "no_disjoint_monitor_anchors",
        }
    if int(max_steps) < 1 or int(patience) < 1:
        raise ValueError("adaptive utility repair needs positive caps")

    teacher = None
    if anchor_state is not None:
        teacher = copy.deepcopy(repaired)
        teacher.load_state_dict(anchor_state)
        teacher.eval()
    optimizer = torch.optim.Adam(
        repaired.parameters(), lr=float(lr), weight_decay=5e-4)

    def monitor_metrics():
        repaired.eval()
        with torch.no_grad():
            monitor_logits = repaired(x, adj)[monitor_idx]
            risk = float(_class_balanced_cross_entropy(
                monitor_logits, y[monitor_idx]).item())
            truth = y[monitor_idx]
            prediction = monitor_logits.argmax(1)
            accuracy = float((prediction == truth).float().mean().item())
            recalls = {}
            for class_id in torch.unique(truth).tolist():
                class_mask = truth == int(class_id)
                recalls[str(int(class_id))] = float(
                    (prediction[class_mask] == truth[class_mask])
                    .float().mean())
            macro_recall = float(np.mean(list(recalls.values())))
        return {
            "balanced_ce": risk,
            "accuracy": accuracy,
            "macro_recall": macro_recall,
            "class_recall": recalls,
        }

    selection_mode = str(selection_mode).lower()
    if selection_mode not in ("balanced-risk", "accuracy-pareto"):
        raise ValueError(
            "adaptive utility selection must be balanced-risk or "
            "accuracy-pareto")
    initial_metrics = monitor_metrics()
    initial_risk = initial_metrics["balanced_ce"]
    initial_accuracy = initial_metrics["accuracy"]
    best_risk = initial_risk
    best_accuracy = initial_accuracy
    best_metrics = initial_metrics
    best_rank = (initial_accuracy, initial_metrics["macro_recall"],
                 -initial_risk)
    best_step = 0
    best_state = copy.deepcopy(repaired.state_dict())
    stale_steps = 0
    trace = [{"step": 0, "monitor_balanced_ce": initial_risk,
              "monitor_accuracy": initial_accuracy}]
    for step in range(1, int(max_steps) + 1):
        repaired.train()
        logits = repaired(x, adj)
        loss = _class_balanced_cross_entropy(logits[train_idx], y[train_idx])
        if teacher is not None:
            with torch.no_grad():
                teacher_logits = teacher(x, adj)[train_idx]
            loss = loss + float(w_align) * F.mse_loss(
                logits[train_idx], teacher_logits)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(repaired.parameters(), 5.0)
        optimizer.step()

        metrics = monitor_metrics()
        risk = metrics["balanced_ce"]
        accuracy = metrics["accuracy"]
        trace.append({"step": int(step), "monitor_balanced_ce": risk,
                      "monitor_accuracy": accuracy,
                      "monitor_macro_recall": metrics["macro_recall"],
                      "monitor_class_recall": metrics["class_recall"]})
        # Permit a one-sample accuracy fluctuation, but never select a model
        # that materially sacrifices held-out utility for lower CE.
        accuracy_floor = initial_accuracy - 1.0 / monitor_idx.numel()
        if selection_mode == "accuracy-pareto":
            recall_ok = all(
                metrics["class_recall"].get(class_id, 0.0)
                >= recall - float(max_class_drop) - 1e-8
                for class_id, recall in initial_metrics[
                    "class_recall"].items())
            rank = (accuracy, metrics["macro_recall"], -risk)
            improved = bool(
                accuracy >= accuracy_floor - 1e-8
                and recall_ok
                and (rank[:2] > best_rank[:2]
                     or (rank[:2] == best_rank[:2]
                         and risk < best_risk - float(min_delta))))
        else:
            rank = (accuracy, metrics["macro_recall"], -risk)
            improved = (accuracy >= accuracy_floor
                        and risk < best_risk - float(min_delta))
        if improved:
            best_risk = risk
            best_accuracy = accuracy
            best_metrics = metrics
            best_rank = rank
            best_step = int(step)
            best_state = copy.deepcopy(repaired.state_dict())
            stale_steps = 0
        else:
            stale_steps += 1
        if stale_steps >= int(patience):
            break

    repaired.load_state_dict(best_state)
    repaired.eval()
    if hasattr(repaired, "_feats"):
        repaired._feats = None
    if hasattr(repaired, "_causal_gate"):
        repaired._causal_gate = None
    return repaired, {
        "applied": True,
        "selection": "disjoint held-out class-balanced risk",
        "selection_mode": selection_mode,
        "train_anchors": int(train_idx.numel()),
        "monitor_anchors": int(monitor_idx.numel()),
        "selected_step": int(best_step),
        "executed_steps": int(len(trace) - 1),
        "max_steps_safety_cap": int(max_steps),
        "patience": int(patience),
        "initial_monitor_balanced_ce": float(initial_risk),
        "selected_monitor_balanced_ce": float(best_risk),
        "initial_monitor_accuracy": float(initial_accuracy),
        "selected_monitor_accuracy": float(best_accuracy),
        "initial_monitor_macro_recall": float(
            initial_metrics["macro_recall"]),
        "selected_monitor_macro_recall": float(
            best_metrics["macro_recall"]),
        "max_class_recall_drop": float(max_class_drop),
        "trace": trace,
        "suspicious_updates_reintroduced": False,
        "target_class_used": False,
    }


def crossfit_server_utility_repair(
        model, x, adj, y, anchor_indices, folds=3, max_steps=50,
        patience=6, min_delta=1e-4, lr=0.001, w_align=1.0,
        max_class_drop=0.05, sensitivity_guard=False,
        sensitivity_views=3, sensitivity_budget=0.20,
        dose_monitor_idx=None, seed=0, device="cpu"):
    """Average independently monitored class-balanced repair paths.

    Every trusted anchor is held out from the optimization of exactly one
    repair path.  The averaged update is committed only when concatenated
    out-of-fold predictions preserve accuracy and every represented class
    recall while reducing class-balanced risk.  No attack identity, target
    class, trigger template, or test label is consumed.
    """
    x, adj, y = x.to(device), adj.to(device), y.to(device)
    anchor_indices = np.unique(np.asarray(
        anchor_indices, dtype=np.int64).reshape(-1))
    fold_count = max(int(folds), 2)
    if anchor_indices.size < fold_count:
        return copy.deepcopy(model), {
            "applied": False, "committed": False,
            "reason": "insufficient_anchors_for_crossfit",
        }
    labels_np = y.detach().cpu().numpy()
    fold_members = [[] for _ in range(fold_count)]
    for class_id in np.unique(labels_np[anchor_indices]):
        members = anchor_indices[labels_np[anchor_indices] == class_id].copy()
        rng = np.random.RandomState(int(seed) + int(class_id) * 1009)
        rng.shuffle(members)
        for position, node_idx in enumerate(members.tolist()):
            fold_members[position % fold_count].append(int(node_idx))
    fold_members = [np.sort(np.asarray(values, dtype=np.int64))
                    for values in fold_members]
    if any(values.size == 0 for values in fold_members):
        return copy.deepcopy(model), {
            "applied": False, "committed": False,
            "reason": "empty_stratified_crossfit_fold",
        }

    repaired_models = []
    fold_diagnostics = []
    baseline_logits, repaired_logits, heldout_labels = [], [], []
    reference = copy.deepcopy(model).to(device).eval()
    anchor_set = set(int(value) for value in anchor_indices.tolist())
    for fold_idx, monitor_values in enumerate(fold_members):
        monitor_set = set(int(value) for value in monitor_values.tolist())
        train_values = np.asarray(sorted(anchor_set - monitor_set),
                                  dtype=np.int64)
        fold_model, fold_diag = adaptive_server_utility_repair(
            reference, x, adj, y, train_values,
            monitor_idx=monitor_values, anchor_state=None,
            max_steps=max_steps, patience=patience, min_delta=min_delta,
            lr=lr, w_align=w_align, max_class_drop=max_class_drop,
            selection_mode="balanced-risk", device=device)
        repaired_models.append(fold_model)
        fold_diag = {"fold": int(fold_idx), **fold_diag}
        fold_diagnostics.append(fold_diag)
        monitor_tensor = torch.as_tensor(
            monitor_values, dtype=torch.long, device=device)
        with torch.no_grad():
            baseline_logits.append(reference(x, adj)[monitor_tensor])
            repaired_logits.append(fold_model(x, adj)[monitor_tensor])
            heldout_labels.append(y[monitor_tensor])

    baseline_logits = torch.cat(baseline_logits, dim=0)
    repaired_logits = torch.cat(repaired_logits, dim=0)
    heldout_labels = torch.cat(heldout_labels, dim=0)

    def out_of_fold_metrics(logits):
        prediction = logits.argmax(dim=1)
        recalls = {}
        for class_id in torch.unique(heldout_labels).tolist():
            mask = heldout_labels == int(class_id)
            recalls[str(int(class_id))] = float(
                (prediction[mask] == heldout_labels[mask]).float().mean())
        return {
            "accuracy": float(
                (prediction == heldout_labels).float().mean()),
            "balanced_ce": float(_class_balanced_cross_entropy(
                logits, heldout_labels)),
            "macro_recall": float(np.mean(list(recalls.values()))),
            "class_recall": recalls,
        }

    baseline_oof = out_of_fold_metrics(baseline_logits)
    repaired_oof = out_of_fold_metrics(repaired_logits)
    accuracy_floor = baseline_oof["accuracy"] - 1.0 / max(
        int(anchor_indices.size), 1)
    recall_ok = all(
        repaired_oof["class_recall"].get(class_id, 0.0)
        >= recall - float(max_class_drop) - 1e-8
        for class_id, recall in baseline_oof["class_recall"].items())
    committed = bool(
        repaired_oof["accuracy"] >= accuracy_floor - 1e-8
        and repaired_oof["balanced_ce"]
        < baseline_oof["balanced_ce"] - float(min_delta)
        and recall_ok)
    averaged = weighted_average_models(
        repaired_models, [1.0 / fold_count] * fold_count)
    selected_dose = 1.0 if committed else 0.0
    selected_head_dose = selected_dose
    sensitivity_diagnostics = None
    if committed and sensitivity_guard:
        monitor_values = (anchor_indices if dose_monitor_idx is None
                          else np.asarray(
                              dose_monitor_idx, dtype=np.int64).reshape(-1))
        anchor_tensor = torch.as_tensor(
            monitor_values, dtype=torch.long, device=device)
        baseline_sensitivity = _head_repair_sensitivity(
            reference, x, adj, anchor_tensor, sensitivity_views,
            sensitivity_budget, int(seed) + 33017)
        baseline_monitor = _head_repair_metrics(
            reference, x, adj, y, anchor_tensor)
        monitor_accuracy_floor = baseline_monitor["accuracy"] - (
            1.0 / max(int(anchor_tensor.numel()), 1))
        parameterized_modules = [
            name for name, module in reference.named_modules()
            if name and any(True for _ in module.parameters(recurse=False))]
        head_prefix = ((parameterized_modules[-1] + ".")
                       if parameterized_modules else None)

        def layerwise_candidate(body_dose, head_dose):
            candidate = copy.deepcopy(reference).to(device)
            reference_parameters = dict(reference.named_parameters())
            repaired_parameters = dict(averaged.named_parameters())
            for name, parameter in candidate.named_parameters():
                dose = (float(head_dose)
                        if head_prefix and name.startswith(head_prefix)
                        else float(body_dose))
                start = reference_parameters[name].to(
                    device=parameter.device, dtype=parameter.dtype)
                end = repaired_parameters[name].to(
                    device=parameter.device, dtype=parameter.dtype)
                with torch.no_grad():
                    parameter.copy_(start + dose * (end - start))
            candidate.eval()
            return candidate

        dose_candidates = []
        for body_dose in (0.0, 0.10, 0.25, 0.50):
            for head_dose in (0.0, 0.50, 1.0):
                candidate = layerwise_candidate(body_dose, head_dose)
                metrics = _head_repair_metrics(
                    candidate, x, adj, y, anchor_tensor)
                sensitivity = _head_repair_sensitivity(
                    candidate, x, adj, anchor_tensor, sensitivity_views,
                    sensitivity_budget, int(seed) + 33017)
                candidate_recall_ok = all(
                    metrics["class_recall"].get(class_id, 0.0)
                    >= recall - float(max_class_drop) - 1e-8
                    for class_id, recall in baseline_monitor[
                        "class_recall"].items())
                eligible = bool(
                    metrics["accuracy"] >= monitor_accuracy_floor - 1e-8
                    and candidate_recall_ok
                    and sensitivity <= baseline_sensitivity * 1.02 + 1e-8)
                dose_candidates.append({
                    "body_dose": float(body_dose),
                    "head_dose": float(head_dose),
                    "eligible": eligible,
                    "monitor": metrics,
                    "intervention_sensitivity": float(sensitivity),
                    "model": candidate,
                })
        eligible_doses = [item for item in dose_candidates
                          if item["eligible"]]
        chosen = max(
            eligible_doses,
            key=lambda item: (
                item["monitor"]["accuracy"],
                item["monitor"]["macro_recall"],
                -item["monitor"]["balanced_ce"],
                -item["intervention_sensitivity"],
                -item["body_dose"], -item["head_dose"]))
        selected = chosen.pop("model")
        selected_dose = float(chosen["body_dose"])
        selected_head_dose = float(chosen["head_dose"])
        committed = bool(selected_dose > 0.0 or selected_head_dose > 0.0)
        for item in dose_candidates:
            item.pop("model", None)
        sensitivity_diagnostics = {
            "enabled": True,
            "baseline_intervention_sensitivity": float(
                baseline_sensitivity),
            "baseline_monitor": baseline_monitor,
            "selected_body_dose": float(selected_dose),
            "selected_head_dose": float(selected_head_dose),
            "monitor_nodes": int(anchor_tensor.numel()),
            "head_prefix": head_prefix,
            "candidates": dose_candidates,
            "selection": (
                "independent monitor utility subject to target-free graph "
                "intervention sensitivity"),
        }
    else:
        selected = averaged if committed else copy.deepcopy(reference)
    return selected, {
        "applied": True,
        "committed": committed,
        "folds": int(fold_count),
        "anchors": int(anchor_indices.size),
        "baseline_out_of_fold": baseline_oof,
        "repaired_out_of_fold": repaired_oof,
        "max_class_recall_drop": float(max_class_drop),
        "selected_repair_dose": float(selected_dose),
        "selected_head_repair_dose": float(selected_head_dose),
        "intervention_sensitivity_guard": sensitivity_diagnostics,
        "fold_diagnostics": fold_diagnostics,
        "suspicious_updates_reintroduced": False,
        "attack_identity_used": False,
        "target_class_used": False,
        "trigger_template_used": False,
        "test_labels_used": False,
    }


@torch.no_grad()
def _head_repair_metrics(model, x, adj, y, indices):
    indices = torch.as_tensor(
        indices, dtype=torch.long, device=x.device).reshape(-1)
    model.eval()
    logits = model(x, adj)[indices]
    truth = y[indices]
    prediction = logits.argmax(dim=1)
    recalls = {}
    for class_id in torch.unique(truth).tolist():
        mask = truth == int(class_id)
        recalls[str(int(class_id))] = float(
            (prediction[mask] == truth[mask]).float().mean())
    recall_values = list(recalls.values())
    return {
        "balanced_ce": float(_class_balanced_cross_entropy(logits, truth)),
        "accuracy": float((prediction == truth).float().mean()),
        "macro_recall": float(np.mean(recall_values)),
        "worst_class_recall": float(min(recall_values)),
        "class_recall": recalls,
    }


def _head_repair_world(x, adj, budget, seed):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    gate = torch.rand(x.shape[0], generator=generator, dtype=x.dtype)
    gate = gate.to(x.device)
    gate = (gate * (float(budget) / gate.mean().clamp(min=1e-6)))\
        .clamp(max=1.0)
    reference = x.median(dim=0).values
    x_do = x + gate[:, None] * (reference[None, :] - x)
    keep = (1.0 - gate[:, None]) * (1.0 - gate[None, :])
    return x_do, adj * keep


@torch.no_grad()
def _head_repair_sensitivity(model, x, adj, indices, views, budget, seed):
    indices = torch.as_tensor(
        indices, dtype=torch.long, device=x.device).reshape(-1)
    model.eval()
    factual = F.softmax(model(x, adj)[indices], dim=1)
    values = []
    for view_idx in range(max(int(views), 1)):
        x_do, adj_do = _head_repair_world(
            x, adj, budget, int(seed) + view_idx * 104729)
        intervened = F.softmax(model(x_do, adj_do)[indices], dim=1)
        mixture = 0.5 * (factual + intervened)
        values.append(0.5 * (
            F.kl_div(mixture.clamp(min=1e-8).log(), factual,
                     reduction="batchmean")
            + F.kl_div(mixture.clamp(min=1e-8).log(), intervened,
                       reduction="batchmean")))
    return float(torch.stack(values).mean())


def weighted_average_models(models, weights):
    keys = list(models[0].state_dict().keys())
    for m in models:
        if hasattr(m, "_feats"):
            m._feats = None
        if hasattr(m, "_causal_gate"):
            m._causal_gate = None
    avg = copy.deepcopy(models[0])
    sd = avg.state_dict()
    for k in keys:
        acc = None
        for w, m in zip(weights, models):
            term = w * m.state_dict()[k].float()
            acc = term if acc is None else acc + term
        sd[k] = acc
    avg.load_state_dict(sd)
    return avg


def _model_update_vector(reference_model, model):
    reference = reference_model.state_dict()
    current = model.state_dict()
    keys = [key for key, value in reference.items()
            if torch.is_floating_point(value)]
    vector = torch.cat([
        (current[key].detach().float() - reference[key].detach().float())
        .reshape(-1) for key in keys])
    return vector, keys


def _model_from_update_vector(reference_model, vector, keys):
    repaired = copy.deepcopy(reference_model)
    state = repaired.state_dict()
    offset = 0
    for key in keys:
        count = state[key].numel()
        delta = vector[offset:offset + count].reshape(state[key].shape)
        state[key] = state[key] + delta.to(
            device=state[key].device, dtype=state[key].dtype)
        offset += count
    repaired.load_state_dict(state)
    if hasattr(repaired, "_feats"):
        repaired._feats = None
    return repaired


def _model_from_group_doses(reference_model, model, group_doses):
    repaired = copy.deepcopy(reference_model)
    reference_state = reference_model.state_dict()
    current_state = model.state_dict()
    repaired_state = repaired.state_dict()
    for key, reference_value in reference_state.items():
        if torch.is_floating_point(reference_value):
            group = key.split(".", 1)[0]
            dose = float(group_doses.get(group, 0.0))
            repaired_state[key] = (reference_value.float()
                                   + dose * (current_state[key].float()
                                             - reference_value.float())).to(
                device=reference_value.device, dtype=reference_value.dtype)
        else:
            repaired_state[key] = reference_value
    repaired.load_state_dict(repaired_state)
    if hasattr(repaired, "_feats"):
        repaired._feats = None
    return repaired


@torch.no_grad()
def causal_parameter_path_repair(reference_model, models, suspicious_indices,
                                 variance_keep=0.95, norm_clip=1.5,
                                 removal_strength=1.0,
                                 min_pair_cosine=-1.0):
    """Block the shared do(theta) residual of a causally detected group."""
    suspicious = sorted(int(v) for v in suspicious_indices)
    benign = [idx for idx in range(len(models)) if idx not in suspicious]
    if not suspicious or not benign:
        return list(models), {"repair_ready": False}
    updates = []
    keys = None
    for model in models:
        update, model_keys = _model_update_vector(reference_model, model)
        updates.append(update)
        keys = model_keys
    matrix = torch.stack(updates)
    benign_matrix = matrix[benign]
    _, singular, vh = torch.linalg.svd(benign_matrix, full_matrices=False)
    energy = singular.square()
    cumulative = torch.cumsum(energy, dim=0) / energy.sum().clamp(min=1e-12)
    rank = int(torch.searchsorted(
        cumulative, torch.tensor(float(variance_keep),
                                 device=cumulative.device)).item()) + 1
    rank = min(rank, vh.shape[0])
    task_basis = vh[:rank]
    suspicious_matrix = matrix[suspicious]
    task_projection = ((suspicious_matrix @ task_basis.T) @ task_basis)
    residual = suspicious_matrix - task_projection
    normalized_residual = residual / residual.norm(
        dim=1, keepdim=True).clamp(min=1e-12)
    pair_cosines = normalized_residual @ normalized_residual.T
    off_diag = pair_cosines[~torch.eye(
        len(suspicious), dtype=torch.bool, device=pair_cosines.device)]
    mean_pair_cosine = (
        float(off_diag.mean()) if off_diag.numel() else 0.0)
    if mean_pair_cosine < float(min_pair_cosine):
        return list(models), {
            "repair_ready": False,
            "reason": "coalition residual lacks cross-client coherence",
            "task_subspace_rank": rank,
            "task_variance_keep": float(variance_keep),
            "shared_residual_mean_pair_cosine": mean_pair_cosine,
            "minimum_pair_cosine": float(min_pair_cosine),
            "target_class_used": False,
            "clean_graph_used": False,
            "energy_used": False,
        }
    common = residual.mean(dim=0)
    common_norm = common.norm()
    if float(common_norm) <= 1e-12:
        return list(models), {
            "repair_ready": False, "task_subspace_rank": rank}
    direction = common / common_norm
    coefficients = suspicious_matrix @ direction
    strength = min(max(float(removal_strength), 0.0), 1.0)
    repaired_matrix = (
        suspicious_matrix - strength * coefficients[:, None] * direction)

    benign_norm = benign_matrix.norm(dim=1).median()
    max_norm = float(norm_clip) * benign_norm
    repaired_norm = repaired_matrix.norm(dim=1).clamp(min=1e-12)
    scales = torch.minimum(torch.ones_like(repaired_norm),
                           max_norm / repaired_norm)
    repaired_matrix = repaired_matrix * scales[:, None]
    repaired_models = list(models)
    for row, idx in enumerate(suspicious):
        repaired_models[idx] = _model_from_update_vector(
            reference_model, repaired_matrix[row], keys)

    removed_fraction = (
        (suspicious_matrix - repaired_matrix).norm(dim=1)
        / suspicious_matrix.norm(dim=1).clamp(min=1e-12))
    return repaired_models, {
        "repair_ready": True,
        "task_subspace_rank": rank,
        "task_variance_keep": float(variance_keep),
        "shared_residual_mean_pair_cosine": (
            mean_pair_cosine),
        "minimum_pair_cosine": float(min_pair_cosine),
        "removed_update_fraction": [float(v) for v in removed_fraction],
        "norm_clip_relative_to_benign_median": float(norm_clip),
        "shared_residual_removal_strength": float(strength),
        "target_class_used": False,
        "clean_graph_used": False,
        "energy_used": False,
        "estimand": "shared residual pathway after do(G)-by-do(theta) audit",
    }


def causal_trigger_path_block(reference_model, models, suspicious_indices,
                              utility_probe=None, variance_keep=0.95,
                              norm_clip=1.5, utility_shards_per_class=1,
                              device="cpu"):
    """Retain trusted task utility while blocking untrusted trigger paths.

    LCCT supplies ``suspicious_indices``.  This routine never inspects an
    attack name, target class, or trigger template.  It builds a protected
    task span from the other clients and, when available, class-conditional
    gradients on the server's fixed clean probe.  A suspicious update may
    return only through that protected span; its entire orthogonal residual
    is blocked instead of merely removing one coalition-average direction.
    """
    suspicious = sorted(int(value) for value in suspicious_indices)
    benign = [idx for idx in range(len(models)) if idx not in suspicious]
    if not suspicious or len(benign) < 2:
        return list(models), {
            "repair_ready": False,
            "reason": "trigger blocking needs suspicious and benign clients",
        }

    updates = []
    keys = None
    for model in models:
        update, model_keys = _model_update_vector(reference_model, model)
        updates.append(update.to(device))
        keys = model_keys
    matrix = torch.stack(updates)
    benign_matrix = matrix[benign]
    benign_basis = _row_subspace_basis(
        benign_matrix, variance_keep=variance_keep,
        max_rank=max(len(benign) - 1, 1))
    utility_basis, utility_diag = _probe_utility_basis(
        reference_model, utility_probe, keys, device=device,
        shards_per_class=utility_shards_per_class)
    protected_parts = [
        value for value in (benign_basis, utility_basis)
        if value is not None and value.numel()]
    protected_basis = (_row_subspace_basis(
        torch.cat(protected_parts, dim=0), variance_keep=0.999,
        max_rank=sum(value.shape[0] for value in protected_parts))
        if protected_parts else None)
    if protected_basis is None or protected_basis.numel() == 0:
        return list(models), {
            "repair_ready": False,
            "reason": "protected task span is empty",
            "utility_basis": utility_diag,
        }

    suspicious_matrix = matrix[suspicious]
    protected_projection = (
        suspicious_matrix @ protected_basis.T) @ protected_basis
    blocked_residual = suspicious_matrix - protected_projection

    benign_norm = benign_matrix.norm(dim=1).median().clamp_min(1e-10)
    max_norm = float(norm_clip) * benign_norm
    repaired_norm = protected_projection.norm(dim=1).clamp_min(1e-12)
    scales = torch.minimum(
        torch.ones_like(repaired_norm), max_norm / repaired_norm)
    repaired_matrix = protected_projection * scales[:, None]
    repaired_models = list(models)
    for row, idx in enumerate(suspicious):
        repaired_models[idx] = _model_from_update_vector(
            reference_model, repaired_matrix[row], keys)

    removed_fraction = (
        blocked_residual.norm(dim=1)
        / suspicious_matrix.norm(dim=1).clamp_min(1e-12))
    retained_fraction = (
        repaired_matrix.norm(dim=1)
        / suspicious_matrix.norm(dim=1).clamp_min(1e-12))
    return repaired_models, {
        "repair_ready": True,
        "protected_basis_rank": int(protected_basis.shape[0]),
        "benign_basis_rank": (
            int(benign_basis.shape[0])
            if benign_basis is not None else 0),
        "utility_basis": utility_diag,
        "blocked_residual_fraction": [
            float(value) for value in removed_fraction],
        "retained_task_fraction": [
            float(value) for value in retained_fraction],
        "norm_clip_relative_to_benign_median": float(norm_clip),
        "target_class_used": False,
        "attack_identity_used": False,
        "trigger_template_used": False,
        "clean_probe_used": bool(utility_probe is not None),
        "clean_probe_labels_used": bool(utility_probe is not None),
        "estimand": (
            "LCCT-attributed update projected onto trusted task and "
            "class-utility span; orthogonal trigger pathway blocked"),
    }


def restore_invariant_private_amplitude(
        reference_model, uploaded_models, repaired_models, repaired_indices,
        trusted_indices, device="cpu"):
    """Restore only the magnitude of an already-sanitized update direction.

    The uploaded update is used solely to estimate a scalar coefficient along
    the cross-intervention-stable repaired direction.  No uploaded residual or
    orthogonal coordinate can re-enter the aggregate.  The restored norm is
    capped by the median trusted-client update norm, making the operation
    independent of attack, target class, dataset, and parameter dimension.
    """
    repaired_indices = sorted(int(value) for value in repaired_indices)
    trusted_indices = sorted(int(value) for value in trusted_indices)
    if not repaired_indices or not trusted_indices:
        return list(repaired_models), {
            "applied": False,
            "reason": "stable amplitude restore needs repaired and trusted clients",
        }

    trusted_norms = []
    keys = None
    for idx in trusted_indices:
        update, model_keys = _model_update_vector(
            reference_model, uploaded_models[idx])
        trusted_norms.append(update.to(device).norm())
        keys = model_keys
    if not trusted_norms:
        return list(repaired_models), {
            "applied": False, "reason": "trusted update norms unavailable"}
    trusted_median_norm = torch.stack(trusted_norms).median().clamp_min(1e-12)

    restored_models = list(repaired_models)
    records = {}
    for idx in repaired_indices:
        uploaded_update, uploaded_keys = _model_update_vector(
            reference_model, uploaded_models[idx])
        repaired_update, repaired_keys = _model_update_vector(
            reference_model, repaired_models[idx])
        if uploaded_keys != repaired_keys:
            raise ValueError("uploaded and repaired model parameter keys differ")
        keys = repaired_keys
        uploaded_update = uploaded_update.to(device)
        repaired_update = repaired_update.to(device)
        repaired_norm = repaired_update.norm().clamp_min(1e-12)
        projected_scale = torch.clamp(
            torch.dot(uploaded_update, repaired_update)
            / repaired_norm.square(), min=0.0)
        norm_bound_scale = torch.maximum(
            torch.ones_like(repaired_norm),
            trusted_median_norm / repaired_norm)
        selected_scale = torch.minimum(
            torch.maximum(torch.ones_like(projected_scale), projected_scale),
            norm_bound_scale)
        restored_update = repaired_update * selected_scale
        restored_models[idx] = _model_from_update_vector(
            reference_model, restored_update, keys)
        cosine = torch.dot(uploaded_update, repaired_update) / (
            uploaded_update.norm().clamp_min(1e-12) * repaired_norm)
        records[str(idx)] = {
            "uploaded_to_repaired_cosine": float(cosine),
            "uploaded_projection_scale": float(projected_scale),
            "selected_scale": float(selected_scale),
            "repaired_norm_before": float(repaired_norm),
            "repaired_norm_after": float(restored_update.norm()),
        }
    return restored_models, {
        "applied": True,
        "trusted_median_update_norm": float(trusted_median_norm),
        "clients": records,
        "new_uploaded_directions_admitted": False,
        "target_class_used": False,
        "attack_identity_used": False,
        "dataset_identity_used": False,
        "estimand": (
            "uploaded coefficient along the independently reconstructed "
            "cross-intervention-stable update direction"),
    }


def _row_subspace_basis(rows, variance_keep=0.95, max_rank=None):
    """Return an orthonormal row basis with an energy-selected rank."""
    if rows is None or rows.numel() == 0:
        return None
    rows = rows.float()
    nonzero = rows.norm(dim=1) > 1e-10
    rows = rows[nonzero]
    if rows.numel() == 0:
        return rows.new_zeros((0, rows.shape[1]))
    _, singular, vh = torch.linalg.svd(rows, full_matrices=False)
    energy = singular.square()
    if float(energy.sum()) <= 1e-12:
        return rows.new_zeros((0, rows.shape[1]))
    cumulative = torch.cumsum(energy, dim=0) / energy.sum()
    rank = int(torch.searchsorted(
        cumulative, torch.tensor(float(variance_keep),
                                 device=cumulative.device)).item()) + 1
    if max_rank is not None:
        rank = min(rank, max(int(max_rank), 0))
    return vh[:max(rank, 0)]


def _project_off_basis(rows, basis):
    if basis is None or basis.numel() == 0:
        return rows
    return rows - (rows @ basis.T) @ basis


def _probe_utility_basis(reference_model, probe, keys, device="cpu",
                         shards_per_class=1):
    """Build a class-conditional clean-gradient span used only as protection."""
    if probe is None:
        return None, {"available": False, "reason": "no clean probe"}
    x, adj, labels, probe_idx = probe
    model = copy.deepcopy(reference_model).to(device).eval()
    x = x.to(device)
    adj = adj.to(device)
    labels = labels.to(device)
    probe_idx = torch.as_tensor(
        probe_idx, dtype=torch.long, device=device).reshape(-1)
    named_parameters = dict(model.named_parameters())
    gradients = []
    classes = []
    shard_limit = max(int(shards_per_class), 1)
    for class_id in torch.unique(labels[probe_idx]).tolist():
        class_idx = probe_idx[labels[probe_idx] == int(class_id)]
        if class_idx.numel() == 0:
            continue
        shard_count = min(shard_limit, int(class_idx.numel()))
        for shard_idx in torch.tensor_split(class_idx, shard_count):
            if shard_idx.numel() == 0:
                continue
            model.zero_grad(set_to_none=True)
            logits = model(x, adj)
            loss = F.cross_entropy(logits[shard_idx], labels[shard_idx])
            loss.backward()
            pieces = []
            for key in keys:
                parameter = named_parameters.get(key)
                if parameter is None or parameter.grad is None:
                    pieces.append(torch.zeros_like(
                        model.state_dict()[key],
                        dtype=torch.float32).reshape(-1))
                else:
                    pieces.append(
                        parameter.grad.detach().float().reshape(-1))
            gradient = torch.cat(pieces)
            if float(gradient.norm()) > 1e-10:
                gradients.append(gradient)
                classes.append(int(class_id))
    if not gradients:
        return None, {"available": False,
                      "reason": "all class-conditional gradients vanished"}
    basis = _row_subspace_basis(
        torch.stack(gradients), variance_keep=0.999,
        max_rank=len(gradients))
    return basis, {
        "available": True,
        "classes": classes,
        "rank": int(basis.shape[0]),
        "probe_nodes": int(probe_idx.numel()),
        "shards_per_class": int(shard_limit),
    }


def class_semantic_update_filter(
        reference_model, models, suspicious_indices, utility_probe,
        shards_per_class=3, norm_clip=1.0, robust_coefficient_clip=False,
        utility_deficit_gate=False, device="cpu"):
    """Keep only clean class-descent paths from LCCT-attributed updates.

    Unlike the final-round recovery routines, this filter is intended for the
    CRFU aggregation path itself.  The trusted probe defines one clean descent
    direction per class (averaged across stratified shards).  A suspicious
    client may contribute only its positive projection onto those directions;
    every orthogonal coordinate remains blocked.  Cross-class gradient
    conflicts are removed before projection so restoring one class does not
    knowingly increase another trusted class risk.
    """
    suspicious = sorted(set(int(value) for value in suspicious_indices))
    trusted = [index for index in range(len(models))
               if index not in set(suspicious)]
    if utility_probe is None or not suspicious or not trusted:
        return list(models), {
            "repair_ready": False,
            "reason": "clean probe, suspicious, and trusted clients required",
        }
    if hasattr(reference_model, "_feats"):
        reference_model._feats = None
    if hasattr(reference_model, "_causal_gate"):
        reference_model._causal_gate = None
    x, adj, labels, probe_idx = utility_probe
    x, adj, labels = x.to(device), adj.to(device), labels.to(device)
    probe_idx = torch.as_tensor(
        probe_idx, dtype=torch.long, device=device).reshape(-1)
    model = copy.deepcopy(reference_model).to(device).eval()
    named_parameters = dict(model.named_parameters())
    keys = [name for name, parameter in model.named_parameters()
            if parameter.dtype.is_floating_point]

    def descent_for(indices):
        model.zero_grad(set_to_none=True)
        logits = model(x, adj)
        loss = F.cross_entropy(logits[indices], labels[indices])
        loss.backward()
        pieces = []
        for key in keys:
            parameter = named_parameters[key]
            pieces.append(parameter.grad.detach().float().reshape(-1))
        descent = -torch.cat(pieces)
        norm = descent.norm()
        return descent / norm if float(norm) > 1e-12 else None

    class_directions = {}
    class_records = {}
    shard_limit = max(int(shards_per_class), 1)
    for class_id in torch.unique(labels[probe_idx]).tolist():
        class_nodes = probe_idx[labels[probe_idx] == int(class_id)]
        shards = torch.tensor_split(
            class_nodes, min(shard_limit, int(class_nodes.numel())))
        directions = [descent_for(shard) for shard in shards
                      if shard.numel()]
        directions = [value for value in directions if value is not None]
        if not directions:
            continue
        direction_matrix = torch.stack(directions)
        mean_direction = direction_matrix.mean(dim=0)
        mean_norm = mean_direction.norm()
        if float(mean_norm) <= 1e-12:
            continue
        mean_direction = mean_direction / mean_norm
        gram = direction_matrix @ direction_matrix.T
        off_diagonal = gram[~torch.eye(
            len(directions), dtype=torch.bool, device=gram.device)]
        stability = (float(off_diagonal.median())
                     if off_diagonal.numel() else 1.0)
        if stability < 0.0:
            continue
        class_directions[int(class_id)] = mean_direction
        class_records[str(int(class_id))] = {
            "probe_support": int(class_nodes.numel()),
            "shards": int(len(directions)),
            "direction_stability": float(stability),
        }
    model.zero_grad(set_to_none=True)
    if not class_directions:
        return list(models), {
            "repair_ready": False,
            "reason": "no fold-stable clean class direction",
            "classes": class_records,
        }

    class_release_weights = {
        int(class_id): 1.0 for class_id in class_directions}
    if utility_deficit_gate:
        with torch.no_grad():
            predictions = model(x, adj).argmax(dim=1)
        recalls = {}
        for class_id in class_directions:
            class_nodes = probe_idx[labels[probe_idx] == int(class_id)]
            recalls[int(class_id)] = float(
                (predictions[class_nodes] == int(class_id)).float().mean())
        recall_center = float(torch.tensor(
            list(recalls.values()), dtype=torch.float32).median())
        if recall_center > 1e-8:
            class_release_weights = {
                class_id: min(max(
                    (recall_center - recall) / recall_center, 0.0), 1.0)
                for class_id, recall in recalls.items()}
        else:
            class_release_weights = {
                class_id: 1.0 for class_id in recalls}
        for class_id, recall in recalls.items():
            absolute_gap = max(recall_center - recall, 0.0)
            class_records[str(class_id)].update({
                "reference_probe_recall": float(recall),
                "reference_recall_median": float(recall_center),
                "utility_deficit_absolute_gap": float(absolute_gap),
                "utility_deficit_weight": float(
                    class_release_weights[class_id]),
            })

    # Symmetric PCGrad surgery: class identities are used only as clean task
    # environments, never as an attack-target guess.
    repaired_directions = {}
    for class_id in sorted(class_directions):
        direction = class_directions[class_id].clone()
        conflicts = []
        for _ in range(2):
            for other_class in sorted(class_directions):
                if other_class == class_id:
                    continue
                other = class_directions[other_class]
                alignment = torch.dot(direction, other)
                if float(alignment) < 0.0:
                    conflicts.append(int(other_class))
                    direction = direction - alignment * other
        direction = direction / direction.norm().clamp_min(1e-12)
        if float(torch.dot(direction, class_directions[class_id])) > 0.0:
            repaired_directions[class_id] = direction
        class_records[str(class_id)]["conflicting_classes_removed"] = (
            sorted(set(conflicts)))

    updates = []
    model_keys = None
    for uploaded in models:
        update, current_keys = _model_update_vector(
            reference_model, uploaded)
        updates.append(update.to(device))
        model_keys = current_keys
    matrix = torch.stack(updates)
    trusted_norm = matrix[trusted].norm(dim=1).median().clamp_min(1e-12)
    maximum_norm = max(float(norm_clip), 0.0) * trusted_norm
    class_coefficient_caps = {}
    for class_id, direction in repaired_directions.items():
        trusted_coefficients = (
            matrix[trusted] @ direction).clamp_min(0.0)
        median = trusted_coefficients.median()
        mad = (trusted_coefficients - median).abs().median()
        robust_upper = median + 2.5 * 1.4826 * mad
        observed_upper = trusted_coefficients.max()
        cap = torch.minimum(robust_upper, observed_upper)
        class_coefficient_caps[class_id] = cap.clamp_min(0.0)
        class_records[str(class_id)].update({
            "trusted_coefficient_median": float(median),
            "trusted_coefficient_mad": float(mad),
            "trusted_coefficient_max": float(observed_upper),
            "robust_coefficient_cap": float(cap),
        })
    filtered_models = list(models)
    client_records = {}
    for index in suspicious:
        update = matrix[index]
        recovered = torch.zeros_like(update)
        class_energy = {}
        class_raw_energy = {}
        for class_id, direction in repaired_directions.items():
            raw_coefficient = torch.dot(update, direction).clamp_min(0.0)
            coefficient = (
                torch.minimum(
                    raw_coefficient, class_coefficient_caps[class_id])
                if robust_coefficient_clip else raw_coefficient)
            coefficient = (coefficient
                           * float(class_release_weights[class_id]))
            component = coefficient * direction
            recovered = recovered + component
            class_energy[str(class_id)] = float(component.norm())
            class_raw_energy[str(class_id)] = float(raw_coefficient)
        recovered_norm = recovered.norm()
        if float(recovered_norm) > 1e-12:
            recovered = recovered * torch.minimum(
                torch.ones((), device=device),
                maximum_norm / recovered_norm.clamp_min(1e-12))
        filtered_models[index] = _model_from_update_vector(
            reference_model, recovered, model_keys)
        client_records[str(index)] = {
            "uploaded_norm": float(update.norm()),
            "released_norm": float(recovered.norm()),
            "released_fraction": float(
                recovered.norm() / update.norm().clamp_min(1e-12)),
            "class_path_norms": class_energy,
            "class_path_raw_norms": class_raw_energy,
        }
    return filtered_models, {
        "repair_ready": True,
        "classes": class_records,
        "clients": client_records,
        "trusted_median_update_norm": float(trusted_norm),
        "norm_clip": float(norm_clip),
        "robust_coefficient_clip": bool(robust_coefficient_clip),
        "utility_deficit_gate": bool(utility_deficit_gate),
        "orthogonal_residual_released": False,
        "target_class_used": False,
        "attack_identity_used": False,
        "trigger_template_used": False,
        "dataset_identity_used": False,
        "estimand": (
            "positive LCCT-attributed update transport through stable clean "
            "class-risk descent paths"),
    }


def add_nonrecursive_semantic_adapter(
        reference_model, repaired_models, uploaded_models,
        suspicious_indices, utility_probe, release_dose=0.25,
        shards_per_class=3, norm_clip=1.0,
        robust_coefficient_clip=True, minimum_recall_gap=0.25,
        device="cpu"):
    """Add only novel clean-class paths to an already repaired update.

    ``repaired_models`` is the existing cross-intervention-stable CRFU path.
    The uploaded update is first passed through
    :func:`class_semantic_update_filter`, so no coordinate outside the clean
    class-risk descent span is available to this adapter.  Any positive
    component already represented by the repaired update is then removed;
    only the remaining semantic component is released at a bounded dose.

    The caller keeps this model outside the recursive safe-core path.  Hence
    this routine cannot redefine the next-round reference used by LCCT or by
    the causal interventions.
    """
    suspicious = sorted(set(int(value) for value in suspicious_indices))
    if not suspicious:
        return list(repaired_models), {
            "repair_ready": False,
            "reason": "no LCCT-attributed client",
        }
    semantic_models, semantic_diag = class_semantic_update_filter(
        reference_model, uploaded_models, suspicious, utility_probe,
        shards_per_class=shards_per_class, norm_clip=norm_clip,
        robust_coefficient_clip=robust_coefficient_clip,
        utility_deficit_gate=True, device=device)
    if not semantic_diag.get("repair_ready"):
        return list(repaired_models), {
            "repair_ready": False,
            "reason": "clean semantic filter unavailable",
            "semantic_filter": semantic_diag,
        }
    class_records = semantic_diag.get("classes", {})
    maximum_recall_gap = max(
        (float(record.get("utility_deficit_absolute_gap", 0.0))
         for record in class_records.values()), default=0.0)
    semantic_diag["maximum_absolute_recall_gap"] = float(
        maximum_recall_gap)
    semantic_diag["minimum_release_recall_gap"] = float(
        minimum_recall_gap)
    if maximum_recall_gap + 1e-8 < float(minimum_recall_gap):
        return list(repaired_models), {
            "repair_ready": False,
            "reason": "no material clean class-coverage deficit",
            "semantic_filter": semantic_diag,
            "maximum_absolute_recall_gap": float(maximum_recall_gap),
            "minimum_release_recall_gap": float(minimum_recall_gap),
            "attack_identity_used": False,
            "target_class_used": False,
            "dataset_identity_used": False,
        }

    dose = min(max(float(release_dose), 0.0), 1.0)
    adapted_models = list(repaired_models)
    client_records = {}
    for index in suspicious:
        repaired_update, repaired_keys = _model_update_vector(
            reference_model, repaired_models[index])
        semantic_update, semantic_keys = _model_update_vector(
            reference_model, semantic_models[index])
        if repaired_keys != semantic_keys:
            raise ValueError("repaired and semantic model keys differ")
        repaired_update = repaired_update.to(device)
        semantic_update = semantic_update.to(device)

        # Do not count a clean semantic direction twice.  Only a positive
        # projection already present in the repaired path is subtracted;
        # disagreement with a clean descent path remains correctable.
        repaired_energy = repaired_update.square().sum().clamp_min(1e-12)
        represented_scale = torch.clamp(
            torch.dot(semantic_update, repaired_update) / repaired_energy,
            min=0.0)
        novel_semantic = semantic_update - (
            represented_scale * repaired_update)
        adapter = dose * novel_semantic
        adapted_update = repaired_update + adapter
        adapted_models[index] = _model_from_update_vector(
            reference_model, adapted_update, repaired_keys)
        client_records[str(index)] = {
            "repaired_norm": float(repaired_update.norm()),
            "semantic_norm": float(semantic_update.norm()),
            "represented_scale": float(represented_scale),
            "novel_semantic_norm": float(novel_semantic.norm()),
            "released_adapter_norm": float(adapter.norm()),
            "adapted_norm": float(adapted_update.norm()),
        }
    return adapted_models, {
        "repair_ready": True,
        "release_dose": float(dose),
        "clients": client_records,
        "semantic_filter": semantic_diag,
        "recursive": False,
        "orthogonal_uploaded_residual_released": False,
        "attack_identity_used": False,
        "target_class_used": False,
        "trigger_template_used": False,
        "dataset_identity_used": False,
        "estimand": (
            "novel clean-class descent component of an LCCT-attributed "
            "update after cross-intervention CRFU repair"),
    }


def causal_interaction_subspace_repair(
        reference_model, models, clients, suspicious_indices,
        utility_probe=None, trials=2, budget=0.2, seed=0, device="cpu",
        benign_variance_keep=0.95, harmful_variance_keep=0.90,
        norm_clip=1.5):
    """Remove pairwise causal-interaction directions, preserving utility span.

    This prototype operates on current-round parameter updates.  A protected
    span combines benign-update directions and class-conditional clean-probe
    gradients.  Candidate harmful directions are suspicious-update pair sums
    weighted by their non-additive do(G), do(X) response on the clients'
    actually trained graphs.  No attack identity or target class is used.
    """
    suspicious = sorted(int(value) for value in suspicious_indices)
    benign = [idx for idx in range(len(models)) if idx not in suspicious]
    if len(suspicious) < 2 or len(benign) < 2:
        return list(models), [], {
            "repair_ready": False,
            "reason": "interaction repair needs >=2 suspicious and benign clients",
        }

    updates = []
    keys = None
    for model in models:
        update, model_keys = _model_update_vector(reference_model, model)
        updates.append(update.to(device))
        keys = model_keys
    matrix = torch.stack(updates)
    benign_matrix = matrix[benign]
    benign_basis = _row_subspace_basis(
        benign_matrix, variance_keep=benign_variance_keep,
        max_rank=max(len(benign) - 1, 1))
    utility_basis, utility_diag = _probe_utility_basis(
        reference_model, utility_probe, keys, device=device)
    protected_parts = [part for part in (benign_basis, utility_basis)
                       if part is not None and part.numel()]
    protected_basis = (_row_subspace_basis(
        torch.cat(protected_parts, dim=0), variance_keep=0.999,
        max_rank=sum(part.shape[0] for part in protected_parts))
        if protected_parts else None)

    profile_cache = {}
    pair_records = []
    harmful_rows = []
    harmful_weights = []
    for left_pos, left in enumerate(suspicious):
        for right in suspicious[left_pos + 1:]:
            pair_update = matrix[left] + matrix[right]
            pair_model = _model_from_update_vector(
                reference_model, pair_update, keys)
            interaction_values = []
            for graph_idx in (left, right):
                graph_client = clients[graph_idx]
                profile_seed = (int(seed) + int(graph_client.cid) * 1009)
                for model_idx in (left, right):
                    cache_key = (model_idx, graph_idx)
                    if cache_key not in profile_cache:
                        profile_cache[cache_key] = (
                            client_local_causal_profile(
                                reference_model, models[model_idx],
                                graph_client.x, graph_client.adj,
                                trials=max(int(trials), 2), budget=budget,
                                seed=profile_seed, device=device,
                                outcome="logits").mean(dim=0).to(device))
                pair_profile = client_local_causal_profile(
                    reference_model, pair_model,
                    graph_client.x, graph_client.adj,
                    trials=max(int(trials), 2), budget=budget,
                    seed=profile_seed, device=device,
                    outcome="logits").mean(dim=0).to(device)
                left_profile = profile_cache[(left, graph_idx)]
                right_profile = profile_cache[(right, graph_idx)]
                interaction = pair_profile - left_profile - right_profile
                scale = (left_profile.abs().mean()
                         + right_profile.abs().mean()).clamp(min=1e-8)
                interaction_values.append(interaction.abs().mean() / scale)
            interaction_score = torch.stack(interaction_values).mean()
            candidate = _project_off_basis(
                pair_update.reshape(1, -1), protected_basis).reshape(-1)
            candidate_norm = candidate.norm()
            if float(candidate_norm) > 1e-10:
                harmful_rows.append(candidate / candidate_norm)
                harmful_weights.append(interaction_score.clamp(min=1e-6))
            pair_records.append({
                "clients": [int(left), int(right)],
                "interaction_score": float(interaction_score.detach()),
                "candidate_norm": float(candidate_norm.detach()),
            })

    if not harmful_rows:
        return list(models), [], {
            "repair_ready": False,
            "reason": "no harmful direction outside protected span",
            "pairs": pair_records,
            "utility_basis": utility_diag,
        }
    weighted_rows = torch.stack(harmful_rows) * torch.stack(
        harmful_weights).sqrt().reshape(-1, 1)
    harmful_basis = _row_subspace_basis(
        weighted_rows, variance_keep=harmful_variance_keep,
        max_rank=max(len(suspicious) - 1, 1))
    if harmful_basis is None or harmful_basis.numel() == 0:
        return list(models), [], {
            "repair_ready": False,
            "reason": "harmful subspace rank is zero",
            "pairs": pair_records,
            "utility_basis": utility_diag,
        }

    suspicious_matrix = matrix[suspicious]
    protected_projection = (
        (suspicious_matrix @ protected_basis.T) @ protected_basis
        if protected_basis is not None and protected_basis.numel()
        else torch.zeros_like(suspicious_matrix))
    unprotected = suspicious_matrix - protected_projection
    harmful_projection = (unprotected @ harmful_basis.T) @ harmful_basis
    uncertain_residual = unprotected - harmful_projection
    harmful_fraction = (harmful_projection.norm(dim=1)
                        / unprotected.norm(dim=1).clamp(min=1e-12))
    residual_release_multipliers = (1.0 - harmful_fraction).clamp(
        min=0.0, max=1.0).square()
    repaired_matrix = (protected_projection
                       + uncertain_residual
                       * residual_release_multipliers[:, None])
    benign_norm = benign_matrix.norm(dim=1).median().clamp(min=1e-10)
    max_norm = float(norm_clip) * benign_norm
    repaired_norm = repaired_matrix.norm(dim=1).clamp(min=1e-12)
    scales = torch.minimum(torch.ones_like(repaired_norm),
                           max_norm / repaired_norm)
    repaired_matrix = repaired_matrix * scales[:, None]
    repaired_models = list(models)
    for row, idx in enumerate(suspicious):
        repaired_models[idx] = _model_from_update_vector(
            reference_model, repaired_matrix[row], keys)
    removed_fraction = ((suspicious_matrix - repaired_matrix).norm(dim=1)
                        / suspicious_matrix.norm(dim=1).clamp(min=1e-12))
    protected_overlap = (
        (harmful_basis @ protected_basis.T).abs().max()
        if protected_basis is not None and protected_basis.numel()
        else torch.tensor(0.0, device=harmful_basis.device))
    return repaired_models, suspicious, {
        "repair_ready": True,
        "repaired_clients": suspicious,
        "pairs": pair_records,
        "harmful_rank": int(harmful_basis.shape[0]),
        "benign_basis_rank": int(benign_basis.shape[0]),
        "protected_basis_rank": (int(protected_basis.shape[0])
                                 if protected_basis is not None else 0),
        "utility_basis": utility_diag,
        "removed_update_fraction": [float(value) for value in removed_fraction],
        "harmful_projection_fraction": [
            float(value) for value in harmful_fraction],
        "residual_release_multipliers": [
            float(value) for value in residual_release_multipliers],
        "max_harmful_protected_overlap": float(protected_overlap),
        "norm_clip_relative_to_benign_median": float(norm_clip),
        "target_class_used": False,
        "attack_identity_used": False,
        "clean_probe_used": bool(utility_probe is not None),
        "clean_probe_labels_used": bool(utility_probe is not None),
        "estimand": (
            "pairwise non-additive Delta_G Delta_theta / "
            "Delta_A Delta_X Delta_theta parameter subspace"),
    }


def _affine_subspace(values, rank):
    mean = values.mean(dim=0)
    centered = values - mean
    if centered.shape[0] <= 1 or float(centered.norm()) <= 1e-12:
        return mean, centered.new_zeros((0, centered.shape[1]))
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    usable = min(max(int(rank), 0), centered.shape[0] - 1, vh.shape[0])
    return mean, vh[:usable]


def _subspace_distance(value, mean, basis):
    centered = value - mean
    if basis.numel():
        centered = centered - (centered @ basis.T) @ basis
    return centered.square().mean().sqrt()


@torch.no_grad()
def causal_mechanism_subspace_repair(
        reference_model, models, clients, suspicious_indices, profiles,
        trials=4, budget=0.2, seed=0, device="cpu",
        doses=(1.0, 0.75, 0.5, 0.25), variance_keep=0.90,
        threshold_multiplier=1.25, dose_caps=None, layerwise=False):
    """Keep the largest update dose compatible with benign do(G) responses.

    The representation being projected is a fixed-length summary of
    Delta_G Delta_theta and Delta_A Delta_X Delta_theta, not the parameter
    vector.  A scalar dose preserves the suspicious client's unique Non-IID
    direction instead of replacing it with a benign-client update.
    """
    suspicious = sorted(int(value) for value in suspicious_indices)
    benign = [idx for idx in range(len(models)) if idx not in suspicious]
    repaired_models = list(models)
    if not suspicious or len(benign) < 3:
        return repaired_models, [], {"repair_ready": False,
                                     "reason": "insufficient benign clients"}
    profile_tensor = torch.as_tensor(
        profiles, dtype=torch.float32, device=device)
    benign_profiles = profile_tensor[benign]
    center = benign_profiles.median(dim=0).values
    mad = (benign_profiles - center).abs().median(dim=0).values
    scale_floor = 1e-3 * center.abs().median().clamp(min=1e-6)
    scale = torch.maximum(1.4826 * mad,
                          torch.full_like(mad, scale_floor)).clamp(min=1e-8)
    benign_standardized = (benign_profiles - center) / scale

    singular = torch.linalg.svdvals(
        benign_standardized - benign_standardized.mean(dim=0))
    energy = singular.square()
    if float(energy.sum()) > 1e-12:
        cumulative = torch.cumsum(energy, dim=0) / energy.sum()
        rank = int(torch.searchsorted(
            cumulative, torch.tensor(float(variance_keep),
                                     device=device)).item()) + 1
    else:
        rank = 0
    rank = min(rank, len(benign) - 2)
    subspace_mean, subspace_basis = _affine_subspace(
        benign_standardized, rank)

    # Leave-one-benign-client-out errors set the acceptance scale without a
    # target label, attack identity, or a clean graph.
    loo_distances = []
    for row in range(len(benign)):
        keep = torch.ones(len(benign), dtype=torch.bool, device=device)
        keep[row] = False
        loo_mean, loo_basis = _affine_subspace(
            benign_standardized[keep], min(rank, len(benign) - 2))
        loo_distances.append(_subspace_distance(
            benign_standardized[row], loo_mean, loo_basis))
    subspace_threshold = (torch.stack(loo_distances).max()
                          * float(threshold_multiplier) + 1e-6)
    benign_total = benign_standardized.square().mean(dim=1).sqrt()
    total_threshold = (benign_total.max()
                       * float(threshold_multiplier) + 1e-6)

    accepted = []
    client_records = {}
    for idx in suspicious:
        update, keys = _model_update_vector(reference_model, models[idx])
        dose_records = []
        selected_model = None
        selected_dose = 0.0
        selected_group_doses = None
        if layerwise:
            reference_state = reference_model.state_dict()
            current_state = models[idx].state_dict()
            groups = sorted({key.split(".", 1)[0] for key in keys})
            group_energy = {}
            for group in groups:
                group_energy[group] = sum(float(
                    (current_state[key].float()
                     - reference_state[key].float()).square().sum())
                    for key in keys if key.split(".", 1)[0] == group)
            cap_record = (dose_caps.get(idx, {})
                          if dose_caps is not None else {})
            choices = []
            for group in groups:
                cap = float(cap_record.get(group, 1.0))
                choices.append([float(value) for value in doses
                                if float(value) <= cap + 1e-12] + [0.0])
            candidates = []
            total_energy = max(sum(group_energy.values()), 1e-12)
            for combination in itertools.product(*choices):
                gates = dict(zip(groups, combination))
                retained = sum(group_energy[group] * gates[group] ** 2
                               for group in groups)
                retention = float(np.sqrt(retained / total_energy))
                if retention > 0.0:
                    candidates.append((retention, gates))
            candidates.sort(key=lambda item: item[0], reverse=True)
            candidate_iterator = candidates
        else:
            cap = (float(dose_caps.get(idx, 1.0))
                   if dose_caps is not None else 1.0)
            allowed_doses = [float(value) for value in doses
                             if float(value) <= cap + 1e-12]
            candidate_iterator = [(dose, dose) for dose in sorted(
                allowed_doses, reverse=True)]
        for retention, intervention in candidate_iterator:
            if layerwise:
                candidate = _model_from_group_doses(
                    reference_model, models[idx], intervention)
                dose_label = {key: float(value)
                              for key, value in intervention.items()}
            else:
                candidate = _model_from_update_vector(
                    reference_model, float(intervention) * update, keys)
                dose_label = float(intervention)
            candidate_trials = client_local_causal_profile(
                reference_model, candidate, clients[idx].x, clients[idx].adj,
                trials=trials, budget=budget,
                seed=int(seed) + int(clients[idx].cid) * 1009,
                device=device, outcome="logits")
            candidate_profile = candidate_trials.mean(dim=0).to(device)
            standardized = (candidate_profile - center) / scale
            subspace_distance = _subspace_distance(
                standardized, subspace_mean, subspace_basis)
            total_distance = standardized.square().mean().sqrt()
            passed = bool(subspace_distance <= subspace_threshold
                          and total_distance <= total_threshold)
            dose_records.append({
                "dose": dose_label,
                "parameter_retention": float(retention),
                "subspace_distance": float(subspace_distance),
                "total_distance": float(total_distance),
                "passed": passed,
            })
            if passed:
                selected_model = candidate
                if layerwise:
                    selected_group_doses = dose_label
                    selected_dose = float(retention)
                else:
                    selected_dose = float(intervention)
                break
        if selected_model is not None:
            repaired_models[idx] = selected_model
            accepted.append(idx)
        client_records[str(idx)] = {
            "incoming_dose_cap": (cap_record if layerwise else float(cap)),
            "selected_dose": selected_dose,
            "selected_group_doses": selected_group_doses,
            "parameter_groups": groups if layerwise else None,
            "dose_records": dose_records,
        }
    return repaired_models, accepted, {
        "repair_ready": bool(accepted),
        "accepted_clients": accepted,
        "subspace_rank": int(rank),
        "variance_keep": float(variance_keep),
        "subspace_threshold": float(subspace_threshold),
        "total_threshold": float(total_threshold),
        "threshold_multiplier": float(threshold_multiplier),
        "dose_grid": sorted([float(value) for value in doses], reverse=True),
        "layerwise_do_theta": bool(layerwise),
        "clients": client_records,
        "target_class_used": False,
        "labels_used": False,
        "clean_graph_used": False,
        "energy_used": False,
        "attack_identity_used": False,
        "estimand": (
            "benign affine subspace of Delta_G Delta_theta and "
            "Delta_A Delta_X Delta_theta response summaries"),
    }


@torch.no_grad()
def causal_transfer_weight_multipliers(
        reference_model, models, clients, suspicious_indices,
        retained_indices, trials=4, budget=0.2, seed=0, device="cpu",
        strength=1.5, floor=0.35, score_cap=4.0,
        min_consensus=0.90):
    """Estimate target-free inherited suspicious-direction risk.

    LCCT removes the clients with the strongest causal anomaly, but a retained
    client can still transport part of the same mechanism after training from
    a contaminated global model.  Each client therefore reports only a
    fixed-length, reliability-weighted class-response vector under randomized
    ``do(G)`` worlds.  The suspicious cohort defines an unnamed response
    direction; retained updates with unusually large positive projection on
    that direction are softly attenuated.

    This is deliberately a weighting rule rather than another rejection rule.
    A positive floor keeps every LCCT-retained client represented, which is
    important when clean class information is distributed non-uniformly.
    """
    client_count = len(models)
    suspicious = sorted(set(int(v) for v in suspicious_indices))
    retained = sorted(set(int(v) for v in retained_indices))
    multipliers = torch.ones(client_count, dtype=torch.float32)
    empty_diag = {
        "enabled": False,
        "reason": "insufficient suspicious or retained clients",
        "target_class_used": False,
        "trigger_template_used": False,
        "attack_identity_used": False,
        "full_graph_uploaded": False,
    }
    if not suspicious or len(retained) < 3:
        return multipliers, empty_diag

    response_vectors = []
    response_reliability = []
    for idx, (client, model) in enumerate(zip(clients, models)):
        response, _, node_reliability = causal_node_transport_vectors(
            reference_model, model, client.x, client.adj,
            trials=max(2, int(trials)), budget=float(budget),
            seed=int(seed) + idx * 1009, device=device)
        train_idx = torch.as_tensor(
            client.train_idx, dtype=torch.long, device=response.device)
        if train_idx.numel() == 0:
            vector = response.mean(dim=0)
            reliability = node_reliability.mean()
        else:
            node_weight = node_reliability[train_idx].clamp(min=0.0, max=1.0)
            vector = (response[train_idx] * node_weight[:, None]).sum(dim=0)
            vector = vector / node_weight.sum().clamp(min=1e-8)
            reliability = node_weight.mean()
        response_vectors.append(vector.detach().float().cpu())
        response_reliability.append(float(reliability.detach().cpu()))
    vectors = torch.stack(response_vectors)

    # Normalize each suspicious vote before averaging so one high-magnitude
    # graph cannot name the cohort direction by itself.  Subtracting the
    # retained median isolates the mechanism shared specifically by the
    # quarantined cohort rather than ordinary task learning.
    retained_center = vectors[retained].median(dim=0).values
    suspicious_residual = vectors[suspicious] - retained_center
    suspicious_unit = suspicious_residual / suspicious_residual.norm(
        dim=1, keepdim=True).clamp(min=1e-8)
    direction = suspicious_unit.mean(dim=0)
    direction_norm = float(direction.norm())
    if direction_norm < float(min_consensus):
        diag = dict(empty_diag)
        diag["reason"] = "suspicious causal direction consensus below threshold"
        diag["suspicious_direction_consensus"] = direction_norm
        diag["minimum_direction_consensus"] = float(min_consensus)
        return multipliers, diag
    direction = direction / direction.norm().clamp(min=1e-8)

    projections = torch.mv(vectors - retained_center, direction)
    retained_projection = projections[retained]
    center = retained_projection.median()
    mad = (retained_projection - center).abs().median()
    magnitude_floor = (1e-3 * retained_projection.abs().median()
                       ).clamp(min=1e-6)
    scale = torch.maximum(1.4826 * mad, magnitude_floor)
    risk_z = torch.relu((retained_projection - center)
                        / scale.clamp(min=1e-8))
    if float(score_cap) > 0.0:
        risk_z = risk_z.clamp(max=float(score_cap))
    raw = torch.exp(-float(strength) * risk_z)
    raw = raw.clamp(min=float(floor), max=1.0)
    multipliers[retained] = raw

    return multipliers, {
        "enabled": True,
        "estimand": (
            "positive retained-client projection on the quarantined-cohort "
            "signed do(G) x do(theta) response direction"),
        "suspicious_clients": suspicious,
        "retained_clients": retained,
        "suspicious_direction_consensus": direction_norm,
        "minimum_direction_consensus": float(min_consensus),
        "response_reliability": response_reliability,
        "projections": [float(v) for v in projections],
        "retained_projection_center": float(center),
        "retained_projection_scale": float(scale),
        "retained_risk_z": {
            str(idx): float(value)
            for idx, value in zip(retained, risk_z)},
        "raw_multipliers": [float(v) for v in multipliers],
        "strength": float(strength),
        "floor": float(floor),
        "target_class_used": False,
        "trigger_template_used": False,
        "attack_identity_used": False,
        "client_labels_used": False,
        "full_graph_uploaded": False,
        "fixed_length_client_summary": True,
    }


@torch.no_grad()
def _restore_bounded_classifier_head(reference_model, source_models,
                                     target_models, suspicious_indices,
                                     trusted_indices, norm_clip=1.0):
    """Restore only the bounded final-head update after body projection.

    The last parameterized module is treated as the task head.  This avoids
    hard-coding a GCN layer name and leaves the model's forward logic intact.
    The head direction comes from the invariant-repaired client, while its
    norm is capped by the median trusted-client head update norm.
    """
    parameterized_modules = [
        name for name, module in reference_model.named_modules()
        if name and any(True for _ in module.parameters(recurse=False))]
    if not parameterized_modules:
        return list(target_models), {
            "applied": False, "reason": "classifier_head_not_found"}
    head_prefix = parameterized_modules[-1] + "."
    reference_state = reference_model.state_dict()
    head_keys = [key for key in reference_state if key.startswith(head_prefix)]
    if not head_keys:
        return list(target_models), {
            "applied": False, "reason": "classifier_head_state_not_found"}

    def head_delta(model):
        state = model.state_dict()
        return torch.cat([
            (state[key].detach().cpu() - reference_state[key].detach().cpu())
            .reshape(-1).float() for key in head_keys])

    trusted_norms = [head_delta(source_models[idx]).norm()
                     for idx in trusted_indices]
    if not trusted_norms:
        return list(target_models), {
            "applied": False, "reason": "no_trusted_head_reference"}
    max_norm = (torch.stack(trusted_norms).median().clamp_min(1e-12)
                * max(float(norm_clip), 0.0))
    repaired_models = list(target_models)
    scales = {}
    for idx in suspicious_indices:
        source_state = source_models[idx].state_dict()
        target_state = copy.deepcopy(target_models[idx].state_dict())
        source_norm = head_delta(source_models[idx]).norm().clamp_min(1e-12)
        scale = min(1.0, float(max_norm / source_norm))
        for key in head_keys:
            reference_value = reference_state[key].to(
                device=target_state[key].device,
                dtype=target_state[key].dtype)
            source_value = source_state[key].to(
                device=target_state[key].device,
                dtype=target_state[key].dtype)
            target_state[key] = reference_value + scale * (
                source_value - reference_value)
        restored = copy.deepcopy(target_models[idx])
        restored.load_state_dict(target_state)
        repaired_models[idx] = restored
        scales[str(int(idx))] = float(scale)
    return repaired_models, {
        "applied": True,
        "head_prefix": head_prefix,
        "head_keys": head_keys,
        "trusted_median_head_update_norm": float(max_norm) / max(
            float(norm_clip), 1e-12),
        "norm_clip": float(norm_clip),
        "client_scales": scales,
        "attack_identity_used": False,
        "target_class_used": False,
        "trigger_template_used": False,
    }


@torch.no_grad()
def _blend_invariant_self_residual(projected_models, source_models,
                                   suspicious_indices, residual_keep,
                                   protected_prefix=None):
    """Restore a bounded share of a client's own invariant residual.

    ``source_models`` already passed the client's cross-intervention gradient
    stability repair.  The blend therefore does not reopen the raw uploaded
    update: it interpolates only between the trusted-span projection and that
    repaired invariant path.  A separately restored classifier head can be
    protected from the interpolation through ``protected_prefix``.
    """
    fraction = min(max(float(residual_keep), 0.0), 1.0)
    blended = list(projected_models)
    for idx in suspicious_indices:
        projected = copy.deepcopy(projected_models[idx])
        projected_parameters = dict(projected.named_parameters())
        source_parameters = dict(source_models[idx].named_parameters())
        for name, parameter in projected_parameters.items():
            if protected_prefix and name.startswith(protected_prefix):
                continue
            source = source_parameters[name].to(
                device=parameter.device, dtype=parameter.dtype)
            parameter.copy_(parameter + fraction * (source - parameter))
        if hasattr(projected, "_feats"):
            projected._feats = None
        if hasattr(projected, "_causal_gate"):
            projected._causal_gate = None
        blended[idx] = projected
    return blended


def local_causal_energy_aggregate(reference_model, models, clients, state,
                                  trials=4, budget=0.2, keep_ratio=0.7,
                                  z_margin=1.5, penalty=2.0, seed=0,
                                  device="cpu", outcome="logits",
                                  persistent=True, min_reliability=0.8,
                                  gap_ratio=1.5, repair=True,
                                  repair_epochs=5, repair_lr=0.01,
                                  repair_node_z=2.5,
                                  repair_mode="none", repair_warmup=5,
                                  invariant_sign=0.5,
                                  invariant_release_cap=0.03,
                                  invariant_release_utility_guard=False,
                                  invariant_nonrecursive_release=False,
                                  invariant_release_max_class_drop=0.04,
                                  invariant_release_contrast_guard=False,
                                  invariant_release_contrast_threshold=100.0,
                                  invariant_release_contrast_min_cap=0.03,
                                  invariant_release_contrast_hard_fallback=False,
                                  invariant_projection_release=False,
                                  invariant_projection_variance_keep=0.95,
                                  invariant_projection_norm_clip=1.0,
                                  invariant_projection_preserve_head=False,
                                  invariant_projection_utility_shards=1,
                                  invariant_projection_residual_guard=False,
                                  invariant_projection_residual_max=0.06,
                                  invariant_local_label_rescue=False,
                                  invariant_rescue_transport_z=1.5,
                                  invariant_high_contrast_reconstruction=False,
                                  invariant_reconstruction_epochs=2,
                                  invariant_reconstruction_release_cap=0.5,
                                  invariant_stable_private_amplitude=False,
                                  invariant_continuous_label_weight=False,
                                  invariant_coalition_residual_block=False,
                                  invariant_coalition_variance_keep=0.99,
                                  invariant_coalition_norm_clip=1.0,
                                  invariant_coalition_block_strength=1.0,
                                  invariant_coalition_min_pair_cosine=-1.0,
                                  invariant_reconstruction_coverage_gate=False,
                                  invariant_reconstruction_coverage_floor=0.5,
                                  invariant_reconstruction_coverage_min_support=3,
                                  invariant_prototype_rescue=False,
                                  invariant_prototype_margin=0.0,
                                  invariant_prototype_policy_gate=False,
                                  invariant_prototypes_per_class=1,
                                  invariant_release_stability_only=False,
                                  semantic_path_release=False,
                                  semantic_path_release_weight=0.25,
                                  semantic_path_shards_per_class=3,
                                  semantic_path_norm_clip=1.0,
                                  semantic_path_robust_coefficient_clip=False,
                                  semantic_path_min_recall_gap=0.25,
                                  final_round=False,
                                  temporal_quarantine=False,
                                  temporal_quarantine_decay=0.8,
                                  temporal_quarantine_warmup=3,
                                   score_weighting="absolute",
                                   score_cap=4.0,
                                   pardon_floor=0.10,
                                   pardon_ceiling=0.50,
                                   pardon_variance_keep=0.95,
                                   pardon_hard_streak=2,
                                   pardon_release_streak=2,
                                  base_weights=None,
                                  interaction_probe=None,
                                  semantic_path_probe=None,
                                  release_calibration_probe=None,
                                  release_monitor_probe=None,
                                   trusted_utility_reweight=False,
                                   trusted_utility_temperature=0.25,
                                   trusted_utility_max_class_drop=0.04,
                                   trusted_utility_max_mix=0.25,
                                   trusted_utility_max_quarantine_changes=1,
                                  ci_heterogeneity_threshold=0.10,
                                  transfer_strength=1.5,
                                  transfer_floor=0.35,
                                  transfer_mix=0.5,
                                  transfer_min_consensus=0.90,
                                  ego_focus_fraction=0.15,
                                  ego_model_steps=3,
                                  ego_consistency_weight=3.0,
                                  ego_max_model_steps=6,
                                  ego_residual_ratio_threshold=5.0):
    """Aggregate using client-local do(theta) x do(G) response profiles."""
    configured_repair_mode = str(repair_mode)
    invariant_shadow_guarded = bool(
        configured_repair_mode == "invariant_shadow_guarded")
    if configured_repair_mode in (
            "shadow_ci_adaptive", "shadow_ci_strong_adaptive",
            "shadow_ci_transfer_adaptive", "shadow_ci_ego_adaptive",
            "shadow_ci_ego_transfer_adaptive",
            "shadow_ci_ego_guarded_adaptive",
            "invariant_shadow_guarded"):
        class_count = max(int(client.y.max().item()) for client in clients) + 1
        distributions = []
        masses = []
        for client in clients:
            index = torch.as_tensor(
                client.train_idx, dtype=torch.long, device=client.y.device)
            counts = torch.bincount(
                client.y[index].detach().cpu(), minlength=class_count).float()
            masses.append(counts.sum().clamp_min(1.0))
            distributions.append(
                (counts + 1e-6) / (counts.sum() + 1e-6 * class_count))
        distribution = torch.stack(distributions)
        mass = torch.stack(masses)
        pooled = (distribution * mass[:, None]).sum(dim=0) / mass.sum()
        heterogeneity = float((distribution * (
            distribution.clamp_min(1e-8).log()
            - pooled.clamp_min(1e-8).log())).sum(dim=1).mean())
        if configured_repair_mode == "shadow_ci_strong_adaptive":
            low_heterogeneity_mode = "shadow_ci_strong"
        elif configured_repair_mode == "shadow_ci_transfer_adaptive":
            low_heterogeneity_mode = "shadow_ci_transfer"
        elif configured_repair_mode in (
                "shadow_ci_ego_adaptive",
                "shadow_ci_ego_guarded_adaptive",
                "invariant_shadow_guarded"):
            low_heterogeneity_mode = "shadow_ci_ego"
        elif configured_repair_mode == "shadow_ci_ego_transfer_adaptive":
            low_heterogeneity_mode = "shadow_ci_ego_transfer"
        else:
            low_heterogeneity_mode = "shadow_ci"
        high_heterogeneity = (
            heterogeneity > float(ci_heterogeneity_threshold))
        # The guarded variant does not disable CRFU wholesale under Non-IID.
        # It trains a candidate below and commits it only when the learned
        # intervention leaves a persistent residual after the minimum steps.
        if configured_repair_mode in (
                "shadow_ci_ego_guarded_adaptive",
                "invariant_shadow_guarded"):
            repair_mode = low_heterogeneity_mode
        else:
            repair_mode = (low_heterogeneity_mode
                           if not high_heterogeneity else "shadow_only")
        state["ci_adaptation"] = {
            "metric": "mean_client_KL_to_pooled_label_distribution",
            "heterogeneity": heterogeneity,
            "threshold": float(ci_heterogeneity_threshold),
            "high_heterogeneity": bool(high_heterogeneity),
            "effective_repair_mode": repair_mode,
            "split_flag_used": False,
            "attack_identity_used": False,
            "target_class_used": False,
        }
    invariant_release_enabled = bool(
        invariant_shadow_guarded
        and state.get("ci_adaptation", {}).get(
            "high_heterogeneity", False))
    scores_np, profiles, _, reliability = compute_local_causal_scores(
        reference_model, models, clients, trials=trials, budget=budget,
        seed=seed, device=device, outcome=outcome)
    scores = torch.as_tensor(scores_np, dtype=torch.float32)
    client_count = len(models)
    min_keep = min(client_count, max(
        1, int(np.ceil(float(keep_ratio) * client_count))))
    max_quarantine = client_count - min_keep
    newly_quarantined = []
    separation_ratio = 0.0
    temporal_quarantine_enabled = bool(
        configured_repair_mode in (
            "invariant_shadow_guarded",
            "shadow_ci_ego_guarded_adaptive")
        and temporal_quarantine)
    temporal_selection = []
    temporal_round = int(state.get("temporal_quarantine_round", 0)) + 1
    if temporal_quarantine_enabled and max_quarantine > 0:
        # LCCT scores can be correctly ordered overall while an early
        # Non-IID round temporarily promotes a rare benign client. Accumulate
        # scale-free causal rank evidence so the protected set can correct
        # that early decision instead of making the first top-k list sticky.
        # Raw suspicious updates are still blocked from round one.
        ranked_now = torch.argsort(scores, descending=True)
        rank_evidence = torch.zeros(client_count, dtype=torch.float32)
        if client_count > 1:
            rank_evidence[ranked_now] = torch.linspace(
                1.0, 0.0, client_count, dtype=torch.float32)
        else:
            rank_evidence[ranked_now] = 1.0
        old_evidence = state.get("temporal_causal_rank_evidence")
        if old_evidence is None or len(old_evidence) != client_count:
            accumulated_evidence = rank_evidence
        else:
            decay = min(max(float(temporal_quarantine_decay), 0.0), 0.999)
            accumulated_evidence = (
                decay * torch.as_tensor(old_evidence, dtype=torch.float32)
                + (1.0 - decay) * rank_evidence)
        state["temporal_causal_rank_evidence"] = [
            float(value) for value in accumulated_evidence]
        state["temporal_quarantine_round"] = temporal_round
        # Before the short evidence warmup, current-round top-k still forms a
        # provisional firewall. Afterwards, the temporal estimate controls
        # membership and may replace an early false positive.
        if temporal_round < max(int(temporal_quarantine_warmup), 1):
            initial_firewall = state.get("provisional", [])
            temporal_selection = (
                [int(value) for value in initial_firewall]
                if initial_firewall else [
                    int(value)
                    for value in ranked_now[:max_quarantine].tolist()])
            temporal_source = "initial_firewall_evidence_warmup"
        else:
            temporal_selection = [
                int(value) for value in torch.argsort(
                    accumulated_evidence, descending=True
                )[:max_quarantine].tolist()]
            temporal_source = "temporal_causal_consensus"
        state["temporal_quarantine_source"] = temporal_source
    causal_trigger_block_mode = bool(
        repair and repair_mode == "causal_trigger_block")
    causal_pardon_mode = bool(
        repair and repair_mode in (
            "causal_pardon", "causal_trigger_block"))
    if (persistent and not causal_pardon_mode
            and not temporal_quarantine_enabled
            and not state.get("quarantined") and max_quarantine > 0):
        ranked = torch.argsort(scores, descending=True)
        suspicious_floor = float(scores[ranked[max_quarantine - 1]])
        benign_ceiling = float(scores[ranked[max_quarantine]])
        separation_ratio = suspicious_floor / max(benign_ceiling, 1e-8)
        if (reliability is not None
                and reliability >= float(min_reliability)
                and separation_ratio >= float(gap_ratio)):
            newly_quarantined = [
                int(v) for v in ranked[:max_quarantine].tolist()]
            state["quarantined"] = list(newly_quarantined)
            state["quarantine_round"] = int(state.get("round", 0)) + 1
    quarantined = set(int(v) for v in state.get("quarantined", []))
    provisional = set()
    if (repair and repair_mode in ("shadow_only", "shadow_ci",
                                  "shadow_ci_strong",
                                  "shadow_ci_transfer",
                                  "shadow_ci_ego",
                                  "shadow_ci_ego_transfer",
                                  "stable_gate", "query_cate")
            and max_quarantine > 0):
        previous_provisional = set(int(value) for value in
                                   state.get("provisional", []))
        if temporal_quarantine_enabled:
            state["provisional"] = list(temporal_selection)
            # Keep the original LCCT state contract intact. Some downstream
            # recovery components read ``quarantined`` directly, whereas the
            # shadow path reads ``provisional`` first. Synchronizing both
            # makes temporal selection a membership update, not a different
            # defense path, and preserves legacy behavior whenever the set is
            # unchanged.
            previous_quarantined = set(int(value) for value in
                                       state.get("quarantined", []))
            state["quarantined"] = list(temporal_selection)
            state.setdefault("quarantine_round", temporal_round)
            newly_quarantined = sorted(
                set(temporal_selection) - previous_quarantined)
            if previous_provisional != set(temporal_selection):
                state.setdefault("temporal_quarantine_changes", []).append({
                    "round": temporal_round,
                    "from": sorted(previous_provisional),
                    "to": sorted(temporal_selection),
                    "source": state.get("temporal_quarantine_source"),
                })
            state.setdefault("provisional_round", temporal_round)
        elif not state.get("provisional"):
            ranked = torch.argsort(scores, descending=True)
            state["provisional"] = [
                int(value) for value in ranked[:max_quarantine].tolist()]
            state["provisional_round"] = int(state.get("round", 0)) + 1
        provisional = set(int(value) for value in state["provisional"])
        # The provisional set is intentionally sticky: it defines the
        # do(Delta_theta_Q=0) shadow trajectory before sequential evidence is
        # strong enough for permanent attribution.
        quarantined = set(provisional)
    repair_records = {}
    aggregation_models = list(models)
    repaired_indices = []
    invariant_projection_ready = False
    invariant_projection_source_models = None
    invariant_projection_head_prefix = None
    repair_weight_multipliers = torch.ones(
        client_count, dtype=torch.float32)
    current_round = int(state.get("round", 0)) + 1
    quarantine_round = int(state.get("quarantine_round", current_round))
    warmup_complete = (current_round - quarantine_round) >= int(repair_warmup)
    invariant_release_contrast = None
    if quarantined:
        trusted_indices_for_contrast = [
            idx for idx in range(client_count) if idx not in quarantined]
        if trusted_indices_for_contrast:
            suspicious_mean = float(scores[list(quarantined)].mean())
            trusted_mean = float(scores[trusted_indices_for_contrast].mean())
            invariant_release_contrast = (
                suspicious_mean / max(trusted_mean, 1e-12))
    contrast_threshold = max(
        float(invariant_release_contrast_threshold), 1.0)
    high_contrast_legacy_fallback = bool(
        invariant_release_contrast_guard
        and invariant_release_contrast_hard_fallback
        and invariant_release_contrast is not None
        and invariant_release_contrast > contrast_threshold)

    semantic_path_ready = False
    # The legacy semantic path replaces suspicious uploads before every
    # recursive aggregation.  Safe-Core mode instead owns a final-round,
    # nonrecursive adapter below.  These paths must be mutually exclusive:
    # otherwise enabling the adapter silently changes the first 99 rounds
    # even when its final coverage gate rejects release.
    if (semantic_path_release and not invariant_nonrecursive_release
            and quarantined and repair
            and repair_mode in (
                "shadow_only", "shadow_ci", "shadow_ci_strong",
                "shadow_ci_transfer", "shadow_ci_ego",
                "shadow_ci_ego_transfer")):
        aggregation_models, semantic_path_diag = (
            class_semantic_update_filter(
                reference_model, models, quarantined,
                utility_probe=(semantic_path_probe
                               if semantic_path_probe is not None
                               else interaction_probe),
                shards_per_class=semantic_path_shards_per_class,
                norm_clip=semantic_path_norm_clip,
                robust_coefficient_clip=(
                    semantic_path_robust_coefficient_clip),
                device=device))
        semantic_path_diag["release_weight"] = float(
            semantic_path_release_weight)
        semantic_path_diag["aggregation_stage"] = "every_round_pre_average"
        repair_records["class_semantic_update_filter"] = semantic_path_diag
        if semantic_path_diag.get("repair_ready"):
            semantic_path_ready = True
            repaired_indices = sorted(quarantined)

    # Hybrid recovery keeps the original server-side counterfactual repair,
    # while allowing only cross-intervention-stable local gradients from a
    # quarantined client to receive a small, risk-weighted contribution.
    if (invariant_release_enabled and quarantined and repair
            and warmup_complete):
        clean_prototype_bundle = (
            _trusted_hidden_prototypes(
                reference_model,
                (release_calibration_probe
                 if release_calibration_probe is not None
                 else interaction_probe),
                device=device,
                prototypes_per_class=(
                    1 if high_contrast_legacy_fallback
                    else invariant_prototypes_per_class))
            if invariant_prototype_rescue else None)
        for idx in sorted(quarantined):
            repaired_model, repair_diag = (
                train_client_with_invariant_causal_gradients(
                    reference_model, clients[idx], epochs=repair_epochs,
                    lr=repair_lr, worlds=trials, budget=budget,
                    sign_agreement=invariant_sign,
                    seed=int(seed) + idx * 1009, device=device,
                    client_model=models[idx],
                    local_label_rescue=(
                        invariant_local_label_rescue
                        and not high_contrast_legacy_fallback),
                    rescue_transport_z=invariant_rescue_transport_z,
                    clean_start_label_reconstruction=(
                        invariant_high_contrast_reconstruction
                        and high_contrast_legacy_fallback),
                    clean_start_reconstruction_epochs=(
                        invariant_reconstruction_epochs),
                    continuous_causal_label_weight=(
                        invariant_continuous_label_weight
                        and high_contrast_legacy_fallback),
                    clean_prototype_bundle=clean_prototype_bundle,
                    prototype_margin=invariant_prototype_margin,
                    prototype_policy=state.get("shadow_ci_policy"),
                    prototype_require_policy=(
                        invariant_prototype_policy_gate)))
            aggregation_models[idx] = repaired_model
            repair_records[str(idx)] = repair_diag
            repaired_indices.append(idx)
        if (invariant_stable_private_amplitude
                and high_contrast_legacy_fallback
                and set(repaired_indices) == set(quarantined)):
            trusted_indices = [
                idx for idx in range(client_count)
                if idx not in quarantined]
            aggregation_models, amplitude_diag = (
                restore_invariant_private_amplitude(
                    reference_model, models, aggregation_models,
                    repaired_indices, trusted_indices, device=device))
            repair_records[
                "invariant_stable_private_amplitude"] = amplitude_diag
        # Clean-start reconstruction can retain useful Non-IID information,
        # but a direction shared only by the LCCT-attributed coalition is not
        # independently supported by the trusted clients.  Remove that common
        # residual after reconstruction while preserving both the trusted task
        # subspace and each repaired client's private residual.  Unlike full
        # task-span projection, this does not force rare client utility to look
        # like the majority clients.  The rule is dataset-, attack-, target-
        # and trigger-agnostic.
        if (invariant_coalition_residual_block
                and high_contrast_legacy_fallback
                and set(repaired_indices) == set(quarantined)):
            aggregation_models, coalition_diag = (
                causal_parameter_path_repair(
                    reference_model, aggregation_models, repaired_indices,
                    variance_keep=invariant_coalition_variance_keep,
                    norm_clip=invariant_coalition_norm_clip,
                    removal_strength=invariant_coalition_block_strength,
                    min_pair_cosine=invariant_coalition_min_pair_cosine))
            coalition_diag["input"] = (
                "clean-start cross-intervention-stable reconstructed updates")
            coalition_diag["release_role"] = (
                "remove coalition-only residual before bounded release")
            coalition_diag["dataset_identity_used"] = False
            coalition_diag["attack_identity_used"] = False
            coalition_diag["target_class_used"] = False
            coalition_diag["trigger_template_used"] = False
            repair_records["invariant_coalition_residual_block"] = (
                coalition_diag)
        # Invariant-gradient repair removes locally unstable coordinates, but
        # the remaining update can still contain a stable client-specific
        # direction that is unrelated to the shared task.  Before granting a
        # larger Non-IID utility release, retain only the component supported
        # by the other clients and the server's fixed clean class gradients.
        # This projection is attack-, trigger-, and target-class agnostic.
        if (invariant_projection_release
                and not high_contrast_legacy_fallback
                and set(repaired_indices) == set(quarantined)):
            invariant_projection_source_models = list(aggregation_models)
            projected_models, projection_diag = causal_trigger_path_block(
                reference_model, aggregation_models, repaired_indices,
                utility_probe=interaction_probe,
                variance_keep=invariant_projection_variance_keep,
                norm_clip=invariant_projection_norm_clip,
                utility_shards_per_class=(
                    invariant_projection_utility_shards),
                device=device)
            projection_diag["input"] = (
                "cross-intervention-stable repaired client updates")
            projection_diag["release_role"] = (
                "sanitize before bounded Non-IID utility release")
            repair_records["invariant_task_projection"] = projection_diag
            if projection_diag.get("repair_ready"):
                invariant_projection_ready = True
                if invariant_projection_preserve_head:
                    trusted_indices = [
                        idx for idx in range(client_count)
                        if idx not in quarantined]
                    projected_models, head_diag = (
                        _restore_bounded_classifier_head(
                            reference_model, aggregation_models,
                            projected_models, repaired_indices,
                            trusted_indices,
                            norm_clip=invariant_projection_norm_clip))
                    repair_records[
                        "invariant_bounded_head_restore"] = head_diag
                    if head_diag.get("applied"):
                        invariant_projection_head_prefix = head_diag.get(
                            "head_prefix")
                for idx in repaired_indices:
                    aggregation_models[idx] = projected_models[idx]
        if (semantic_path_release
                and invariant_nonrecursive_release
                and final_round
                and state.get("ci_adaptation", {}).get(
                    "high_heterogeneity", False)
                and set(repaired_indices) == set(quarantined)):
            aggregation_models, semantic_adapter_diag = (
                add_nonrecursive_semantic_adapter(
                    reference_model, aggregation_models, models,
                    repaired_indices,
                    utility_probe=(
                        semantic_path_probe
                        if semantic_path_probe is not None
                        else interaction_probe),
                    release_dose=semantic_path_release_weight,
                    shards_per_class=semantic_path_shards_per_class,
                    norm_clip=semantic_path_norm_clip,
                    robust_coefficient_clip=(
                        semantic_path_robust_coefficient_clip),
                    minimum_recall_gap=semantic_path_min_recall_gap,
                    device=device))
            semantic_adapter_diag["aggregation_stage"] = (
                "post-invariant-repair_nonrecursive_adapter")
            repair_records["nonrecursive_semantic_adapter"] = (
                semantic_adapter_diag)
            if semantic_adapter_diag.get("repair_ready"):
                semantic_path_ready = True
    # ``parameter_soft`` is deliberately not a quarantine rule.  In the
    # Non-IID setting a client with a large interventional response can still
    # carry a unique, useful task component.  We therefore use the causal
    # response only to nominate an anomalous *path*, remove the residual
    # direction shared by those nominees, and retain their remaining update
    # under the normal continuous risk weights below.
    soft_parameter_candidates = set()
    if (repair and repair_mode == "parameter_soft"
            and str(score_weighting).lower() == "robust"):
        center = scores.median()
        mad = (scores - center).abs().median()
        floor = 1e-3 * scores.abs().median().clamp(min=1e-6)
        scale = torch.maximum(1.4826 * mad, floor).clamp(min=1e-8)
        soft_parameter_z = torch.relu((scores - center) / scale)
        soft_parameter_candidates = set(
            int(idx) for idx in torch.nonzero(
                soft_parameter_z > float(z_margin), as_tuple=False
            ).reshape(-1).tolist())
        # A shared residual is only identifiable for two or more candidates.
        if len(soft_parameter_candidates) >= 2:
            aggregation_models, parameter_diag = causal_parameter_path_repair(
                reference_model, models, soft_parameter_candidates)
            parameter_diag["candidate_rule"] = (
                "MAD-standardized do(G,X)xdo(theta) response > z_margin")
            parameter_diag["candidates"] = sorted(soft_parameter_candidates)
            repair_records["parameter_soft_path"] = parameter_diag
            if parameter_diag.get("repair_ready"):
                repaired_indices = sorted(soft_parameter_candidates)
        else:
            repair_records["parameter_soft_path"] = {
                "repair_ready": False,
                "candidates": sorted(soft_parameter_candidates),
                "reason": "fewer than two causally anomalous updates",
            }

    # Causal pardon keeps LCCT as the detector but replaces immediate sticky
    # exclusion with three states: trusted, repairable boundary, and stable
    # high-confidence quarantine.  This matters under Non-IID because a large
    # causal response may encode a rare benign class rather than a backdoor.
    # Boundary updates are projected onto the task subspace learned only from
    # the other clients; their raw residual is never released.
    pardon_candidates = set()
    pardon_alignment = {}
    pardon_weight_overrides = {}
    pardoned_clients = []
    if causal_pardon_mode:
        center = scores.median()
        mad = (scores - center).abs().median()
        floor = 1e-3 * scores.abs().median().clamp(min=1e-6)
        scale = torch.maximum(1.4826 * mad, floor).clamp(min=1e-8)
        pardon_z = torch.relu((scores - center) / scale)
        ranked = torch.argsort(scores, descending=True)
        if max_quarantine > 0:
            suspicious_floor = float(scores[ranked[max_quarantine - 1]])
            benign_ceiling = float(scores[ranked[max_quarantine]])
            separation_ratio = suspicious_floor / max(benign_ceiling, 1e-8)

        risk_streak = state.setdefault(
            "causal_pardon_risk_streak", [0] * client_count)
        safe_streak = state.setdefault(
            "causal_pardon_safe_streak", [0] * client_count)
        if len(risk_streak) != client_count:
            risk_streak[:] = [0] * client_count
        if len(safe_streak) != client_count:
            safe_streak[:] = [0] * client_count
        for idx in range(client_count):
            risky = bool(float(pardon_z[idx]) > float(z_margin))
            risk_streak[idx] = int(risk_streak[idx]) + 1 if risky else 0
            safe_streak[idx] = 0 if risky else int(safe_streak[idx]) + 1
            if risky:
                pardon_candidates.add(idx)

        # Hard attribution needs both cross-intervention reliability and a
        # stable temporal streak.  A one-round Non-IID outlier is not enough.
        reliable_separation = bool(
            reliability is not None
            and reliability >= float(min_reliability)
            and separation_ratio >= float(gap_ratio))
        stable_risky = [
            int(idx) for idx in ranked[:max_quarantine].tolist()
            if (idx in pardon_candidates
                and int(risk_streak[idx]) >= max(
                    int(pardon_hard_streak), 1))]
        previous_quarantine = set(quarantined)
        if reliable_separation:
            quarantined.update(stable_risky)
        newly_quarantined = sorted(quarantined - previous_quarantine)

        repair_candidates = set(pardon_candidates) | set(quarantined)
        if repair_candidates:
            if causal_trigger_block_mode:
                aggregation_models, parameter_diag = (
                    causal_trigger_path_block(
                        reference_model, models, repair_candidates,
                        utility_probe=interaction_probe,
                        variance_keep=pardon_variance_keep,
                        device=device))
            else:
                aggregation_models, parameter_diag = (
                    causal_parameter_path_repair(
                        reference_model, models, repair_candidates,
                        variance_keep=pardon_variance_keep))
            parameter_diag["candidate_rule"] = (
                "MAD-standardized do(G,X)xdo(theta) response with "
                "two-round reliable hard attribution")
            parameter_diag["candidates"] = sorted(repair_candidates)
            record_key = ("causal_trigger_block_path"
                          if causal_trigger_block_mode
                          else "causal_pardon_path")
            repair_records[record_key] = parameter_diag
            if parameter_diag.get("repair_ready"):
                repaired_indices = sorted(repair_candidates)
        else:
            repair_records["causal_pardon_path"] = {
                "repair_ready": False,
                "candidates": [],
                "reason": "no causally anomalous update",
            }

        # SPMC-style coalition agreement is used only after causal path
        # removal.  It controls how much repaired utility can return; it does
        # not replace LCCT's causal detector or inspect an attack target.
        trusted_indices = [
            idx for idx in range(client_count)
            if idx not in repair_candidates]
        if not trusted_indices:
            trusted_indices = [
                idx for idx in range(client_count) if idx not in quarantined]
        repaired_updates = []
        for model in aggregation_models:
            update, _ = _model_update_vector(reference_model, model)
            repaired_updates.append(update)
        if trusted_indices:
            trusted_direction = torch.stack(
                [repaired_updates[idx] for idx in trusted_indices]).mean(dim=0)
            trusted_norm = trusted_direction.norm().clamp(min=1e-12)
            for idx in sorted(repair_candidates):
                update = repaired_updates[idx]
                cosine = float(torch.dot(update, trusted_direction) / (
                    update.norm().clamp(min=1e-12) * trusted_norm))
                pardon_alignment[idx] = cosine
                positive_alignment = min(max(cosine, 0.0), 1.0)
                # At most half of an ordinary client's raw mass is restored.
                # A small floor preserves rare benign class information after
                # the suspicious residual itself has been removed.
                bounded_floor = min(max(float(pardon_floor), 0.0), 1.0)
                bounded_ceiling = min(max(
                    float(pardon_ceiling), bounded_floor), 1.0)
                aligned_ceiling = (
                    bounded_floor
                    + (bounded_ceiling - bounded_floor) * positive_alignment)
                if causal_trigger_block_mode:
                    # A persistent high-confidence coalition may contribute
                    # only a small amount of its sanitized task projection.
                    # Borderline clients retain more mass, so Non-IID utility
                    # is not tied to a binary client-level exclusion.
                    cap = max(float(score_cap), float(z_margin) + 1e-6)
                    risk_strength = min(max(
                        (float(pardon_z[idx]) - float(z_margin))
                        / max(cap - float(z_margin), 1e-6), 0.0), 1.0)
                    persistence = min(
                        float(risk_streak[idx])
                        / max(float(pardon_hard_streak), 1.0), 1.0)
                    block_strength = risk_strength * persistence
                    pardon_weight_overrides[idx] = (
                        aligned_ceiling * (1.0 - block_strength)
                        + bounded_floor * block_strength)
                else:
                    pardon_weight_overrides[idx] = aligned_ceiling

        # A previously quarantined client can return only after two calm
        # rounds and non-conflicting repaired contribution.  Pardon is thus
        # reversible, unlike the old sticky first-round decision.
        for idx in sorted(quarantined):
            if (int(safe_streak[idx]) >= max(
                    int(pardon_release_streak), 1)
                    and float(pardon_alignment.get(idx, -1.0)) >= 0.0):
                pardoned_clients.append(idx)
        quarantined.difference_update(pardoned_clients)
        state["quarantined"] = sorted(quarantined)

    if (quarantined and repair and repair_mode == "invariant"
            and warmup_complete):
        for idx in sorted(quarantined):
            repaired_model, repair_diag = (
                train_client_with_invariant_causal_gradients(
                    reference_model, clients[idx], epochs=repair_epochs,
                    lr=repair_lr, worlds=trials, budget=budget,
                    sign_agreement=invariant_sign,
                    seed=int(seed) + idx * 1009, device=device))
            aggregation_models[idx] = repaired_model
            repair_records[str(idx)] = repair_diag
            repaired_indices.append(idx)
        keep_indices = torch.arange(client_count, dtype=torch.long)
    elif quarantined and repair and repair_mode == "parameter":
        aggregation_models, parameter_diag = causal_parameter_path_repair(
            reference_model, models, quarantined)
        repair_records["parameter_path"] = parameter_diag
        if parameter_diag.get("repair_ready"):
            repaired_indices = sorted(quarantined)
        keep_indices = torch.as_tensor(
            [idx for idx in range(client_count)
             if idx not in quarantined or idx in repaired_indices],
            dtype=torch.long)
    elif quarantined and repair and repair_mode == "interaction":
        aggregation_models, repaired_indices, interaction_diag = (
            causal_interaction_subspace_repair(
                reference_model, models, clients, quarantined,
                utility_probe=interaction_probe,
                trials=max(2, min(int(trials), 3)), budget=budget,
                seed=seed, device=device))
        repair_records["causal_interaction_subspace"] = interaction_diag
        keep_indices = torch.as_tensor(
            [idx for idx in range(client_count)
             if idx not in quarantined or idx in repaired_indices],
            dtype=torch.long)
    elif quarantined and repair and repair_mode in (
            "cms", "cms_layer", "cms_release", "cms_layer_release"):
        aggregation_models, repaired_indices, mechanism_diag = (
            causal_mechanism_subspace_repair(
                reference_model, models, clients, quarantined, profiles,
                trials=trials, budget=budget, seed=seed, device=device,
                dose_caps=state.get("cms_dose_caps"),
                layerwise=repair_mode in ("cms_layer", "cms_layer_release")))
        if repair_mode in ("cms_layer", "cms_layer_release"):
            state["cms_dose_caps"] = {
                int(idx): (mechanism_diag["clients"][str(idx)].get(
                    "selected_group_doses") or {
                        group: 0.0 for group in mechanism_diag[
                            "clients"][str(idx)]["parameter_groups"]})
                for idx in sorted(quarantined)}
        else:
            state["cms_dose_caps"] = {
                int(idx): float(mechanism_diag["clients"][str(idx)][
                    "selected_dose"])
                for idx in sorted(quarantined)}
        mechanism_diag["monotone_dose_caps"] = True
        repair_records["causal_mechanism_subspace"] = mechanism_diag
        keep_indices = torch.as_tensor(
            [idx for idx in range(client_count)
             if idx not in quarantined or idx in repaired_indices],
            dtype=torch.long)
    elif quarantined and repair and repair_mode == "label":
        for idx in sorted(quarantined):
            repaired_model, repair_diag = (
                repair_client_by_causal_transport(
                    reference_model, models[idx], clients[idx],
                    epochs=repair_epochs, lr=repair_lr, trials=trials,
                    budget=budget, node_z=repair_node_z,
                    seed=int(seed) + idx * 1009, device=device))
            repair_records[str(idx)] = repair_diag
            if repair_diag["repair_ready"]:
                aggregation_models[idx] = repaired_model
                repaired_indices.append(idx)
        # Restore a quarantined client's non-IID contribution only after its
        # label mechanism has enough interventional evidence for repair.
        keep_indices = torch.as_tensor(
            [idx for idx in range(client_count)
             if idx not in quarantined or idx in repaired_indices],
            dtype=torch.long)
    elif (quarantined and repair
          and repair_mode in ("shadow_only", "shadow_ci",
                              "shadow_ci_strong",
                              "shadow_ci_transfer",
                              "shadow_ci_ego",
                              "shadow_ci_ego_transfer",
                              "stable_gate", "query_cate")):
        keep_indices = torch.as_tensor(
            (list(range(client_count))
             if (invariant_release_enabled or semantic_path_ready) else
             [idx for idx in range(client_count)
              if idx not in quarantined]), dtype=torch.long)
        record_name = {
            "shadow_only": "causal_shadow_quarantine",
            "stable_gate": "stable_causal_message_gate",
            "query_cate": "query_cate_shadow",
        }.get(repair_mode, "counterfactual_shadow")
        repair_records[record_name] = {
            "provisional_clients": sorted(provisional),
            "provisional_round": state.get("provisional_round"),
            "broadcast_excludes_provisional_from_round_one": True,
            "post_aggregation_counterfactual_training": bool(
                repair_mode != "shadow_only"),
        }
    elif causal_pardon_mode:
        # Repaired suspicious clients are allowed to contribute only through
        # the sanitized task-subspace projection.  If repair is unavailable,
        # stable quarantined clients remain fully excluded.
        keep_indices = torch.as_tensor([
            idx for idx in range(client_count)
            if idx not in quarantined or idx in repaired_indices
        ], dtype=torch.long)
    elif quarantined:
        keep_indices = torch.as_tensor(
            [idx for idx in range(client_count) if idx not in quarantined],
            dtype=torch.long)
    else:
        keep_indices = torch.topk(
            scores, k=min_keep, largest=False).indices
    keep_mask = torch.zeros(client_count, dtype=torch.float32)
    keep_mask[keep_indices] = 1.0
    causal_z = None
    if causal_pardon_mode:
        # Pardon always uses relative robust scores; raw causal-response
        # magnitudes are not comparable across datasets or IID regimes.
        causal_z = pardon_z
        excess = torch.relu(causal_z - float(z_margin))
        if float(score_cap) > 0:
            excess = excess.clamp(max=float(score_cap))
        soft_weights = torch.exp(-float(penalty) * excess)
    elif ((quarantined and repair and repaired_indices)
            and repair_mode not in ("parameter_soft", "invariant")
            and not invariant_release_enabled):
        # After explicit causal repair, score-based attenuation would again
        # erase non-IID client distributions.  Use ordinary FL weights.
        soft_weights = torch.ones_like(scores)
    elif str(score_weighting).lower() == "robust":
        # Response magnitudes are attack- and dataset-scale dependent.  A
        # round-wise median/MAD standardization estimates relative excess
        # causal transport under a minority-corruption assumption, so a
        # single causal risk budget is meaningful for IID and Non-IID data.
        center = scores.median()
        mad = (scores - center).abs().median()
        floor = 1e-3 * scores.abs().median().clamp(min=1e-6)
        scale = torch.maximum(1.4826 * mad, floor).clamp(min=1e-8)
        causal_z = torch.relu((scores - center) / scale)
        excess = torch.relu(causal_z - float(z_margin))
        if float(score_cap) > 0:
            excess = excess.clamp(max=float(score_cap))
        soft_weights = torch.exp(-float(penalty) * excess)
    else:
        soft_weights = torch.exp(
            -float(penalty) * torch.relu(scores - float(z_margin)))
    if semantic_path_ready:
        semantic_weight = min(max(
            float(semantic_path_release_weight), 0.0), 1.0)
        for idx in repaired_indices:
            soft_weights[idx] = semantic_weight
            pardon_weight_overrides[idx] = semantic_weight
    invariant_release_quality = {}
    effective_invariant_release_cap = max(
        float(invariant_release_cap), 0.0)
    high_contrast_reconstruction_active = bool(
        high_contrast_legacy_fallback
        and invariant_high_contrast_reconstruction)
    if high_contrast_reconstruction_active:
        # The raw high-contrast upload remains blocked.  This cap applies
        # only to the clean-start reconstructed update and is deliberately
        # separate from the monitor-selected low-contrast release path.
        effective_invariant_release_cap = min(
            effective_invariant_release_cap,
            max(float(invariant_reconstruction_release_cap), 0.0))
    if (invariant_release_enabled and invariant_release_contrast_guard
            and invariant_release_contrast is not None):
            threshold = contrast_threshold
            if invariant_release_contrast > threshold:
                # The ratio is dimensionless, so the same causal-confidence
                # gate applies across attacks and datasets. High-confidence
                # legacy fallback retains the previously validated CRFU path
                # from the first affected round; otherwise use smooth inverse
                # attenuation for the repaired release.
                if not invariant_release_contrast_hard_fallback:
                    effective_invariant_release_cap *= (
                        threshold / invariant_release_contrast)
            if not high_contrast_legacy_fallback:
                effective_invariant_release_cap = max(
                    min(float(invariant_release_contrast_min_cap),
                        max(float(invariant_release_cap), 0.0)),
                    effective_invariant_release_cap)
    if invariant_release_enabled:
        for idx in repaired_indices:
            repair_diag = repair_records.get(str(idx), {})
            node_count = (int(repair_diag.get("trusted_train_nodes", 0))
                          + int(repair_diag.get(
                              "counterfactual_train_nodes", 0)))
            trusted_fraction = (
                float(repair_diag.get("trusted_train_nodes", 0))
                / max(node_count, 1))
            stability = float(repair_diag.get(
                "mean_gradient_coordinate_retention", 0.0))
            # Stable-gradient repair and task-span projection are independent
            # sanitizers.  Once both succeed, multiplying by the coordinate
            # retention again would suppress Non-IID utility twice.  The
            # release cap (and optional clean monitor) therefore controls the
            # projected client's mass directly.  Without projection, retain
            # the conservative evidence-weighted release used previously.
            invariant_release_quality[idx] = (
                1.0 if invariant_projection_ready
                else (stability if invariant_release_stability_only
                      else trusted_fraction * stability))
            release_mass = (effective_invariant_release_cap
                            * invariant_release_quality[idx])
            # Quarantine is persistent, so a temporarily low current-round
            # score must not restore a repaired client at ordinary mass.  The
            # causal evidence defines its entire release budget, not a floor.
            soft_weights[idx] = release_mass
            pardon_weight_overrides[idx] = float(release_mass)
    if causal_pardon_mode:
        for idx, value in pardon_weight_overrides.items():
            if idx in repaired_indices:
                soft_weights[idx] = float(value)
    if base_weights is None:
        aggregation_base_weights = torch.ones_like(scores)
    else:
        aggregation_base_weights = torch.as_tensor(
            base_weights, dtype=torch.float32).reshape(-1)
        if aggregation_base_weights.numel() != client_count:
            raise ValueError(
                "local causal base weights must match the client count")
        aggregation_base_weights = aggregation_base_weights.clamp(min=0.0)
        if float(aggregation_base_weights.sum()) <= 0.0:
            raise ValueError("local causal base weights must have positive mass")
    transfer_diag = None
    if repair_mode in ("shadow_ci_transfer",
                       "shadow_ci_ego_transfer") and quarantined:
        repair_weight_multipliers, transfer_diag = (
            causal_transfer_weight_multipliers(
                reference_model, models, clients,
                suspicious_indices=quarantined,
                retained_indices=keep_indices.tolist(), trials=trials,
                budget=budget, seed=seed, device=device,
                strength=transfer_strength, floor=transfer_floor,
                score_cap=score_cap,
                min_consensus=transfer_min_consensus))
        repair_records["causal_transfer_weighting"] = transfer_diag

    base_mass = soft_weights * keep_mask * aggregation_base_weights
    base_distribution = base_mass / base_mass.sum().clamp(min=1e-8)
    transfer_mass = base_mass * repair_weight_multipliers
    transfer_distribution = transfer_mass / transfer_mass.sum().clamp(min=1e-8)
    if repair_mode in ("shadow_ci_transfer",
                       "shadow_ci_ego_transfer") and quarantined:
        mix = min(max(float(transfer_mix), 0.0), 1.0)
        weights = ((1.0 - mix) * base_distribution
                   + mix * transfer_distribution)
        if transfer_diag is not None:
            transfer_diag["base_distribution"] = [
                float(v) for v in base_distribution]
            transfer_diag["final_distribution"] = [float(v) for v in weights]
            transfer_diag["mix"] = mix
            transfer_diag["effective_client_count"] = float(
                1.0 / weights.square().sum().clamp(min=1e-8))
    else:
        weights = transfer_distribution
    weights = weights / weights.sum().clamp(min=1e-8)

    # High-contrast reconstruction is optional, never the default path.  A
    # release must pass three target-free tests: (1) the reconstructed path
    # improves held-out balanced utility, (2) clean accuracy and every class
    # recall are preserved up to finite-sample uncertainty, and (3) graph-
    # intervention sensitivity does not increase.  Otherwise the original
    # zero-release firewall is retained.
    coverage_monitor_reliable = True
    coverage_resolution_guard = None
    if (high_contrast_reconstruction_active
            and invariant_reconstruction_coverage_gate
            and repaired_indices and release_monitor_probe is not None):
        _, _, coverage_probe_y, coverage_probe_idx = release_monitor_probe
        coverage_probe_idx = np.asarray(
            coverage_probe_idx, dtype=np.int64).reshape(-1)
        coverage_index_tensor = torch.as_tensor(
            coverage_probe_idx, dtype=torch.long,
            device=coverage_probe_y.device)
        coverage_labels = coverage_probe_y[coverage_index_tensor]
        coverage_support = {
            int(class_id): int((coverage_labels == int(class_id)).sum())
            for class_id in torch.unique(coverage_labels).tolist()
        }
        # A per-round classwise dose selector is too noisy when moving one
        # monitor node changes a class recall by more than ten percentage
        # points.  In that case retain the stable reconstruction dose already
        # established by the causal release path.  This is a finite-sample
        # reliability check, not a dataset- or attack-specific branch.
        coverage_max_one_node_resolution = 0.10
        coverage_resolution_support_floor = int(np.ceil(
            1.0 / coverage_max_one_node_resolution))
        coverage_monitor_reliable = bool(
            coverage_support
            and min(coverage_support.values())
            >= coverage_resolution_support_floor)
        coverage_resolution_guard = {
            "applied": bool(coverage_monitor_reliable),
            "reason": ("sufficient_classwise_monitor_resolution"
                       if coverage_monitor_reliable
                       else "fallback_to_stable_causal_release"),
            "class_support": coverage_support,
            "maximum_one_node_recall_resolution": float(
                coverage_max_one_node_resolution),
            "minimum_reliable_class_support": int(
                coverage_resolution_support_floor),
            "dataset_identity_used": False,
            "attack_identity_used": False,
            "target_class_used": False,
        }
        if not coverage_monitor_reliable:
            repair_records["high_contrast_reconstruction_coverage_gate"] = {
                **coverage_resolution_guard,
                "configured": True,
                "reconstruction_released": True,
                "selected_release_cap": float(
                    effective_invariant_release_cap),
                "selection_rule": (
                    "use classwise adaptive dosing only when every represented "
                    "monitor class has at most ten-percent one-node recall "
                    "resolution; otherwise retain the stable causal dose"),
            }

    if (high_contrast_reconstruction_active
            and invariant_reconstruction_coverage_gate
            and coverage_monitor_reliable
            and repaired_indices and release_monitor_probe is not None):
        probe_x, probe_adj, probe_y, probe_idx = release_monitor_probe
        probe_x = probe_x.to(device)
        probe_adj = probe_adj.to(device)
        probe_y = probe_y.to(device)
        probe_idx = np.asarray(probe_idx, dtype=np.int64).reshape(-1)
        zero_release_soft = soft_weights.clone()
        for idx in repaired_indices:
            zero_release_soft[idx] = 0.0
        zero_release_mass = (zero_release_soft * keep_mask
                             * aggregation_base_weights
                             * repair_weight_multipliers)
        zero_release_weights = zero_release_mass / zero_release_mass.sum().clamp(
            min=1e-8)
        zero_release_model = weighted_average_models(
            aggregation_models, zero_release_weights.tolist())
        zero_release_metrics = _head_repair_metrics(
            zero_release_model.to(device), probe_x, probe_adj,
            probe_y, probe_idx)
        monitor_labels = probe_y[torch.as_tensor(
            probe_idx, dtype=torch.long, device=probe_y.device)]
        class_support = {
            int(class_id): int((monitor_labels == int(class_id)).sum())
            for class_id in torch.unique(monitor_labels).tolist()
        }
        min_support = max(
            int(invariant_reconstruction_coverage_min_support), 1)
        recall_floor = min(max(
            float(invariant_reconstruction_coverage_floor), 0.0), 1.0)
        deficient_classes = [
            int(class_id) for class_id, recall in
            zero_release_metrics["class_recall"].items()
            if class_support.get(int(class_id), 0) >= min_support
            and float(recall) < recall_floor]
        # Label presence is not evidence that a retained update still carries
        # useful information for that class.  Compare the two counterfactual
        # aggregation paths directly instead.  Finite monitor sets make a
        # one-node recall change unavoidable, so derive the tolerance from
        # class support instead of a dataset-specific threshold.
        class_recall_tolerance = {
            str(class_id): max(
                float(invariant_release_max_class_drop),
                1.0 / max(int(support), 1))
            for class_id, support in class_support.items()
        }
        cap_max = max(float(effective_invariant_release_cap), 0.0)
        candidate_caps = sorted(set(
            min(value, cap_max) for value in
            [0.0, 0.03, 0.06, 0.10, 0.20, 0.35, 0.50, cap_max]))
        candidate_records = []
        candidate_payloads = []
        accuracy_floor = (zero_release_metrics["accuracy"]
                          - 1.0 / max(len(probe_idx), 1) - 1e-8)
        for cap in candidate_caps:
            candidate_soft = soft_weights.clone()
            for idx in repaired_indices:
                candidate_soft[idx] = (
                    float(cap) * invariant_release_quality.get(idx, 0.0))
            candidate_mass = (candidate_soft * keep_mask
                              * aggregation_base_weights
                              * repair_weight_multipliers)
            candidate_weights = candidate_mass / candidate_mass.sum().clamp(
                min=1e-8)
            candidate_model = weighted_average_models(
                aggregation_models, candidate_weights.tolist())
            candidate_metrics = _head_repair_metrics(
                candidate_model.to(device), probe_x, probe_adj,
                probe_y, probe_idx)
            recall_preserved = all(
                candidate_metrics["class_recall"].get(class_id, 0.0)
                >= recall - class_recall_tolerance.get(
                    str(class_id),
                    float(invariant_release_max_class_drop)) - 1e-8
                for class_id, recall in
                zero_release_metrics["class_recall"].items())
            accuracy_preserved = bool(
                candidate_metrics["accuracy"] >= accuracy_floor)
            deficient_recall_gain = float(np.mean([
                candidate_metrics["class_recall"].get(
                    str(class_id), 0.0)
                - zero_release_metrics["class_recall"].get(
                    str(class_id), 0.0)
                for class_id in deficient_classes
            ])) if deficient_classes else 0.0
            ce_gain = float(
                zero_release_metrics["balanced_ce"]
                - candidate_metrics["balanced_ce"])
            utility_benefit = bool(ce_gain > 1e-6)
            deficient_recall_preserved = bool(
                deficient_recall_gain >= -1e-8)
            candidate_record = {
                "cap": float(cap),
                "monitor": candidate_metrics,
                "balanced_ce_gain": ce_gain,
                "deficient_recall_gain": deficient_recall_gain,
                "utility_benefit": utility_benefit,
                "recall_preserved": bool(recall_preserved),
                "deficient_recall_preserved": (
                    deficient_recall_preserved),
                "accuracy_preserved": accuracy_preserved,
                "intervention_sensitivity": None,
                "sensitivity_preserved": None,
                "eligible_before_sensitivity": bool(
                    cap > 0.0 and utility_benefit
                    and recall_preserved and deficient_recall_preserved
                    and accuracy_preserved),
            }
            candidate_records.append(candidate_record)
            candidate_payloads.append((
                float(cap), candidate_weights, candidate_model,
                candidate_record))

        # Balanced CE uses every represented class and is a smoother measure
        # of useful update contribution than client label counts.  Test the
        # most useful clean-safe dose first, but retain the zero-dose firewall
        # when every positive dose violates interventional safety.
        zero_sensitivity = _head_repair_sensitivity(
            zero_release_model, probe_x, probe_adj, probe_idx,
            views=3, budget=budget,
            seed=int(seed) + 23000011)
        selected_cap = 0.0
        selected_weights = zero_release_weights
        selected_metrics = zero_release_metrics
        selected_record = next(
            record for record in candidate_records
            if record["cap"] == 0.0)
        eligible_payloads = sorted(
            [item for item in candidate_payloads
             if item[3]["eligible_before_sensitivity"]],
            key=lambda item: (
                item[3]["balanced_ce_gain"],
                item[3]["monitor"]["accuracy"],
                item[3]["monitor"]["macro_recall"],
                -item[0]),
            reverse=True)
        for cap, candidate_weights, candidate_model, candidate_record in (
                eligible_payloads):
            candidate_sensitivity = _head_repair_sensitivity(
                candidate_model, probe_x, probe_adj, probe_idx,
                views=3, budget=budget,
                seed=int(seed) + 23000011)
            sensitivity_preserved = bool(
                candidate_sensitivity
                <= zero_sensitivity * 1.02 + 1e-6)
            candidate_record["intervention_sensitivity"] = float(
                candidate_sensitivity)
            candidate_record["sensitivity_preserved"] = bool(
                sensitivity_preserved)
            if sensitivity_preserved:
                selected_cap = float(cap)
                selected_weights = candidate_weights
                selected_metrics = candidate_record["monitor"]
                selected_record = candidate_record
                break
        weights = selected_weights
        release_selected = bool(selected_cap > 0.0)
        max_release_record = max(
            candidate_records, key=lambda record: record["cap"])
        repair_records["high_contrast_reconstruction_coverage_gate"] = {
            "applied": True,
            "resolution_guard": coverage_resolution_guard,
            "monitor_nodes": int(len(probe_idx)),
            "zero_release_monitor": zero_release_metrics,
            "class_support": class_support,
            "recall_floor": float(recall_floor),
            "minimum_class_support": int(min_support),
            "deficient_classes": deficient_classes,
            "class_recall_tolerance": class_recall_tolerance,
            "coverage_deficit": bool(deficient_classes),
            "candidate_caps": [float(value) for value in candidate_caps],
            "candidate_evaluations": candidate_records,
            "released_monitor": max_release_record["monitor"],
            "selected_monitor": selected_metrics,
            "deficient_recall_gain": float(
                selected_record["deficient_recall_gain"]),
            "balanced_ce_gain": float(
                selected_record["balanced_ce_gain"]),
            "utility_benefit": bool(
                selected_record["utility_benefit"]),
            "recall_preserved": bool(
                selected_record["recall_preserved"]),
            "deficient_recall_preserved": bool(
                selected_record["deficient_recall_preserved"]),
            "accuracy_preserved": bool(
                selected_record["accuracy_preserved"]),
            "zero_release_intervention_sensitivity": float(
                zero_sensitivity),
            "released_intervention_sensitivity": (
                selected_record["intervention_sensitivity"]),
            "sensitivity_preserved": (
                selected_record["sensitivity_preserved"]),
            "reconstruction_released": bool(release_selected),
            "selected_release_cap": float(selected_cap),
            "target_class_used": False,
            "attack_identity_used": False,
            "dataset_identity_used": False,
            "selection_rule": (
                "Pareto dose selection by held-out counterfactual "
                "contribution, support-derived finite-sample tolerance, "
                "deficient-class preservation, and intervention safety"),
        }

    if (invariant_release_enabled and invariant_release_utility_guard
            and not high_contrast_reconstruction_active
            and repaired_indices and release_monitor_probe is not None):
        probe_x, probe_adj, probe_y, probe_idx = release_monitor_probe
        probe_x = probe_x.to(device)
        probe_adj = probe_adj.to(device)
        probe_y = probe_y.to(device)
        probe_idx = np.asarray(probe_idx, dtype=np.int64).reshape(-1)
        cap_max = effective_invariant_release_cap
        candidate_caps = sorted(set(
            min(value, cap_max) for value in
            [0.0, 0.03, 0.06, 0.10, 0.20, 0.35, 0.50, 0.75, cap_max]))
        residual_options = [(0.0, aggregation_models)]
        if (invariant_projection_residual_guard
                and invariant_projection_ready
                and invariant_projection_source_models is not None):
            residual_max = min(max(
                float(invariant_projection_residual_max), 0.0), 1.0)
            residual_grid = sorted(set(
                min(value, residual_max)
                for value in [0.0, 0.03, 0.06, 0.10, 0.15, 0.20,
                              residual_max]))
            residual_options = [
                (residual_keep, _blend_invariant_self_residual(
                    aggregation_models, invariant_projection_source_models,
                    repaired_indices, residual_keep,
                    protected_prefix=invariant_projection_head_prefix))
                for residual_keep in residual_grid]
        candidates = []
        for residual_keep, candidate_models in residual_options:
            for cap in candidate_caps:
                candidate_soft = soft_weights.clone()
                for idx in repaired_indices:
                    candidate_soft[idx] = (
                        float(cap) * invariant_release_quality.get(idx, 0.0))
                candidate_mass = (candidate_soft * keep_mask
                                  * aggregation_base_weights
                                  * repair_weight_multipliers)
                candidate_weights = candidate_mass / candidate_mass.sum().clamp(
                    min=1e-8)
                candidate_model = weighted_average_models(
                    candidate_models, candidate_weights.tolist())
                metrics = _head_repair_metrics(
                    candidate_model.to(device), probe_x, probe_adj,
                    probe_y, probe_idx)
                candidates.append((residual_keep, cap, candidate_models,
                                   candidate_weights, metrics))
        baseline = next(
            item[4] for item in candidates
            if item[0] == 0.0 and item[1] == 0.0)
        accuracy_floor = baseline["accuracy"] - (
            1.0 / max(len(probe_idx), 1))
        eligible = []
        for candidate in candidates:
            residual_keep, cap, candidate_models, candidate_weights, metrics = (
                candidate)
            recall_ok = all(
                metrics["class_recall"].get(class_id, 0.0)
                >= recall - float(invariant_release_max_class_drop) - 1e-8
                for class_id, recall in baseline["class_recall"].items())
            if metrics["accuracy"] >= accuracy_floor - 1e-8 and recall_ok:
                eligible.append(candidate)
        (selected_residual, selected_cap, aggregation_models,
         weights, selected_metrics) = max(
            eligible,
            key=lambda item: (
                item[4]["accuracy"], item[4]["macro_recall"],
                -item[4]["balanced_ce"], -item[0], -item[1]))
        repair_records["invariant_release_utility_guard"] = {
            "applied": True,
            "monitor_nodes": int(len(probe_idx)),
            "candidate_caps": [float(value) for value in candidate_caps],
            "selected_cap": float(selected_cap),
            "candidate_self_residual_fractions": [
                float(value) for value, _ in residual_options],
            "selected_self_residual_fraction": float(selected_residual),
            "baseline_monitor": baseline,
            "selected_monitor": selected_metrics,
            "max_class_recall_drop": float(
                invariant_release_max_class_drop),
            "target_class_used": False,
            "attack_identity_used": False,
            "trigger_template_used": False,
        }

    # Improve Non-IID utility without reopening any quarantined update path.
    # Only LCCT-retained clients are reweighted from class-balanced clean
    # calibration risk, and an independent monitor split chooses the mixture.
    utility_quarantine_changes = len(state.get(
        "temporal_quarantine_changes", []))
    utility_membership_stable = bool(
        not temporal_quarantine_enabled
        or utility_quarantine_changes
        <= int(trusted_utility_max_quarantine_changes))
    if (trusted_utility_reweight and utility_membership_stable
            and release_calibration_probe is not None
            and release_monitor_probe is not None):
        trusted_indices = [
            idx for idx in range(client_count) if idx not in quarantined]
        if len(trusted_indices) >= 2:
            cal_x, cal_adj, cal_y, cal_idx = release_calibration_probe
            mon_x, mon_adj, mon_y, mon_idx = release_monitor_probe
            cal_x, cal_adj, cal_y = (
                cal_x.to(device), cal_adj.to(device), cal_y.to(device))
            mon_x, mon_adj, mon_y = (
                mon_x.to(device), mon_adj.to(device), mon_y.to(device))
            cal_idx = np.asarray(cal_idx, dtype=np.int64).reshape(-1)
            mon_idx = np.asarray(mon_idx, dtype=np.int64).reshape(-1)
            clean_scores = []
            for idx in trusted_indices:
                metrics = _head_repair_metrics(
                    aggregation_models[idx].to(device), cal_x, cal_adj,
                    cal_y, cal_idx)
                clean_scores.append(-float(metrics["balanced_ce"]))
            clean_scores = torch.as_tensor(
                clean_scores, dtype=weights.dtype, device=weights.device)
            temperature = max(float(trusted_utility_temperature), 1e-6)
            utility_distribution = torch.softmax(
                (clean_scores - clean_scores.max()) / temperature, dim=0)
            trusted_mass = weights[trusted_indices].sum()
            utility_weights = weights.clone()
            utility_weights[trusted_indices] = (
                trusted_mass * utility_distribution)
            max_mix = min(max(float(trusted_utility_max_mix), 0.0), 1.0)
            mixes = sorted(set(
                min(value, max_mix)
                for value in [0.0, 0.10, 0.25, 0.50, 0.75, 1.0]))
            candidates = []
            for mix in mixes:
                candidate_weights = (
                    (1.0 - mix) * weights + mix * utility_weights)
                candidate_weights = candidate_weights / (
                    candidate_weights.sum().clamp_min(1e-8))
                candidate_model = weighted_average_models(
                    aggregation_models, candidate_weights.tolist())
                metrics = _head_repair_metrics(
                    candidate_model.to(device), mon_x, mon_adj,
                    mon_y, mon_idx)
                candidates.append((mix, candidate_weights, metrics))
            baseline = candidates[0][2]
            accuracy_floor = baseline["accuracy"] - (
                1.0 / max(len(mon_idx), 1))
            eligible = []
            for mix, candidate_weights, metrics in candidates:
                recall_ok = all(
                    metrics["class_recall"].get(class_id, 0.0)
                    >= recall - float(trusted_utility_max_class_drop) - 1e-8
                    for class_id, recall in baseline[
                        "class_recall"].items())
                if metrics["accuracy"] >= accuracy_floor - 1e-8 and recall_ok:
                    eligible.append((mix, candidate_weights, metrics))
            selected_mix, weights, selected_metrics = max(
                eligible,
                key=lambda item: (
                    item[2]["accuracy"], item[2]["macro_recall"],
                    -item[2]["balanced_ce"], -item[0]))
            repair_records["trusted_utility_reweight"] = {
                "applied": True,
                "trusted_clients": trusted_indices,
                "calibration_nodes": int(len(cal_idx)),
                "monitor_nodes": int(len(mon_idx)),
                "clean_balanced_ce": [float(-value)
                                      for value in clean_scores],
                "candidate_mixes": mixes,
                "selected_mix": float(selected_mix),
                "baseline_monitor": baseline,
                "selected_monitor": selected_metrics,
                "temperature": float(temperature),
                "max_mix": float(max_mix),
                "quarantine_change_count": int(
                    utility_quarantine_changes),
                "max_quarantine_changes": int(
                    trusted_utility_max_quarantine_changes),
                "max_class_recall_drop": float(
                    trusted_utility_max_class_drop),
                "quarantined_mass_unchanged": True,
                "target_class_used": False,
                "attack_identity_used": False,
                "trigger_template_used": False,
            }
    elif trusted_utility_reweight and not utility_membership_stable:
        repair_records["trusted_utility_reweight"] = {
            "applied": False,
            "reason": "temporal_quarantine_membership_unstable",
            "quarantine_change_count": int(utility_quarantine_changes),
            "max_quarantine_changes": int(
                trusted_utility_max_quarantine_changes),
            "quarantined_mass_unchanged": True,
            "target_class_used": False,
            "attack_identity_used": False,
            "trigger_template_used": False,
        }

    aggregated_model = weighted_average_models(
        aggregation_models, weights.tolist())
    if repair_mode in ("shadow_ci", "shadow_ci_strong",
                       "shadow_ci_transfer",
                       "shadow_ci_ego",
                       "shadow_ci_ego_transfer",
                       "stable_gate") and quarantined:
        cpu_rng_state = torch.get_rng_state()
        cuda_rng_states = (torch.cuda.get_rng_state_all()
                           if torch.cuda.is_available() else None)
        try:
            torch.manual_seed(int(seed) + 440003)
            if repair_mode == "stable_gate":
                aggregated_model, shadow_diag = (
                    train_stable_causal_message_gate(
                        reference_model, aggregated_model, models, clients,
                        suspicious_indices=quarantined,
                        benign_indices=[
                            idx for idx in keep_indices.tolist()
                            if idx not in quarantined], state=state,
                        budget=budget, policy_steps=max(3, trials),
                        model_steps=2, lr_policy=0.005,
                        lr_model=min(float(repair_lr), 0.001),
                        seed=seed, device=device))
                repair_records["stable_causal_message_gate"].update(
                    shadow_diag)
            elif repair_mode in ("shadow_ci", "shadow_ci_strong",
                                  "shadow_ci_transfer", "shadow_ci_ego",
                                  "shadow_ci_ego_transfer"):
                strong_ci = repair_mode in (
                    "shadow_ci_strong", "shadow_ci_transfer",
                    "shadow_ci_ego", "shadow_ci_ego_transfer")
                pre_shadow_model = aggregated_model
                candidate_model, shadow_diag = (
                    contrastive_interventional_shadow_train(
                        reference_model, aggregated_model, models, clients,
                        suspicious_indices=quarantined,
                        benign_indices=keep_indices.tolist(), state=state,
                        budget=(min(0.35, 1.5 * float(budget))
                                if strong_ci else budget),
                        policy_steps=(max(6, trials) if strong_ci
                                      else max(3, trials)),
                        model_steps=(int(ego_model_steps)
                                     if repair_mode in (
                                         "shadow_ci_ego",
                                         "shadow_ci_ego_transfer")
                                     else (2 if strong_ci else 1)),
                        lr_policy=0.005,
                        lr_model=min(float(repair_lr), 0.001),
                        consistency_weight=(
                            float(ego_consistency_weight)
                            if repair_mode in (
                                "shadow_ci_ego", "shadow_ci_ego_transfer")
                            else (2.0 if strong_ci else 1.0)),
                        local_focus=repair_mode in (
                            "shadow_ci_ego", "shadow_ci_ego_transfer"),
                        focus_fraction=float(ego_focus_fraction),
                        max_model_steps=int(ego_max_model_steps),
                        residual_ratio_threshold=float(
                            ego_residual_ratio_threshold),
                        seed=seed, device=device))
                guarded_high_heterogeneity = bool(
                    configured_repair_mode in (
                        "shadow_ci_ego_guarded_adaptive",
                        "invariant_shadow_guarded")
                    and state.get("ci_adaptation", {}).get(
                        "high_heterogeneity", False))
                residual_resolved_at_minimum = bool(
                    shadow_diag.get("training_ready", False)
                    and int(shadow_diag.get("model_steps", 0))
                    <= int(shadow_diag.get("minimum_model_steps", 0))
                    and float(shadow_diag.get(
                        "final_suspicious_to_benign_residual_ratio",
                        float("inf")))
                    <= float(ego_residual_ratio_threshold))
                commit_candidate = not (
                    guarded_high_heterogeneity
                    and residual_resolved_at_minimum)
                aggregated_model = (candidate_model if commit_candidate
                                    else pre_shadow_model)
                shadow_diag["guarded_commit_enabled"] = bool(
                    guarded_high_heterogeneity)
                shadow_diag["candidate_committed"] = bool(commit_candidate)
                shadow_diag["candidate_rejected_reason"] = (
                    None if commit_candidate
                    else "residual_resolved_within_minimum_steps")
                shadow_diag["commit_rule"] = (
                    "under high heterogeneity, commit only when the "
                    "interventional residual persists beyond minimum steps")
                repair_records["counterfactual_shadow"].update(shadow_diag)
        finally:
            torch.set_rng_state(cpu_rng_state)
            if cuda_rng_states is not None:
                torch.cuda.set_rng_state_all(cuda_rng_states)
    if repair_mode in ("cms", "cms_layer") and quarantined:
        # Causal firewall: repaired suspicious updates contribute to the
        # current released model, but never redefine the untreated baseline
        # used to train clients and estimate benign mechanisms next round.
        safe_mask = torch.as_tensor(
            [idx not in quarantined for idx in range(client_count)],
            dtype=torch.float32)
        safe_weights = safe_mask / safe_mask.sum().clamp(min=1.0)
        state["_cms_safe_model"] = weighted_average_models(
            models, safe_weights.tolist())
    invariant_safe_core_high_heterogeneity = bool(
        state.get("ci_adaptation", {}).get("high_heterogeneity", False))
    if not invariant_safe_core_high_heterogeneity:
        state.pop("_invariant_safe_model", None)
    if (invariant_nonrecursive_release and invariant_shadow_guarded
            and invariant_safe_core_high_heterogeneity and quarantined):
        # Keep two update paths after LCCT has causally identified the
        # suspicious clients.  The trusted-only path is the recursive safe
        # core broadcast to the next round; repaired suspicious information
        # remains a one-shot utility adapter and is never allowed to redefine
        # the baseline used by subsequent clients or causal interventions.
        safe_mask = torch.as_tensor(
            [idx not in quarantined for idx in range(client_count)],
            dtype=torch.float32)
        safe_mass = safe_mask * aggregation_base_weights
        safe_mass_total = float(safe_mass.sum())
        if safe_mass_total > 0.0:
            safe_weights = safe_mass / safe_mass_total
            state["_invariant_safe_model"] = weighted_average_models(
                models, safe_weights.tolist())
            repair_records["invariant_nonrecursive_safe_core"] = {
                "enabled": True,
                "applied": True,
                "trusted_clients": [
                    int(idx) for idx in range(client_count)
                    if idx not in quarantined],
                "quarantined_clients": sorted(
                    int(idx) for idx in quarantined),
                "safe_core_weights": [
                    float(value) for value in safe_weights],
                "recursive_path": "trusted_clients_only",
                "released_adapter_recursive": False,
                "malicious_identity_used": False,
                "target_class_used": False,
                "trigger_template_used": False,
            }

    malicious_flags = np.asarray(
        [bool(client.malicious) for client in clients])
    positives = scores_np[malicious_flags]
    negatives = scores_np[~malicious_flags]
    diagnostic_auc = None
    if len(positives) and len(negatives):
        wins = sum(float(a > b) + 0.5 * float(a == b)
                   for a in positives for b in negatives)
        diagnostic_auc = float(wins / (len(positives) * len(negatives)))
    record = {
        "scores": [float(v) for v in scores],
        "weights": [float(v) for v in weights],
        "aggregation_base_weights": [
            float(v) for v in aggregation_base_weights],
        "repair_weight_multipliers": [
            float(v) for v in repair_weight_multipliers],
        "score_weighting": str(score_weighting),
        "causal_score_z": ([float(v) for v in causal_z]
                           if causal_z is not None else None),
        "causal_score_cap": float(score_cap),
        "kept_clients": [int(v) for v in keep_indices.tolist()],
        "quarantined_clients": sorted(quarantined),
        "provisional_clients": sorted(provisional),
        "provisional_round": state.get("provisional_round"),
        "newly_quarantined": newly_quarantined,
        "quarantine_round": state.get("quarantine_round"),
        "separation_ratio": separation_ratio,
        "persistence_enabled": bool(persistent),
        "min_reliability": float(min_reliability),
        "gap_ratio": float(gap_ratio),
        "ranked_clients": [int(v) for v in torch.argsort(
            scores, descending=True).tolist()],
        "split_half_rank_reliability": reliability,
        "diagnostic_auc": diagnostic_auc,
        "malicious_identity_used_for_scoring": False,
        "target_class_used": False,
        "clean_graph_used": False,
        "full_graph_uploaded": False,
        "outcome": outcome,
        "energy_used": bool(outcome == "hybrid"),
        "causal_transport_repair": bool(repair),
        "causal_transport_repair_mode": repair_mode,
        "configured_repair_mode": configured_repair_mode,
        "invariant_release_enabled": invariant_release_enabled,
            "invariant_projection_release": {
            "enabled": bool(invariant_projection_release),
            "variance_keep": float(invariant_projection_variance_keep),
            "norm_clip": float(invariant_projection_norm_clip),
            "preserve_bounded_head": bool(
                invariant_projection_preserve_head),
            "utility_shards_per_class": int(
                invariant_projection_utility_shards),
            "self_residual_monitor_guard": bool(
                invariant_projection_residual_guard),
            "self_residual_max_fraction": float(
                invariant_projection_residual_max),
            "local_label_rescue": bool(invariant_local_label_rescue),
                "rescue_transport_z": float(invariant_rescue_transport_z),
                "prototype_rescue": bool(invariant_prototype_rescue),
                "prototype_margin": float(invariant_prototype_margin),
                "prototype_policy_gate": bool(
                    invariant_prototype_policy_gate),
                "prototypes_per_class": int(
                    invariant_prototypes_per_class),
                "release_stability_only": bool(
                    invariant_release_stability_only),
            },
            "invariant_release_contrast_guard": {
            "enabled": bool(invariant_release_contrast_guard),
            "causal_contrast": invariant_release_contrast,
            "threshold": float(invariant_release_contrast_threshold),
            "configured_cap": float(invariant_release_cap),
            "effective_cap": float(effective_invariant_release_cap),
                "minimum_cap": float(invariant_release_contrast_min_cap),
                "hard_fallback": bool(
                    invariant_release_contrast_hard_fallback),
                "legacy_path_fallback_active": bool(
                    high_contrast_legacy_fallback),
                "clean_start_reconstruction_active": bool(
                    high_contrast_reconstruction_active),
                "clean_start_reconstruction_release_cap": float(
                    invariant_reconstruction_release_cap),
                "stable_private_amplitude_restore": bool(
                    invariant_stable_private_amplitude),
                "continuous_causal_label_weight": bool(
                    invariant_continuous_label_weight),
            "attack_identity_used": False,
            "target_class_used": False,
            "trigger_template_used": False,
        },
        "temporal_quarantine": {
            "enabled": temporal_quarantine_enabled,
            "round": temporal_round if temporal_quarantine_enabled else None,
            "decay": (float(temporal_quarantine_decay)
                      if temporal_quarantine_enabled else None),
            "warmup": (int(temporal_quarantine_warmup)
                       if temporal_quarantine_enabled else None),
            "source": state.get("temporal_quarantine_source"),
            "evidence": (state.get("temporal_causal_rank_evidence")
                         if temporal_quarantine_enabled else None),
            "change_count": len(state.get(
                "temporal_quarantine_changes", [])),
            "attack_identity_used": False,
            "target_class_used": False,
            "split_flag_used": False,
        },
        "ci_adaptation": state.get("ci_adaptation"),
        "repair_warmup_rounds": int(repair_warmup),
        "repair_warmup_complete": bool(warmup_complete),
        "repair_records": repair_records,
        "repaired_clients_in_aggregation": repaired_indices,
        "recursive_causal_firewall": bool(
            repair_mode in ("cms", "cms_layer") and quarantined),
        "invariant_nonrecursive_release": bool(
            invariant_nonrecursive_release and invariant_shadow_guarded
            and invariant_safe_core_high_heterogeneity and quarantined),
        "release_repaired_stable_component": bool(
            repair_mode in ("cms_release", "cms_layer_release")
            and quarantined),
        "parameter_soft_candidates": sorted(soft_parameter_candidates),
        "causal_pardon_candidates": sorted(pardon_candidates),
        "causal_pardon_alignment": {
            str(idx): float(value)
            for idx, value in pardon_alignment.items()},
        "causal_pardon_weight_overrides": {
            str(idx): float(value)
            for idx, value in pardon_weight_overrides.items()},
        "causal_pardon_risk_streak": list(
            state.get("causal_pardon_risk_streak", [])),
        "causal_pardon_safe_streak": list(
            state.get("causal_pardon_safe_streak", [])),
        "causal_pardon_released_clients": sorted(pardoned_clients),
        "causal_pardon_policy": {
            "mode": ("causal_trigger_block"
                     if causal_trigger_block_mode else "causal_pardon"),
            "weight_floor": float(pardon_floor),
            "weight_ceiling": float(pardon_ceiling),
            "variance_keep": float(pardon_variance_keep),
            "hard_streak": int(pardon_hard_streak),
            "release_streak": int(pardon_release_streak),
        },
        "estimand": (
            "Delta_G Delta_theta and Delta_A Delta_X Delta_theta "
            "on each current local graph"),
    }
    state["round"] = int(state.get("round", 0)) + 1
    state["last"] = record
    state.setdefault("history", []).append(copy.deepcopy(record))
    return aggregated_model


class LearnedGraphInterventionPolicy(nn.Module):
    """A learned attributed-graph intervention shared by all clients.

    The policy does not choose from trigger templates. A soft node gate
    replaces attributes by a local robust control and cuts incident messages,
    representing one learned ``do(G=(A,X))`` under a fixed average budget.
    """

    def __init__(self, feature_dim, hidden=32, budget=0.2, temperature=1.0):
        super().__init__()
        self.budget = float(budget)
        self.temperature = float(temperature)
        self.net = nn.Sequential(
            nn.Linear(int(feature_dim) + 1, hidden), nn.ReLU(),
            nn.Linear(hidden, 1))

    def forward(self, x, adj):
        degree = adj.sum(dim=1, keepdim=True)
        degree = degree / degree.max().clamp(min=1.0)
        raw = torch.sigmoid(
            self.net(torch.cat([x, degree], dim=1)).squeeze(1)
            / max(self.temperature, 1e-4))
        scale = (self.budget / raw.mean().clamp(min=1e-6)).clamp(max=1.0)
        gate = raw * scale
        reference = x.median(dim=0).values
        x_do = x + gate[:, None] * (reference[None, :] - x)
        keep = (1.0 - gate[:, None]) * (1.0 - gate[None, :])
        adj_do = adj * keep
        return x_do, adj_do, gate


def contrastive_interventional_shadow_train(
        reference_model, safe_model, client_models, clients,
        suspicious_indices, benign_indices, state, budget=0.2,
        policy_steps=4, model_steps=1, lr_policy=0.005, lr_model=0.001,
        consistency_weight=1.0, local_focus=False, focus_fraction=0.25,
        max_model_steps=None, residual_ratio_threshold=5.0,
        seed=0, device="cpu"):
    """Learn a target-free do(G) separating suspicious update mechanisms.

    The policy maximizes the contrast between Delta_G Delta_theta responses
    of the provisional group and the shadow-benign group.  The safe model is
    then trained only on shadow-benign client graphs to make its predictions
    invariant to that learned intervention.
    """
    suspicious = sorted(int(value) for value in suspicious_indices)
    benign = sorted(int(value) for value in benign_indices)
    if not suspicious or not benign:
        return safe_model, {"training_ready": False}
    feature_dim = int(clients[0].x.shape[1])
    policy = state.get("shadow_ci_policy")
    if policy is None:
        policy = LearnedGraphInterventionPolicy(
            feature_dim, budget=budget).to(device)
        state["shadow_ci_policy"] = policy
    else:
        policy = policy.to(device)
        policy.budget = float(budget)

    all_models = [reference_model] + list(client_models)
    flags = [[parameter.requires_grad for parameter in model.parameters()]
             for model in all_models]
    for model in all_models:
        model.to(device).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    optimizer_policy = torch.optim.Adam(
        policy.parameters(), lr=float(lr_policy), weight_decay=1e-4)
    policy.train()
    last_suspicious = 0.0
    last_benign = 0.0
    for step in range(max(int(policy_steps), 1)):
        effects = []
        gate_entropies = []
        for position, (client, client_model) in enumerate(
                zip(clients, client_models)):
            x = client.x.to(device)
            adj = client.adj.to(device)
            train_idx = torch.as_tensor(
                client.train_idx, dtype=torch.long, device=device)
            if train_idx.numel() > 96:
                generator = torch.Generator(device="cpu")
                generator.manual_seed(
                    int(seed) + step * 104729 + position * 1009)
                chosen = torch.randperm(
                    train_idx.numel(), generator=generator)[:96].to(device)
                train_idx = train_idx[chosen]
            with torch.no_grad():
                reference_factual = reference_model(x, adj)
                client_factual = client_model(x, adj)
                factual_update = client_factual - reference_factual
            x_do, adj_do, gate = policy(x, adj)
            reference_do = reference_model(x_do, adj_do)
            client_do = client_model(x_do, adj_do)
            interaction = ((client_do - reference_do) - factual_update)
            node_effect = interaction[train_idx].norm(dim=1)
            if local_focus:
                # A graph trigger is sparse inside a client graph.  Optimizing
                # the whole-graph mean dilutes its mechanism, especially in
                # the paper-split local induced graphs.  A target-free CVaR
                # objective learns from the most intervention-sensitive local
                # neighborhoods without naming a trigger or outcome class.
                tail_count = max(1, int(np.ceil(
                    float(focus_fraction) * node_effect.numel())))
                effects.append(torch.topk(
                    node_effect, k=tail_count, largest=True).values.mean())
            else:
                effects.append(node_effect.mean())
            gate_entropies.append((gate * (1.0 - gate)).mean())
        effects = torch.stack(effects)
        suspicious_effect = effects[suspicious].mean()
        benign_effect = effects[benign].mean()
        contrast = ((suspicious_effect - benign_effect)
                    / benign_effect.detach().clamp(min=1e-4))
        entropy = torch.stack(gate_entropies).mean()
        loss = -contrast + 0.01 * entropy
        optimizer_policy.zero_grad()
        loss.backward()
        optimizer_policy.step()
        last_suspicious = float(suspicious_effect.detach())
        last_benign = float(benign_effect.detach())

    for model, model_flags in zip(all_models, flags):
        for parameter, flag in zip(model.parameters(), model_flags):
            parameter.requires_grad_(flag)
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    policy.eval()

    robust_model = copy.deepcopy(safe_model).to(device)
    optimizer_model = torch.optim.Adam(
        robust_model.parameters(), lr=float(lr_model), weight_decay=5e-4)
    ce_history = []
    consistency_history = []
    suspicious_cf_history = []
    minimum_model_steps = max(int(model_steps), 1)
    maximum_model_steps = max(
        minimum_model_steps,
        int(max_model_steps) if max_model_steps is not None
        else minimum_model_steps)
    steps_completed = 0
    residual_ratio_history = []
    for _ in range(maximum_model_steps):
        robust_model.train()
        losses = []
        ces = []
        consistencies = []
        suspicious_counterfactuals = []
        for idx in benign:
            client = clients[idx]
            x = client.x.to(device)
            adj = client.adj.to(device)
            y = client.y.to(device)
            train_idx = torch.as_tensor(
                client.train_idx, dtype=torch.long, device=device)
            with torch.no_grad():
                x_do, adj_do, gate = policy(x, adj)
            factual_logits = robust_model(x, adj)
            do_logits = robust_model(x_do, adj_do)
            ce = F.cross_entropy(factual_logits[train_idx], y[train_idx])
            teacher = F.softmax(factual_logits.detach(), dim=1)
            if local_focus:
                adjacency_mass = adj.detach().abs()
                neighbor_gate = (adjacency_mass @ gate
                                 / adjacency_mass.sum(dim=1).clamp(min=1e-8))
                exposure = torch.maximum(gate, neighbor_gate).detach()
                node_kl = F.kl_div(
                    F.log_softmax(do_logits, dim=1), teacher,
                    reduction="none").sum(dim=1)
                consistency = ((node_kl * exposure).sum()
                               / exposure.sum().clamp(min=1e-8))
            else:
                consistency = F.kl_div(
                    F.log_softmax(do_logits, dim=1), teacher,
                    reduction="batchmean")
            losses.append(ce + float(consistency_weight) * consistency)
            ces.append(ce.detach())
            consistencies.append(consistency.detach())
        # On provisional graphs, labels and factual predictions may both be
        # poisoned.  Reverse the distillation direction: the prediction after
        # the learned path-cutting do(G) is the counterfactual teacher for the
        # factual graph.  No provisional-client label is consumed.
        for idx in suspicious:
            client = clients[idx]
            x = client.x.to(device)
            adj = client.adj.to(device)
            with torch.no_grad():
                x_do, adj_do, gate = policy(x, adj)
            factual_logits = robust_model(x, adj)
            do_logits = robust_model(x_do, adj_do)
            counterfactual_teacher = F.softmax(do_logits.detach(), dim=1)
            if local_focus:
                adjacency_mass = adj.detach().abs()
                neighbor_gate = (adjacency_mass @ gate
                                 / adjacency_mass.sum(dim=1).clamp(min=1e-8))
                exposure = torch.maximum(gate, neighbor_gate).detach()
                node_kl = F.kl_div(
                    F.log_softmax(factual_logits, dim=1),
                    counterfactual_teacher, reduction="none").sum(dim=1)
                counterfactual_loss = ((node_kl * exposure).sum()
                                       / exposure.sum().clamp(min=1e-8))
            else:
                counterfactual_loss = F.kl_div(
                    F.log_softmax(factual_logits, dim=1),
                    counterfactual_teacher, reduction="batchmean")
            losses.append(float(consistency_weight) * counterfactual_loss)
            suspicious_counterfactuals.append(
                counterfactual_loss.detach())
        loss = torch.stack(losses).mean()
        optimizer_model.zero_grad()
        loss.backward()
        optimizer_model.step()
        ce_history.append(float(torch.stack(ces).mean()))
        consistency_history.append(float(
            torch.stack(consistencies).mean()))
        suspicious_cf_history.append(float(torch.stack(
            suspicious_counterfactuals).mean()))
        steps_completed += 1
        residual_ratio = (
            suspicious_cf_history[-1]
            / max(consistency_history[-1], 1e-4))
        residual_ratio_history.append(float(residual_ratio))
        if (local_focus and steps_completed >= minimum_model_steps
                and residual_ratio
                <= float(residual_ratio_threshold)):
            break
    for parameter in policy.parameters():
        parameter.requires_grad_(True)
    if hasattr(robust_model, "_feats"):
        robust_model._feats = None
    return robust_model, {
        "training_ready": True,
        "policy_steps": int(policy_steps),
        "model_steps": int(steps_completed),
        "minimum_model_steps": int(minimum_model_steps),
        "maximum_model_steps": int(maximum_model_steps),
        "residual_ratio_threshold": float(residual_ratio_threshold),
        "final_suspicious_to_benign_residual_ratio": (
            residual_ratio_history[-1]),
        "intervention_budget": float(budget),
        "suspicious_interaction": last_suspicious,
        "benign_interaction": last_benign,
        "contrast_ratio": float(
            (last_suspicious - last_benign) / max(last_benign, 1e-8)),
        "mean_benign_ce": float(np.mean(ce_history)),
        "mean_interventional_consistency": float(np.mean(
            consistency_history)),
        "mean_suspicious_counterfactual_distillation": float(np.mean(
            suspicious_cf_history)),
        "counterfactual_teacher_direction": "do(G) -> factual G",
        "local_neighborhood_focus": bool(local_focus),
        "policy_tail_fraction": (float(focus_fraction)
                                 if local_focus else None),
        "distillation_weighting": (
            "learned-node-gate plus one-hop exposure"
            if local_focus else "whole local graph"),
        "target_class_used": False,
        "clean_graph_used": False,
        "trigger_template_used": False,
        "attack_identity_used": False,
        "estimand": (
            "contrastive provisional-vs-shadow-benign "
            "Delta_G Delta_theta response"),
    }


def train_stable_causal_message_gate(
        reference_model, safe_model, client_models, clients,
        suspicious_indices, benign_indices, state, budget=0.2,
        policy_steps=4, model_steps=2, lr_policy=0.005, lr_model=0.001,
        seed=0, device="cpu"):
    """Block mediators identified by target-free ``do(G) x do(theta)``.

    The contrastive intervention policy localizes graph variables whose
    treatment changes suspicious update mechanisms more than trusted client
    mechanisms.  Those variables provide negative (not class) supervision for
    the message gate.  Trusted client environments provide sufficiency,
    class-conditional invariance, and interventional stability constraints.
    Suspicious labels are never consumed.
    """
    suspicious = sorted(int(value) for value in suspicious_indices)
    benign = sorted(int(value) for value in benign_indices)
    if not suspicious or not benign:
        return safe_model, {"training_ready": False}
    if not hasattr(safe_model, "causal_gate"):
        raise TypeError("stable_gate requires CausalGateNodeGCN")

    feature_dim = int(clients[0].x.shape[1])
    policy = state.get("stable_gate_policy")
    if policy is None:
        policy = LearnedGraphInterventionPolicy(
            feature_dim, budget=budget).to(device)
        state["stable_gate_policy"] = policy
    else:
        policy = policy.to(device)
        policy.budget = float(budget)

    all_models = [reference_model] + list(client_models)
    flags = [[parameter.requires_grad for parameter in model.parameters()]
             for model in all_models]
    for model in all_models:
        model.to(device).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    optimizer_policy = torch.optim.Adam(
        policy.parameters(), lr=float(lr_policy), weight_decay=1e-4)
    policy.train()
    last_suspicious = 0.0
    last_benign = 0.0
    for step in range(max(int(policy_steps), 1)):
        effects = []
        entropies = []
        for position, (client, client_model) in enumerate(
                zip(clients, client_models)):
            x = client.x.to(device)
            adj = client.adj.to(device)
            train_idx = torch.as_tensor(
                client.train_idx, dtype=torch.long, device=device)
            if train_idx.numel() > 96:
                generator = torch.Generator(device="cpu")
                generator.manual_seed(
                    int(seed) + step * 104729 + position * 1009)
                chosen = torch.randperm(
                    train_idx.numel(), generator=generator)[:96].to(device)
                train_idx = train_idx[chosen]
            with torch.no_grad():
                factual_update = (client_model(x, adj)
                                  - reference_model(x, adj))
            x_do, adj_do, mediator_gate = policy(x, adj)
            do_update = (client_model(x_do, adj_do)
                         - reference_model(x_do, adj_do))
            interaction = do_update - factual_update
            effects.append(interaction[train_idx].norm(dim=1).mean())
            entropies.append(
                (mediator_gate * (1.0 - mediator_gate)).mean())
        effects = torch.stack(effects)
        suspicious_effect = effects[suspicious].mean()
        benign_effect = effects[benign].mean()
        contrast = ((suspicious_effect - benign_effect)
                    / benign_effect.detach().clamp(min=1e-4))
        loss_policy = (-contrast
                       + 0.01 * torch.stack(entropies).mean())
        optimizer_policy.zero_grad()
        loss_policy.backward()
        optimizer_policy.step()
        last_suspicious = float(suspicious_effect.detach())
        last_benign = float(benign_effect.detach())

    for model, model_flags in zip(all_models, flags):
        for parameter, flag in zip(model.parameters(), model_flags):
            parameter.requires_grad_(flag)
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    policy.eval()

    robust_model = copy.deepcopy(safe_model).to(device)
    optimizer_model = torch.optim.Adam(
        robust_model.parameters(), lr=float(lr_model), weight_decay=5e-4)
    ce_history = []
    consistency_history = []
    overlap_history = []
    retention_history = []
    conditional_variance_history = []
    benign_gate_history = []
    suspicious_gate_history = []
    for _ in range(max(int(model_steps), 1)):
        robust_model.train()
        factual_risks = []
        do_risks = []
        consistencies = []
        retentions = []
        class_environment_risks = {}
        benign_gate_means = []
        for idx in benign:
            client = clients[idx]
            x = client.x.to(device)
            adj = client.adj.to(device)
            y = client.y.to(device)
            train_idx = torch.as_tensor(
                client.train_idx, dtype=torch.long, device=device)
            with torch.no_grad():
                x_do, adj_do, _ = policy(x, adj)
            factual_logits = robust_model(x, adj)
            factual_gate = robust_model._causal_gate
            do_logits = robust_model(x_do, adj_do)
            factual_risk = F.cross_entropy(
                factual_logits[train_idx], y[train_idx])
            do_risk = F.cross_entropy(do_logits[train_idx], y[train_idx])
            teacher = F.softmax(factual_logits[train_idx].detach(), dim=1)
            consistency = F.kl_div(
                F.log_softmax(do_logits[train_idx], dim=1), teacher,
                reduction="batchmean")
            retention = (factual_gate.mean() - 0.85).pow(2)
            factual_risks.append(factual_risk)
            do_risks.append(do_risk)
            consistencies.append(consistency)
            retentions.append(retention)
            benign_gate_means.append(factual_gate.mean())
            labels = y[train_idx]
            for label in torch.unique(labels).tolist():
                mask = labels == int(label)
                class_risk = F.cross_entropy(
                    factual_logits[train_idx][mask], labels[mask])
                class_environment_risks.setdefault(
                    int(label), []).append(class_risk)

        overlaps = []
        suspicious_gate_means = []
        # The provisional environments contribute only mediator localization;
        # neither their observed labels nor a guessed target class is used.
        for idx in suspicious:
            client = clients[idx]
            x = client.x.to(device)
            adj = client.adj.to(device)
            with torch.no_grad():
                _, _, mediator_gate = policy(x, adj)
            robust_model(x, adj)
            causal_gate = robust_model._causal_gate
            overlap = ((causal_gate * mediator_gate).sum()
                       / mediator_gate.sum().clamp(min=1e-6))
            overlaps.append(overlap)
            suspicious_gate_means.append(causal_gate.mean())

        conditional_variances = []
        for risks in class_environment_risks.values():
            if len(risks) > 1:
                conditional_variances.append(
                    torch.stack(risks).var(unbiased=False))
        conditional_variance = (
            torch.stack(conditional_variances).mean()
            if conditional_variances
            else torch.zeros((), device=device))
        mean_factual = torch.stack(factual_risks).mean()
        mean_do = torch.stack(do_risks).mean()
        mean_consistency = torch.stack(consistencies).mean()
        mean_retention = torch.stack(retentions).mean()
        mean_overlap = torch.stack(overlaps).mean()
        loss_model = (mean_factual + 0.25 * mean_do
                      + 0.5 * mean_consistency
                      + 0.5 * conditional_variance
                      + 1.0 * mean_overlap
                      + 2.0 * mean_retention)
        optimizer_model.zero_grad()
        loss_model.backward()
        optimizer_model.step()
        ce_history.append(float(mean_factual.detach()))
        consistency_history.append(float(mean_consistency.detach()))
        overlap_history.append(float(mean_overlap.detach()))
        retention_history.append(float(mean_retention.detach()))
        conditional_variance_history.append(
            float(conditional_variance.detach()))
        benign_gate_history.append(float(
            torch.stack(benign_gate_means).mean().detach()))
        suspicious_gate_history.append(float(
            torch.stack(suspicious_gate_means).mean().detach()))

    for parameter in policy.parameters():
        parameter.requires_grad_(True)
    robust_model.eval()
    if hasattr(robust_model, "_feats"):
        robust_model._feats = None
    if hasattr(robust_model, "_causal_gate"):
        robust_model._causal_gate = None
    return robust_model, {
        "training_ready": True,
        "policy_steps": int(policy_steps),
        "model_steps": int(model_steps),
        "intervention_budget": float(budget),
        "suspicious_interaction": last_suspicious,
        "benign_interaction": last_benign,
        "contrast_ratio": float(
            (last_suspicious - last_benign) / max(last_benign, 1e-8)),
        "mean_benign_ce": float(np.mean(ce_history)),
        "mean_interventional_consistency": float(np.mean(
            consistency_history)),
        "mean_mediator_gate_overlap": float(np.mean(overlap_history)),
        "mean_retention_penalty": float(np.mean(retention_history)),
        "mean_class_conditional_risk_variance": float(np.mean(
            conditional_variance_history)),
        "mean_benign_message_gate": float(np.mean(benign_gate_history)),
        "mean_suspicious_message_gate": float(np.mean(
            suspicious_gate_history)),
        "suspicious_labels_used": False,
        "target_class_used": False,
        "complete_clean_graph_used": False,
        "trigger_template_used": False,
        "attack_identity_used": False,
        "training_execution": "client-local graphs in federated simulator",
        "estimand": (
            "contrastive mediator Delta_G Delta_theta effect with "
            "class-conditional environment invariance"),
    }


class NodeClient:
    def __init__(self, cid, x, adj, y, train_idx, malicious=False, attack=None,
                 attack_idx=None, rng_seed=0, trigger=None, backdoor=None,
                 clean_graph=None):
        self.cid = cid
        self.malicious = malicious
        self.trigger = trigger
        self.backdoor = backdoor
        self.clean_graph = clean_graph
        self.attack_idx = (np.asarray(attack_idx, dtype=np.int64)
                           if attack_idx is not None
                           else np.empty(0, dtype=np.int64))
        self.clean_y = y.clone()
        self.clean_train_idx = np.asarray(train_idx, dtype=np.int64).copy()
        self.x = x
        self.adj = adj
        self.y = y
        self.train_idx = np.asarray(train_idx, dtype=np.int64)
        if (malicious and backdoor is None and attack is not None
                and attack_idx is not None and len(attack_idx) > 0):
            self.x, self.adj, self.y, _ = attack.poison_graph(
                x, adj, y, attack_idx, trigger=trigger, rng_seed=rng_seed)
            self.train_idx = np.concatenate(
                [self.train_idx, np.asarray(attack_idx, dtype=np.int64)])

    def local_train(self, global_model, epochs, lr, device="cpu",
                    optimizer_name="adam", training_protocol="current",
                    training_seed=None):
        author_protocol = training_protocol == "fedtd-author"
        return train_node_model(global_model, self.x, self.adj, self.train_idx,
                                self.y, epochs, lr, device,
                                optimizer_name=optimizer_name,
                                val_idx=getattr(self, "author_val_idx", None),
                                select_best_on_val=author_protocol,
                                training_seed=(training_seed
                                               if author_protocol else None))


