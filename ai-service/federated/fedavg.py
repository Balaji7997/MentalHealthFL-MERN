from typing import Dict, List, Optional, Tuple
import torch


def federated_averaging(
    client_updates: List[Tuple[Dict[str, torch.Tensor], int]],
    keys_to_aggregate: Optional[List[str]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Perform weighted Federated Averaging (FedAvg) across client state dicts.

    Args:
        client_updates: List of (state_dict, num_samples) tuples from clients.
        keys_to_aggregate: Optional subset of parameter keys to aggregate.
                           If None, aggregates all keys present in the updates.

    Returns:
        Aggregated global state dict on CPU in float32.
    """
    if not client_updates:
        raise ValueError("client_updates cannot be empty.")

    total_samples = sum(num_samples for _, num_samples in client_updates)
    if total_samples <= 0:
        raise ValueError("Total number of client samples must be positive.")

    first_state = client_updates[0][0]
    target_keys = keys_to_aggregate if keys_to_aggregate is not None else list(first_state.keys())

    # Verify all clients share the target keys
    for idx, (state_dict, _) in enumerate(client_updates):
        missing = [k for k in target_keys if k not in state_dict]
        if missing:
            raise KeyError(f"Client index {idx} is missing keys: {missing[:5]}")

    aggregated_state: Dict[str, torch.Tensor] = {}

    for key in target_keys:
        weighted_sum = None
        for state_dict, num_samples in client_updates:
            weight = float(num_samples) / float(total_samples)
            tensor = state_dict[key].detach().to(device="cpu", dtype=torch.float32)
            contribution = tensor * weight
            if weighted_sum is None:
                weighted_sum = contribution
            else:
                weighted_sum = weighted_sum + contribution
        aggregated_state[key] = weighted_sum

    return aggregated_state


def compute_state_dict_stats(state_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """Calculate exact parameter count and communication size (MB) of a state dict."""
    num_params = sum(int(t.numel()) for t in state_dict.values())
    total_bytes = sum(int(t.numel() * t.element_size()) for t in state_dict.values())
    size_mb = total_bytes / (1024.0 * 1024.0)
    return {
        "num_tensors": len(state_dict),
        "num_parameters": num_params,
        "size_mb": round(size_mb, 4),
    }


def _run_self_test() -> None:
    """Verify weighted FedAvg math on simple tensors."""
    client_a = {"lora_weight": torch.tensor([1.0, 2.0]), "classifier_bias": torch.tensor([0.0])}
    client_b = {"lora_weight": torch.tensor([5.0, 6.0]), "classifier_bias": torch.tensor([4.0])}

    # Weights: client_a = 100/400 = 0.25, client_b = 300/400 = 0.75
    # Expected lora_weight = 0.25*[1,2] + 0.75*[5,6] = [4.0, 5.0]
    # Expected classifier_bias = 0.25*[0] + 0.75*[4] = [3.0]
    global_state = federated_averaging([(client_a, 100), (client_b, 300)])

    assert torch.allclose(global_state["lora_weight"], torch.tensor([4.0, 5.0]))
    assert torch.allclose(global_state["classifier_bias"], torch.tensor([3.0]))

    stats = compute_state_dict_stats(global_state)
    print("FedAvg self-test PASSED!")
    print("Aggregated lora_weight:    ", global_state["lora_weight"].tolist())
    print("Aggregated classifier_bias:", global_state["classifier_bias"].tolist())
    print("Test payload stats:        ", stats)


if __name__ == "__main__":
    _run_self_test()