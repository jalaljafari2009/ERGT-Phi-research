import copy
import sys
from dataclasses import asdict
import pytest
import torch
from ergt_phi.fixtures import example_batch, model_config, seed_all, training_step
from ergt_phi.native_steps import SteppedNative, SteppedSubstrate
from ergt_phi.model import PhaseConfig, ResearchModel
from ergt_phi.checkpoint import rng_state
from ergt_reviewer.native_solver import NativeGeometricBoundaryModel, hard_solutions_from_outputs, native_geometric_training_loss
from ergt_reviewer.physics_core import PhysicsNativeSubstrate


def exact(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0, equal_nan=False)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            exact(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            exact(x, y)
    else:
        assert a == b


def models(hops=1, unsupported=False, pad=None):
    batch, tokenizer = example_batch(hops, unsupported, pad)
    seed_all()
    original = NativeGeometricBoundaryModel(model_config(tokenizer))
    stepped = SteppedNative(original.config)
    stepped.load_state_dict(original.state_dict())
    return batch, original, stepped


@pytest.mark.parametrize("hops,unsupported,pad", [(1, False, None), (2, False, None), (1, True, 64)])
def test_all_outputs_and_native_decisions(hops, unsupported, pad):
    batch, original, stepped = models(hops, unsupported, pad)
    with torch.no_grad():
        a = original(**batch.model_inputs())
        b = stepped(**batch.model_inputs())
    exact(a, b)
    for x, y in zip(hard_solutions_from_outputs(a, original.config), hard_solutions_from_outputs(b, stepped.config)):
        exact(asdict(x), asdict(y))


@pytest.mark.parametrize("intervention", ["no_phi", "no_memory_geometry", "no_transport", "shuffled_geometry", "random_geometry", "only_world_0", "no_action", "no_boundary", "no_terminal_mass", "no_cone"])
def test_intervention_parity(intervention):
    batch, original, stepped = models()
    with torch.no_grad():
        a = original.forward_with_intervention(**batch.model_inputs(), intervention=intervention, intervention_seed=917)
        b = stepped.forward_with_intervention(**batch.model_inputs(), intervention=intervention, intervention_seed=917)
    exact(a, b)


def test_intermediate_field_steps_and_initial_charge():
    batch, original, stepped = models()
    semantic, identity = original.raw_input_adapter(**batch.model_inputs())
    snapshots = []
    code = PhysicsNativeSubstrate.forward.__code__
    loop_line = next(i for i, line in enumerate(__import__("inspect").getsourcelines(PhysicsNativeSubstrate.forward)[0], __import__("inspect").getsourcelines(PhysicsNativeSubstrate.forward)[1]) if "for _ in range(self.config.field_steps)" in line)
    def tracer(frame, event, arg):
        if frame.f_code is code and event == "line" and frame.f_lineno == loop_line:
            local = frame.f_locals
            current = local["final"] or {"psi": local["psi"], "typed_payload_charge": local["typed_payload_charge"]}
            snapshots.append({k: v.detach().clone() for k, v in current.items() if isinstance(v, torch.Tensor)})
        return tracer
    previous = sys.gettrace()
    with torch.no_grad():
        try:
            sys.settrace(tracer)
            original.substrate(semantic, batch.base.attention_mask, identity_fibre=identity)
        finally:
            sys.settrace(previous)
        trace = []
        stepped.substrate(semantic, batch.base.attention_mask, identity_fibre=identity, trace=trace)
    assert len(snapshots) == len(trace) == original.config.field_steps + 1
    exact(snapshots, trace)


@pytest.mark.parametrize("phase", [PhaseConfig(), PhaseConfig(phase_mode="disabled", coupling=.02), PhaseConfig(phase_mode="coupled", coupling=0)])
def test_bypass_forward_gradient_optimizer_rng(phase):
    batch, original, _ = models()
    seed_all(911)
    before = torch.get_rng_state().clone()
    wrapped = ResearchModel(copy.deepcopy(original), phase)
    assert torch.equal(before, torch.get_rng_state())
    left = torch.optim.AdamW(original.parameters(), lr=.0025)
    right = torch.optim.AdamW(wrapped.parameters(), lr=.0025)
    baseline_rng = torch.get_rng_state().clone()
    loss_a = training_step(original, left, batch)
    after_a = torch.get_rng_state().clone()
    torch.set_rng_state(baseline_rng)
    loss_b = training_step(wrapped, right, batch)
    assert torch.equal(after_a, torch.get_rng_state())
    exact(loss_a, loss_b)
    exact(original.state_dict(), wrapped.baseline.state_dict())
    exact(left.state_dict(), right.state_dict())
    for a, b in zip(original.parameters(), wrapped.parameters()):
        exact(a.grad, b.grad)


def test_refactor_training_gradient_optimizer_parity():
    batch, original, stepped = models(hops=2)
    left = torch.optim.AdamW(original.parameters(), lr=.0025)
    right = torch.optim.AdamW(stepped.parameters(), lr=.0025)
    exact(training_step(original, left, batch), training_step(stepped, right, batch))
    exact(original.state_dict(), stepped.state_dict())
    exact(left.state_dict(), right.state_dict())
    for a, b in zip(original.parameters(), stepped.parameters()):
        exact(a.grad, b.grad)


def test_model_construction_preserves_rng_and_parameter_initialization():
    _, tokenizer = example_batch()
    seed_all()
    original = NativeGeometricBoundaryModel(model_config(tokenizer))
    expected = torch.get_rng_state().clone()
    seed_all()
    stepped = SteppedNative(model_config(tokenizer))
    assert torch.equal(expected, torch.get_rng_state())
    exact(original.state_dict(), stepped.state_dict())


def test_probe_stops_before_answer_solving():
    batch, _, stepped = models()
    semantic, identity = stepped.raw_input_adapter(**batch.model_inputs())
    with torch.no_grad():
        state = stepped.substrate(semantic, batch.base.attention_mask, identity_fibre=identity)
        result = stepped.probe_native_proposals(state, batch.base.attention_mask)
    assert "native_answer_logits" not in result
    assert "event_world_closure_soft_gate" not in result
    assert "event_relation_logits" in result


def test_future_activation_is_explicitly_unavailable():
    batch, original, _ = models()
    with pytest.raises(NotImplementedError):
        ResearchModel(original, PhaseConfig(phase_mode="coupled", coupling=.02))(**batch.model_inputs())


def test_wrapper_works_with_original_evaluator_and_architecture_audit():
    from ergt_reviewer.native_solver import architecture_contract
    from ergt_reviewer.evaluation_v9 import evaluate_native
    batch, original, _ = models()
    _, tokenizer = example_batch()
    wrapped = ResearchModel(copy.deepcopy(original))
    assert all(architecture_contract(wrapped).values())
    arguments = dict(device=torch.device("cpu"), batch_size=2, cohort="m0_fixture")
    baseline_metrics, baseline_rows, _ = evaluate_native(original, batch.examples, tokenizer, **arguments)
    metrics, rows, _ = evaluate_native(wrapped, batch.examples, tokenizer, **arguments)
    assert metrics["accuracy"] == baseline_metrics["accuracy"]
    assert rows == baseline_rows


def test_paper_configuration_forward_parity():
    batch, _ = example_batch()
    from ergt_reviewer.fair_data_v9 import protocol_tokenizer
    from ergt_reviewer.matched_data import collate_matched_topology_examples
    tokenizer = protocol_tokenizer(97)
    batch = collate_matched_topology_examples(batch.examples, tokenizer)
    seed_all()
    original = NativeGeometricBoundaryModel(model_config(tokenizer, paper=True))
    stepped = SteppedNative(original.config)
    stepped.load_state_dict(original.state_dict())
    with torch.no_grad():
        exact(original(**batch.model_inputs()), stepped(**batch.model_inputs()))
