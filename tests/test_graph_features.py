import numpy as np
import pytest

from mcdrp.features.graphs import ATOM_FEATURE_DIM, smiles_to_graph


def test_smiles_to_graph_builds_valid_graph() -> None:
    graph = smiles_to_graph("CCO")

    assert graph.valid
    assert graph.n_nodes == 3
    assert graph.node_features.shape == (3, ATOM_FEATURE_DIM)
    assert graph.adjacency.shape == (3, 3)
    assert np.allclose(graph.adjacency.sum(axis=1), 1.0)


def test_collate_graph_batch_pads_variable_size_graphs() -> None:
    pytest.importorskip("torch", reason="B4 collate helpers require PyTorch.")
    from mcdrp.models.gnn_b4 import collate_graph_batch

    graph_a = smiles_to_graph("CCO")
    graph_b = smiles_to_graph("c1ccccc1")
    batch = collate_graph_batch(
        [
            {
                "node_features": graph_a.node_features,
                "adjacency": graph_a.adjacency,
                "cell_features": np.ones(4, dtype=np.float32),
                "target": np.float32(1.0),
            },
            {
                "node_features": graph_b.node_features,
                "adjacency": graph_b.adjacency,
                "cell_features": np.zeros(4, dtype=np.float32),
                "target": np.float32(2.0),
            },
        ]
    )

    assert batch.node_features.shape == (2, 6, ATOM_FEATURE_DIM)
    assert batch.adjacency.shape == (2, 6, 6)
    assert batch.mask.shape == (2, 6)
    assert batch.cell_features.shape == (2, 4)
    assert batch.targets.tolist() == [1.0, 2.0]


def test_collate_hybrid_graph_batch_adds_fingerprints() -> None:
    pytest.importorskip("torch", reason="B5 collate helpers require PyTorch.")
    from mcdrp.models.gnn_b5 import collate_hybrid_graph_batch

    graph_a = smiles_to_graph("CCO")
    graph_b = smiles_to_graph("c1ccccc1")
    batch = collate_hybrid_graph_batch(
        [
            {
                "node_features": graph_a.node_features,
                "adjacency": graph_a.adjacency,
                "fingerprints": np.ones(8, dtype=np.float32),
                "cell_features": np.ones(4, dtype=np.float32),
                "target": np.float32(1.0),
            },
            {
                "node_features": graph_b.node_features,
                "adjacency": graph_b.adjacency,
                "fingerprints": np.zeros(8, dtype=np.float32),
                "cell_features": np.zeros(4, dtype=np.float32),
                "target": np.float32(2.0),
            },
        ]
    )

    assert batch.node_features.shape == (2, 6, ATOM_FEATURE_DIM)
    assert batch.adjacency.shape == (2, 6, 6)
    assert batch.mask.shape == (2, 6)
    assert batch.fingerprints.shape == (2, 8)
    assert batch.cell_features.shape == (2, 4)
    assert batch.targets.tolist() == [1.0, 2.0]


@pytest.mark.parametrize("encoder_name", ["b4", "b5"])
def test_graph_encoders_ignore_padding_and_propagate_gradients(encoder_name):
    torch = pytest.importorskip("torch")
    from mcdrp.models.gnn_b4 import DrugGraphEncoder
    from mcdrp.models.gnn_b5 import AttentiveGraphEncoder

    torch.manual_seed(7)
    encoder = (
        DrugGraphEncoder(hidden_dim=8, n_layers=2, dropout=0.0)
        if encoder_name == "b4"
        else AttentiveGraphEncoder(hidden_dim=8, n_layers=2, output_dim=8, dropout=0.0)
    )
    encoder.eval()
    graph = smiles_to_graph("CCO")
    x = torch.tensor(graph.node_features)[None]
    adjacency = torch.tensor(graph.adjacency)[None]
    mask = torch.ones((1, 3))
    padded_x = torch.nn.functional.pad(x, (0, 0, 0, 4))
    padded_adjacency = torch.nn.functional.pad(adjacency, (0, 4, 0, 4))
    padded_mask = torch.nn.functional.pad(mask, (0, 4))
    direct = encoder(x, adjacency, mask)
    padded = encoder(padded_x, padded_adjacency, padded_mask)
    torch.testing.assert_close(direct, padded)
    direct.square().sum().backward()
    gradients = [p.grad for p in encoder.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert any(g.abs().sum() > 0 for g in gradients)
