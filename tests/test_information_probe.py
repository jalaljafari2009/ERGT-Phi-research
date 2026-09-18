import torch
from ergt_phi.information_probe import LinearRelationProbe,history_pair_features,probe_metrics


def test_history_pair_features_keeps_relation_labels_after_forward_features():
    records=[{'history':torch.arange(2*4*3,dtype=torch.float32).reshape(2,4,3),'events':torch.tensor([[1,2],[2,3]]),'labels':torch.tensor([0,2])}]
    x,y=history_pair_features(records,[0],'all')
    assert x.shape==(2,24) and y.tolist()==[0,2]
    torch.testing.assert_close(x[0,:3],records[0]['history'][0,1])
    torch.testing.assert_close(x[0,12:15],records[0]['history'][1,1])


def test_linear_probe_metrics_and_shapes():
    model=LinearRelationProbe(8)
    x=torch.randn(5,8); y=torch.tensor([0,1,2,0,1])
    assert probe_metrics(model,x,y)['events']==5
