import torch

from hifi_gan_bwe import criteria


def test_content_criteria() -> None:
    crit = criteria.ContentCriteria()
    y_true = torch.zeros([2, 1, 8000])
    y_pred = torch.ones_like(y_true)
    loss = crit(y_pred, y_true)
    assert loss.shape == ()
