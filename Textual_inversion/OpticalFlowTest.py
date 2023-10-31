import torch
from torchvision.models.optical_flow import raft_large

F1_styled = torch.randn(size=(2, 3, 512, 512))
F2_styled = torch.randn(size=(2, 3, 512, 512))
F1 = torch.randn(size=(2, 3, 512, 512))
F2 = torch.randn(size=(2, 3, 512, 512))


Of_model = raft_large(pretrained=True, progress=False)
Of_model = Of_model.eval()

Of_Styled = Of_model(F1_styled, F2_styled)
Of_Original = Of_model(F1, F2)


print(len(Of_Original))
print(Of_Original[-1].shape)